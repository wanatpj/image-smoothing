#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda.h"
#include "filter-noise.h"

using namespace std;

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction cuSobelFilterFn;
CUfunction cuBlurFilterFn;
CUfunction cuMixChannelsFn;
CUfunction cuSmoothenFn;
CUfunction cuParallelMaxFn;
CUfunction cuDivideAllFn;
CUdeviceptr d_channels[3], d_tmp_channels[3], d_gradient, d_uchar_channels[3];
CUresult cu_result;

double* double_channels[3];
int dimY = -1;  // first dimension
int dimX = -1;  // second dimension

static void cuda_init() {
  bool failed = false;
  failed |= cuInit(0) != CUDA_SUCCESS;
  failed |= cuDeviceGet(&cuDevice, 0) != CUDA_SUCCESS;
  failed |= cuCtxCreate(&cuContext, 0, cuDevice) != CUDA_SUCCESS;
  failed |= cuModuleLoad(&cuModule, "filter-noise.ptx") != CUDA_SUCCESS;
  failed |= cuModuleGetFunction(&cuSobelFilterFn, cuModule, "sobel_filter") != CUDA_SUCCESS;
  failed |= cuModuleGetFunction(&cuBlurFilterFn, cuModule, "blur_filter") != CUDA_SUCCESS;
  failed |= cuModuleGetFunction(&cuMixChannelsFn, cuModule, "mix_channels") != CUDA_SUCCESS;
  failed |= cuModuleGetFunction(&cuSmoothenFn, cuModule, "smoothen") != CUDA_SUCCESS;
  failed |= cuModuleGetFunction(&cuParallelMaxFn, cuModule, "parallel_max") != CUDA_SUCCESS;
  failed |= cuModuleGetFunction(&cuDivideAllFn, cuModule, "divide_all") != CUDA_SUCCESS;
  cuCtxSynchronize();
  if (failed) {
    printf ("Cuda initialization failed.\n");
    exit(EXIT_FAILURE);
  }
}

static void allocate(int length) {
  for (int i = 0; i < 3; ++i) {
    double_channels[i] = (double*)malloc(length*sizeof(double));
    cuMemAlloc(&d_channels[i], length*sizeof(double));
    cuMemAlloc(&d_tmp_channels[i], length*sizeof(double));
    cuMemAlloc(&d_uchar_channels[i], length*sizeof(unsigned char));
  }
  cuMemAlloc(&d_gradient, length*sizeof(double));
  cuCtxSynchronize();
}

static void prepare(unsigned char* char_channels[3], int dimy, int dimx) {
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < dimy * dimx; ++i) {
      double_channels[c][i] = char_channels[c][i];
    }
    cuMemcpyHtoD(d_channels[c], double_channels[c], dimy*dimx*sizeof(double));
    cuCtxSynchronize();
  }
  dimY = dimy;
  dimX = dimx;
}

static void compute_gradients() {
  // Sobel filter
  void* args[] = { NULL, &dimY, &dimX, NULL };
  for (int c = 0; c < 3; ++c) {
    args[0] = &d_channels[c];
    args[3] = &d_tmp_channels[c];
    cuLaunchKernel(cuSobelFilterFn,
        (dimX + 31)/32, (dimY + 31)/32, 1,
        32, 32, 1,
        0, 0, args, 0);
    cuCtxSynchronize();
  }
  // Mix channels
  int size = dimY*dimX;
  void* args2[] = {
      &d_tmp_channels[0],
      &d_tmp_channels[1],
      &d_tmp_channels[2],
      &size,
      &d_gradient };
  cuLaunchKernel(cuMixChannelsFn,
      (dimY*dimX + 1023)/1024, 1, 1,
      1024, 1, 1,
      0, 0, args2, 0);
  cuCtxSynchronize();
  // Blur gradient
  void* args3[] = { &d_gradient, &dimY, &dimX, &d_tmp_channels[0] };
  for (int i = 0; i < 8; ++i) {
    cuLaunchKernel(cuBlurFilterFn,
        (dimX + 31)/32, (dimY + 31)/32, 1,
        32, 32, 1,
        0, 0, args3, 0);
    cuCtxSynchronize();
    swap(args3[0], args3[3]);
  }
  // Compute max
  void* args4[] = {
      &d_gradient,
      &size,
      &d_tmp_channels[0]};
  int cnt = 0;
  do {
    cuLaunchKernel(cuParallelMaxFn,
        (size + 1023)/1024, 1, 1,
        1024, 1, 1,
        1024*sizeof(double), 0, args4, 0);
    cuCtxSynchronize();
    size = (size + 1023)/1024;
    args4[0] = &d_tmp_channels[cnt&1];
    args4[2] = &d_tmp_channels[(cnt&1)^1];
    cnt++;
  } while (size > 1);
  size = dimY*dimX;
  // Normalize
  void* args5[] = {
      &d_gradient,
      &size,
      &d_tmp_channels[(cnt&1)^1]};
  cuLaunchKernel(cuDivideAllFn,
      (size + 1023)/1024, 1, 1,
      1024, 1, 1,
      0, 0, args5, 0);
  cuCtxSynchronize();
}

static void sharpen_or_blur() {
  void* args[] = {
      NULL,
      &dimY,
      &dimX,
      &d_gradient,
      NULL};
  for (int c = 0; c < 3; ++c) {
    args[0] = &d_channels[c];
    args[4] = &d_uchar_channels[c];
    cuLaunchKernel(cuSmoothenFn,
        (dimX + 31)/32, (dimY + 31)/32, 1,
        32, 32, 1,
        0, 0, args, 0);
    cuCtxSynchronize();
  }
}

static void save_at(unsigned char* channels[3]) {
  for (int c = 0; c < 3; ++c) {
    cuMemcpyDtoH(channels[c],
        d_uchar_channels[c],
        dimY*dimX*sizeof(unsigned char));
    cuCtxSynchronize();
  }
}

static void cleanup() {
  for (int i = 0; i < 3; ++i) {
    delete [] double_channels[i];
  }
  cuCtxSynchronize();
  cuCtxDestroy(cuContext);
}

void enhance_image(unsigned char* channels[3], int dimy, int dimx) {
  cuda_init();
  allocate(dimy * dimx);
  prepare(channels, dimy, dimx);
  compute_gradients();
  sharpen_or_blur();
  save_at(channels);
  cleanup();
}

