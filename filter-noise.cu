#include "cuda.h"
extern "C" {

__device__ double sobely_kernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
__device__ double sobelx_kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__device__ double sharpen_kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
__device__ double blur_kernel[3][3] =
    {{0, 0.2, 0}, {0.2, 0.2, 0.2}, {0, 0.2, 0}};

__device__
void simple_convolve(const double* channel,
    int dimx,
    const double kernel[3][3],
    int y,
    int x,
    double * out) {
  double result = 0;
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      result += channel[(y + i) * dimx + x + j] * kernel[i + 1][j + 1];
    }
  }
  *out = result;
}

__global__
void sobel_filter(const double* in_channel,
    int dimy,
    int dimx,
    double* out_channel) {
  int idy = blockIdx.y*32 + threadIdx.y;
  int idx = blockIdx.x*32 + threadIdx.x;
  if (idy < dimy - 1 && idx < dimx - 1 && 0 < idy && 0 < idx) {
    double sobely;
    double sobelx;
    simple_convolve(in_channel,
        dimx,
        sobely_kernel,
        idy,
        idx,
        &sobely);
    simple_convolve(in_channel,
        dimx,
        sobelx_kernel,
        idy,
        idx,
        &sobelx);
    out_channel[idy * dimx + idx] = sqrtf(sobely*sobely + sobelx*sobelx);
  } else if (idy < dimy && idx < dimx) {
    out_channel[idy * dimx + idx] = 0;
  }
}

__global__
void blur_filter(const double* in_channel,
    int dimy,
    int dimx,
    double* out_channel) {
  int idy = blockIdx.y*32 + threadIdx.y;
  int idx = blockIdx.x*32 + threadIdx.x;
  if (idy < dimy - 1 && idx < dimx - 1 && 0 < idy && 0 < idx) {
    double result;
    simple_convolve(in_channel,
        dimx,
        blur_kernel,
        idy,
        idx,
        &result);
    out_channel[idy * dimx + idx] = result;
  } else if (idy < dimy && idx < dimx) {
    out_channel[idy * dimx + idx] = in_channel[idy * dimx + idx];
  }
}

__global__
void mix_channels(const double * rchannels,
    const double * gchannels,
    const double * bchannels,
    int len,
    double* out_channel) {
  int id = blockIdx.x*1024 + threadIdx.x;
  if (id < len) {
    out_channel[id] = rchannels[id] * 0.33
        + gchannels[id] * 0.34
        + bchannels[id] * 0.33;
  }
}

__global__
void smoothen(const double* in_channel,
    int dimy,
    int dimx,
    const double* edge_channel,
    unsigned char* out_channel) {
  int idy = blockIdx.y*32 + threadIdx.y;
  int idx = blockIdx.x*32 + threadIdx.x;
  if (idy >= dimy || idx >= dimx) {
    return;
  }
  double result = 0;
  if (idy < dimy - 1 && idx < dimx - 1 && 0 < idy && 0 < idx) {
    simple_convolve(in_channel,
        dimx,
        (edge_channel[idy * dimx + idx] >= 30 ? sharpen_kernel : blur_kernel),
        idy,
        idx,
        &result);
  } else {
    result = in_channel[idy * dimx + idx];
  }
  out_channel[idy * dimx + idx] =
      (unsigned char)max(0., min(255., round(result)));
}

__global__
void parallel_max(const double* in, int length, double* out) {
  __shared__ double values[1024];
  int i = 1;
  int idx = blockIdx.x*1024 + threadIdx.x;
  values[threadIdx.x] = idx < length ? in[idx] : 0;
  __syncthreads();
  while (i < 1024) {
    if (threadIdx.x + i < 1024) {
      values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + i]);
    }
    i *= 2;
    __syncthreads();
  }
  out[blockIdx.x] = values[0];
}

__global__
void divide_all(double * array, int length, double* divider) {  // remove *
  int idx = blockIdx.x*1024 + threadIdx.x;
  if (idx < length) {
    array[idx] *= 255./(*divider);
  }
}
}
