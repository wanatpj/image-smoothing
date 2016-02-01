#include "CImg.h"
#include "filter-noise.h"

using namespace cimg_library;

unsigned char * channels[3];

static void alloc(int n) {
  for (int i = 0; i < 3; ++i) {
    channels[i] = new unsigned char[n];
  }
}

static void dealloc() {
  for (int i = 0; i < 3; ++i) {
    delete [] channels[i];
  }
}

int main() {
  CImg<unsigned char> image("test.jpg");
  alloc(image.width() * image.height());
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < image.height(); ++y) {
      for (int x = 0; x < image.width(); ++x) {
        channels[c][y * image.width() + x] = image(x, y, 0, c);
      }
    }
  }
  enhance_image(channels, image.height(), image.width());
  for (int x = 0; x < image.width(); ++x) {
    for (int y = 0; y < image.height(); ++y) {
      unsigned char arr[] = {
          channels[0][y * image.width() + x],
          channels[1][y * image.width() + x],
          channels[2][y * image.width() + x]
      };
      image.draw_point(x, y, arr);
    }
  }
  image.save("testout.jpg");
  dealloc();
  return 0;
}
