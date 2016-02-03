# image-smoothing
CUDA implementation of certain image smoothing algorithm.

This project is some time-free image smoothing method. The algorithm is quite
simple. Basically, we detect edges and we sharpen region around them, and the
rest of the image is blured. Usually, applying sharpening filter
{{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}} may causing bad efects for already
sharpened images, so this solution will fail in trying giving good results.
However, this will work pretty well for old noisy cartoons or photos of
lectures.

Due to IP rights, I cannot include any examples with shots of cartoons.

Example image:
![before smoothing](https://raw.githubusercontent.com/wanatpj/image-smoothing/master/test.jpg)
After smooting:
![after smoothing](https://raw.githubusercontent.com/wanatpj/image-smoothing/master/testout.jpg)


If you are eager to contribute to this project, you can try to:
* verify, if there is a way to find a value d, different for each pixel that
  applying kernel {{0, 1 - d, 0}, {1 - d, 1 + 4d, 1 - d}, {0, 1 - d, 0}} will
  not destroy the image when we iterate applying over and over.
* change reading the input image from file location passed by a flag. The same
  for the output image.

Feel free to contact me at gmail. Same id as on github.
