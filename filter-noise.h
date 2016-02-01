/**
 * Smoothen an image. The image is provided by RGB {@code channels}. It has
 * dimy {@code rows} and {@code dimx} columns. The value of R/G/B channel in
 * i'th row and j'th column must be found at
 * {@code channels[0/1/2][i*dimx + j]}.
 */
void enhance_image(unsigned char* channels[3], int dimy, int dimx);
