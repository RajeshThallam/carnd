- color, shape, orientation and position in the image are useful features in the identification of
  lane lines on the road

- numpy.polyfit helps extrapolating or fittinga polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y). It returns a vector of coefficients p that minimises the squared error.
  [link](https://peteris.rocks/blog/extrapolate-lines-with-numpy-polyfit/)

- we expect to find edges where the pixel values are changing rapidly.

- canny edge detection
  - gradient in an image
  - gaussian blurring