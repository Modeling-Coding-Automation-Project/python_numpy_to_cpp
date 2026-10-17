/**
 * @file convolve.cpp
 * @brief Demonstrates one-dimensional convolution in full, same, and valid
 * modes.
 *
 * The inputs and expected results match sample/convolve.py, allowing the
 * PythonNumpy functions to be compared directly with numpy.convolve.
 */
#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

template <typename MatrixType> void print_vector(const MatrixType &matrix) {
  const std::size_t length = matrix.ROWS == 1 ? matrix.COLS : matrix.ROWS;
  for (std::size_t i = 0; i < length; ++i) {
    std::cout << (matrix.ROWS == 1 ? matrix(0, i) : matrix(i, 0));
    if (i + 1 < length) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;
}

int main() {
  auto a = make_DenseMatrix<1, 5>(1.0, 2.0, 3.0, 4.0, 5.0);
  auto v = make_DenseMatrix<1, 3>(1.0, 1.0, 1.0);

  std::cout << "a = ";
  print_vector(a);
  std::cout << "v = ";
  print_vector(v);

  std::cout << "full = ";
  print_vector(convolve_full(a, v));
  std::cout << "same = ";
  print_vector(convolve_same(a, v));
  std::cout << "valid = ";
  print_vector(convolve_valid(a, v));

  return 0;
}
