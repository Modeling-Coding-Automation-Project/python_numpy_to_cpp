#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto A = make_DenseMatrix<4, 4>(16, 2, 3, 13, 5, 11, 10, 8, 9, 7, 6, 12, 4,
                                  14, 15, 1);

  auto B = reshape<8, 2>(A);

  std::cout << "B = " << std::endl;
  for (size_t i = 0; i < B.cols(); ++i) {
    for (size_t j = 0; j < B.rows(); ++j) {
      std::cout << B(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Update */
  A = make_DenseMatrix<4, 4>(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  update_reshaped_matrix(B, A);

  std::cout << "B = " << std::endl;
  for (size_t i = 0; i < B.cols(); ++i) {
    for (size_t j = 0; j < B.rows(); ++j) {
      std::cout << B(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
