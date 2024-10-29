#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  Matrix<DefDense, double, 3, 3> A({{1, 2, 3}, {5, 4, 6}, {9, 8, 7}});

  Matrix<DefDiag, double, 3> B({1, 2, 3});

  Matrix<DefSparse, double, 3, 3, 5> C({1, 3, 8, 2, 4}, {0, 0, 2, 1, 2},
                                       {0, 1, 3, 5});

  auto vertical = concatenate_vertically(A, C);

  auto vertical_dense = vertical.create_dense();
  std::cout << "vertical = " << std::endl;
  for (size_t j = 0; j < vertical_dense.cols(); ++j) {
    for (size_t i = 0; i < vertical_dense.rows(); ++i) {
      std::cout << vertical_dense.matrix(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto horizontal = concatenate_horizontally(A, B);

  auto horizontal_dense = horizontal.create_dense();
  std::cout << "horizontal = " << std::endl;
  for (size_t j = 0; j < horizontal_dense.cols(); ++j) {
    for (size_t i = 0; i < horizontal_dense.rows(); ++i) {
      std::cout << horizontal_dense.matrix(j, i) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
