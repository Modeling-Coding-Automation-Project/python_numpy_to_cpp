#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  Matrix<DefDense, double, 3, 3> A({{1, 2, 3}, {5, 4, 6}, {9, 8, 7}});

  Matrix<DefDiag, double, 3> B({1, 2, 3});

  Matrix<DefSparse, double, 3, 3,
         SparseAvailable<ColumnAvailable<true, false, false>,
                         ColumnAvailable<true, false, true>,
                         ColumnAvailable<false, true, true>>>
      C({1, 3, 8, 2, 4});

  auto result = A * C;

  std::cout << "result = " << std::endl;
  for (size_t j = 0; j < result.matrix.cols(); ++j) {
    for (size_t i = 0; i < result.matrix.rows(); ++i) {
      std::cout << result.matrix(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
