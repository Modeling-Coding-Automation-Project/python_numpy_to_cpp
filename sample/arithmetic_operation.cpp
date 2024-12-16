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
  for (size_t i = 0; i < result.cols(); ++i) {
    for (size_t j = 0; j < result.rows(); ++j) {
      std::cout << result(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Create Empty Sparse Matrix */
  Matrix<DefSparse, double, 3, 4, SparseAvailableEmpty<3, 4>> E;
  auto E_dense = E.create_dense(); // convert SparseMatrix to DenseMatrix

  std::cout << "E = " << std::endl;
  for (size_t i = 0; i < E_dense.cols(); ++i) {
    for (size_t j = 0; j < E_dense.rows(); ++j) {
      std::cout << E_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Concatenate Two Matrices */
  auto concat_A_B = concatenate_vertically(A, B);
  auto concat_A_B_dense =
      concat_A_B.create_dense(); // convert SparseMatrix to DenseMatrix

  std::cout << "concat_A_B = " << std::endl;
  for (size_t i = 0; i < concat_A_B_dense.cols(); ++i) {
    for (size_t j = 0; j < concat_A_B_dense.rows(); ++j) {
      std::cout << concat_A_B_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto concat_E_C = concatenate_horizontally(E, C);
  auto concat_E_C_dense =
      concat_E_C.create_dense(); // convert SparseMatrix to DenseMatrix

  std::cout << "concat_E_C = " << std::endl;
  for (size_t i = 0; i < concat_E_C_dense.cols(); ++i) {
    for (size_t j = 0; j < concat_E_C_dense.rows(); ++j) {
      std::cout << concat_E_C_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
