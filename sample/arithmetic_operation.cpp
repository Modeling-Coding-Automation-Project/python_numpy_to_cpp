#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto A = make_DenseMatrix<3, 3>(1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 9.0, 8.0, 7.0);

  auto B = make_DiagMatrix<3>(1.0, 2.0, 3.0);

  auto C =
      make_SparseMatrix<SparseAvailable<ColumnAvailable<true, false, false>,
                                        ColumnAvailable<true, false, true>,
                                        ColumnAvailable<false, true, true>>>(
          1.0, 3.0, 8.0, 2.0, 4.0);

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
  auto E = make_SparseMatrixEmpty<double, 3, 4>();

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

  // if you want to expand columns and rows of sparse matrix, you should use
  // Empty Sparse Matrix and "concatenate_horizontally" and
  // "concatenate_vertically".
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
