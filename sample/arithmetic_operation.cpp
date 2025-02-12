#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto A = make_DenseMatrix<3, 3>(1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 9.0, 8.0, 7.0);
  // If you want to use specific type of Dense Matrix, you can use
  // "DenseMatrix_Type".
  DenseMatrix_Type<double, 3, 3> A_t = A;

  auto B = make_DiagMatrix<3>(1.0, 2.0, 3.0);
  // If you want to use specific type of Diag Matrix, you can use
  // "DiagMatrix_Type".
  DiagMatrix_Type<double, 3> B_t = B;

  using SparseAvailable_C = SparseAvailable<ColumnAvailable<true, false, false>,
                                            ColumnAvailable<true, false, true>,
                                            ColumnAvailable<false, true, true>>;

  auto C = make_SparseMatrix<SparseAvailable_C>(1.0, 3.0, 8.0, 2.0, 4.0);
  // If you want to use specific type of Sparse Matrix, you can use
  // "SparseMatrix_Type".
  SparseMatrix_Type<double, SparseAvailable_C> C_t = C;

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

  /* Transpose */
  auto A_T = A.transpose();

  std::cout << "A_T = " << std::endl;
  for (size_t i = 0; i < A_T.cols(); ++i) {
    for (size_t j = 0; j < A_T.rows(); ++j) {
      std::cout << A_T(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Transpose Multiply */
  auto A_T_mul_C = ATranspose_mul_B(A, C);

  std::cout << "A_T_mul_C = " << std::endl;
  for (size_t i = 0; i < A_T_mul_C.cols(); ++i) {
    for (size_t j = 0; j < A_T_mul_C.rows(); ++j) {
      std::cout << A_T_mul_C(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto A_mul_C_T = A_mul_BTranspose(A, C);

  std::cout << "A_mul_C_T = " << std::endl;
  for (size_t i = 0; i < A_mul_C_T.cols(); ++i) {
    for (size_t j = 0; j < A_mul_C_T.rows(); ++j) {
      std::cout << A_mul_C_T(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
