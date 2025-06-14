/**
 * @file arithmetic_operation.cpp
 * @brief Demonstrates various matrix operations using custom matrix types in
 * C++.
 *
 * This file contains a sample program that showcases the creation and
 * manipulation of different types of matrices, including dense, diagonal, and
 * sparse matrices. It demonstrates matrix initialization, arithmetic
 * operations, transposition, conversion between matrix types, and the creation
 * of matrices filled with zeros, ones, or a specific value. The program also
 * illustrates how to print the contents of these matrices.
 */
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

  /* Zeros */
  auto Dense_zeros = make_DenseMatrixZeros<double, 3, 3>();

  std::cout << "Dense_zeros = " << std::endl;
  for (size_t i = 0; i < Dense_zeros.cols(); ++i) {
    for (size_t j = 0; j < Dense_zeros.rows(); ++j) {
      std::cout << Dense_zeros(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto Diag_zeros = make_DiagMatrixZeros<double, 3>();

  auto Diag_zeros_dense = Diag_zeros.create_dense();
  std::cout << "Diag_zeros = " << std::endl;
  for (size_t i = 0; i < Diag_zeros.cols(); ++i) {
    for (size_t j = 0; j < Diag_zeros.rows(); ++j) {
      std::cout << Diag_zeros_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto Sparse_zeros = make_SparseMatrixZeros<double, SparseAvailable_C>();

  auto Sparse_zeros_dense = Sparse_zeros.create_dense();
  std::cout << "Sparse_zeros = " << std::endl;
  for (size_t i = 0; i < Sparse_zeros_dense.cols(); ++i) {
    for (size_t j = 0; j < Sparse_zeros_dense.rows(); ++j) {
      std::cout << Sparse_zeros_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Ones */
  auto Dense_ones = make_DenseMatrixOnes<double, 3, 3>();

  std::cout << "Dense_ones = " << std::endl;
  for (size_t i = 0; i < Dense_ones.cols(); ++i) {
    for (size_t j = 0; j < Dense_ones.rows(); ++j) {
      std::cout << Dense_ones(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto Diag_ones = make_DiagMatrixIdentity<double, 3>();

  auto Diag_ones_dense = Diag_ones.create_dense();
  std::cout << "Diag_ones = " << std::endl;
  for (size_t i = 0; i < Diag_ones.cols(); ++i) {
    for (size_t j = 0; j < Diag_ones.rows(); ++j) {
      std::cout << Diag_ones_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto Sparse_ones = make_SparseMatrixOnes<double, SparseAvailable_C>();

  auto Sparse_ones_dense = Sparse_ones.create_dense();
  std::cout << "Sparse_ones = " << std::endl;
  for (size_t i = 0; i < Sparse_ones_dense.cols(); ++i) {
    for (size_t j = 0; j < Sparse_ones_dense.rows(); ++j) {
      std::cout << Sparse_ones_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Full */
  auto Dense_full = make_DenseMatrixFull<3, 3>(2.0);

  std::cout << "Dense_full = " << std::endl;
  for (size_t i = 0; i < Dense_full.cols(); ++i) {
    for (size_t j = 0; j < Dense_full.rows(); ++j) {
      std::cout << Dense_full(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto Diag_full = make_DiagMatrixFull<3>(2.0);
  auto Diag_full_dense = Diag_full.create_dense();

  std::cout << "Diag_full = " << std::endl;
  for (size_t i = 0; i < Diag_full.cols(); ++i) {
    for (size_t j = 0; j < Diag_full.rows(); ++j) {
      std::cout << Diag_full_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto Sparse_full = make_SparseMatrixFull<SparseAvailable_C>(2.0);
  auto Sparse_full_dense = Sparse_full.create_dense();

  std::cout << "Sparse_full = " << std::endl;
  for (size_t i = 0; i < Sparse_full_dense.cols(); ++i) {
    for (size_t j = 0; j < Sparse_full_dense.rows(); ++j) {
      std::cout << Sparse_full_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
