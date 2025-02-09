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

  auto E = make_SparseMatrixEmpty<double, 3, 3>();

  auto vertical = concatenate_vertically(A, C);

  auto vertical_dense = vertical.create_dense();
  std::cout << "vertical = " << std::endl;
  for (size_t j = 0; j < vertical_dense.cols(); ++j) {
    for (size_t i = 0; i < vertical_dense.rows(); ++i) {
      std::cout << vertical_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto horizontal = concatenate_horizontally(A, B);

  auto horizontal_dense = horizontal.create_dense();
  std::cout << "horizontal = " << std::endl;
  for (size_t j = 0; j < horizontal_dense.cols(); ++j) {
    for (size_t i = 0; i < horizontal_dense.rows(); ++i) {
      std::cout << horizontal_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Update */
  update_horizontally_concatenated_matrix(horizontal, A, 2.0 * B);

  horizontal_dense = horizontal.create_dense();
  std::cout << "horizontal = " << std::endl;
  for (size_t j = 0; j < horizontal_dense.cols(); ++j) {
    for (size_t i = 0; i < horizontal_dense.rows(); ++i) {
      std::cout << horizontal_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Create Concatenated Type */
  using A_v_C_Type = ConcatenateVertically_Type<decltype(A), decltype(C)>;

  A_v_C_Type A_v_C = vertical;
  auto A_v_C_dense = A_v_C.create_dense();

  std::cout << "vertical = " << std::endl;
  for (size_t j = 0; j < A_v_C_dense.cols(); ++j) {
    for (size_t i = 0; i < A_v_C_dense.rows(); ++i) {
      std::cout << A_v_C_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  using A_h_B_Type = ConcatenateHorizontally_Type<decltype(A), decltype(B)>;

  A_h_B_Type A_h_B = horizontal;
  auto A_h_B_dense = A_h_B.create_dense();

  std::cout << "horizontal = " << std::endl;
  for (size_t j = 0; j < A_h_B_dense.cols(); ++j) {
    for (size_t i = 0; i < A_h_B_dense.rows(); ++i) {
      std::cout << A_h_B_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Concatenate Block */
  using ABCE_Type =
      ConcatenateBlock_Type<decltype(A), decltype(B), decltype(C), decltype(E)>;

  ABCE_Type ABCE = concatenate_block(A, B, C, E);
  auto ABCE_dense = ABCE.create_dense();

  std::cout << "ABCE = " << std::endl;
  for (size_t j = 0; j < ABCE_dense.cols(); ++j) {
    for (size_t i = 0; i < ABCE_dense.rows(); ++i) {
      std::cout << ABCE_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Update Block */
  update_block_concatenated_matrix(ABCE, A, 2.0 * B, C, E);
  ABCE_dense = ABCE.create_dense();

  std::cout << "ABCE = " << std::endl;
  for (size_t j = 0; j < ABCE_dense.cols(); ++j) {
    for (size_t i = 0; i < ABCE_dense.rows(); ++i) {
      std::cout << ABCE_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
