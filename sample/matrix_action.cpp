#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  /* Reshape */
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

  /* Update reshaped matrix */
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

  /* Substitute */
  auto A_s = make_DenseMatrix<3, 3>(10.0, 20.0, 30.0, 50.0, 40.0, 60.0, 90.0,
                                    80.0, 70.0);

  using SparseAvailable_C = SparseAvailable<ColumnAvailable<true, false, false>,
                                            ColumnAvailable<true, false, true>,
                                            ColumnAvailable<false, true, true>>;

  auto C = make_SparseMatrix<SparseAvailable_C>(1.0, 3.0, 8.0, 2.0, 4.0);

  substitute_matrix(C, A_s);
  auto C_dense = C.create_dense();

  std::cout << "C = " << std::endl;
  for (size_t i = 0; i < C_dense.cols(); ++i) {
    for (size_t j = 0; j < C_dense.rows(); ++j) {
      std::cout << C_dense(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Substitute small matrix to large matrix */
  substitute_part_matrix<1, 1>(A, C);

  std::cout << "A = " << std::endl;
  for (size_t i = 0; i < A.cols(); ++i) {
    for (size_t j = 0; j < A.rows(); ++j) {
      std::cout << A(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
