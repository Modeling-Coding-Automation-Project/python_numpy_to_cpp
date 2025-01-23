#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto A = make_DenseMatrix<3, 3>(2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 3.0, 2.0, 6.0);

  auto B = make_DiagMatrix<3>(1.0, 2.0, 3.0);

  auto C =
      make_SparseMatrix<SparseAvailable<ColumnAvailable<true, false, false>,
                                        ColumnAvailable<true, false, true>,
                                        ColumnAvailable<false, true, true>>>(
          1.0, 3.0, 8.0, 2.0, 4.0);

  static auto solver = make_LinalgSolverCholesky(A);
  auto SA = solver.solve(A);

  auto SA_dense = SA.create_dense();
  std::cout << "SA = " << std::endl;
  for (size_t j = 0; j < SA_dense.cols(); ++j) {
    for (size_t i = 0; i < SA_dense.rows(); ++i) {
      std::cout << SA_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto result = ATranspose_mul_B(SA, SA);
  auto result_dense = result.create_dense();

  std::cout << "result = " << std::endl;
  for (size_t j = 0; j < result.cols(); ++j) {
    for (size_t i = 0; i < result.rows(); ++i) {
      std::cout << result_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
