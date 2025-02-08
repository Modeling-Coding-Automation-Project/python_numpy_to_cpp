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

  auto solver = make_LinalgSolverQR<decltype(A)>();
  // Or, you can use "LinalgSolverQR_Type" to declare the type of
  // LinalgSolverQR.
  LinalgSolverQR_Type<decltype(A)> solver_t = solver;

  solver_t.solve(A);
  auto Q = solver_t.get_Q();
  auto R = solver_t.get_R();

  std::cout << "Q = " << std::endl;
  for (size_t j = 0; j < Q.cols(); ++j) {
    for (size_t i = 0; i < Q.rows(); ++i) {
      std::cout << Q(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto R_dense = R.create_dense();
  std::cout << "R = " << std::endl;
  for (size_t j = 0; j < R_dense.cols(); ++j) {
    for (size_t i = 0; i < R_dense.rows(); ++i) {
      std::cout << R_dense(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto result = Q * R;
  std::cout << "result = " << std::endl;
  for (size_t j = 0; j < result.cols(); ++j) {
    for (size_t i = 0; i < result.rows(); ++i) {
      std::cout << result(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
