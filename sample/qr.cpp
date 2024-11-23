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

  static auto solver = make_LinalgSolverQR(A);
  solver.solve(A);
  auto Q = solver.get_Q();
  auto R = solver.get_R();

  std::cout << "Q = " << std::endl;
  for (size_t j = 0; j < Q.cols(); ++j) {
    for (size_t i = 0; i < Q.rows(); ++i) {
      std::cout << Q.matrix(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto R_dense = R.create_dense();
  std::cout << "R = " << std::endl;
  for (size_t j = 0; j < R_dense.cols(); ++j) {
    for (size_t i = 0; i < R_dense.rows(); ++i) {
      std::cout << R_dense.matrix(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto result = Q * R;
  std::cout << "result = " << std::endl;
  for (size_t j = 0; j < result.cols(); ++j) {
    for (size_t i = 0; i < result.rows(); ++i) {
      std::cout << result.matrix(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
