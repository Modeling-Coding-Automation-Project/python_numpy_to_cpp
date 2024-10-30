#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  Matrix<DefDense, double, 3, 3> A({{2, 1, 3}, {1, 4, 2}, {3, 2, 6}});

  Matrix<DefDiag, double, 3> B({1, 2, 3});

  Matrix<DefSparse, double, 3, 3, 5> C({1, 3, 8, 2, 4}, {0, 0, 2, 1, 2},
                                       {0, 1, 3, 5});

  static auto solver = make_LinalgSolverCholesky(A);
  auto SA = solver.solve(A);

  auto SA_dense = SA.create_dense();
  std::cout << "SA = " << std::endl;
  for (size_t j = 0; j < SA_dense.cols(); ++j) {
    for (size_t i = 0; i < SA_dense.rows(); ++i) {
      std::cout << SA_dense.matrix(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto result = AT_mul_B(SA, SA);
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
