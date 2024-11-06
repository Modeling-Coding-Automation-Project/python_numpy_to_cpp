#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto Zero = Matrix<DefDense, double, 3, 3>::zeros();

  Matrix<DefDense, double, 3, 3> A({{1, 2, 3}, {5, 4, 6}, {9, 8, 7}});

  Matrix<DefDiag, double, 3> B({1, 2, 3});

  Matrix<DefSparse, double, 3, 3, 5> C({1, 3, 8, 2, 4}, {0, 0, 2, 1, 2},
                                       {0, 1, 3, 5});

  static auto solver = make_LinalgSolver(A, C);
  auto result = solver.solve(A, C);

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