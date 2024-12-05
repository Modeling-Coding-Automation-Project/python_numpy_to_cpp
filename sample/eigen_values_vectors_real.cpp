#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  Matrix<DefDense, double, 3, 3> A_r({{6, -3, 5}, {-1, 4, -5}, {-3, 3, -4}});

  static auto solver_r = make_LinalgSolverEigReal(A_r);
  solver_r.solve_eigen_values(A_r);
  auto eigen_values_r = solver_r.get_eigen_values();

  solver_r.solve_eigen_vectors(A_r);
  auto eigen_vectors_r = solver_r.get_eigen_vectors();

  std::cout << "eigen_values_r = " << std::endl;
  for (size_t j = 0; j < eigen_values_r.cols(); ++j) {
    for (size_t i = 0; i < eigen_values_r.rows(); ++i) {
      std::cout << eigen_values_r(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "eigen_vectors_r = " << std::endl;
  for (size_t j = 0; j < eigen_vectors_r.cols(); ++j) {
    for (size_t i = 0; i < eigen_vectors_r.rows(); ++i) {
      std::cout << eigen_vectors_r(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto result = A_r * eigen_vectors_r -
                eigen_vectors_r * Matrix<DefDiag, double, 3>(eigen_values_r);

  std::cout << "result = " << std::endl;
  for (size_t j = 0; j < result.cols(); ++j) {
    for (size_t i = 0; i < result.rows(); ++i) {
      std::cout << result(j, i) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
