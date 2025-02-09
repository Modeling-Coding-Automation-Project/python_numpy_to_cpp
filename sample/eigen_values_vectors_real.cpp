#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto A_r =
      make_DenseMatrix<3, 3>(6.0, -3.0, 5.0, -1.0, 4.0, -5.0, -3.0, 3.0, -4.0);

  auto solver_r = make_LinalgSolverEigReal<decltype(A_r)>();
  // Or, you can use "LinalgSolverEigReal_Type" to declare the type of
  // LinalgSolverEigReal.
  LinalgSolverEigReal_Type<decltype(A_r)> solver_r_t = solver_r;

  solver_r_t.solve_eigen_values(A_r);
  auto eigen_values_r = solver_r_t.get_eigen_values();

  solver_r_t.solve_eigen_vectors(A_r);
  auto eigen_vectors_r = solver_r_t.get_eigen_vectors();

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
