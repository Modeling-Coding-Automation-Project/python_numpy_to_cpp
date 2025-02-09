#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto A_c =
      make_DenseMatrix<3, 3>(1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0);

  auto solver_c = make_LinalgSolverEig<decltype(A_c)>();
  // Or, you can use "LinalgSolverEig_Type" to declare the type of
  // LinalgSolverEig.
  LinalgSolverEig_Type<decltype(A_c)> solver_c_t = solver_c;

  solver_c_t.solve_eigen_values(A_c);
  auto eigen_values_c = solver_c_t.get_eigen_values();

  solver_c_t.solve_eigen_vectors(A_c);
  auto eigen_vectors_c = solver_c_t.get_eigen_vectors();

  std::cout << "eigen_values_c = " << std::endl;
  for (size_t j = 0; j < eigen_values_c.cols(); ++j) {
    for (size_t i = 0; i < eigen_values_c.rows(); ++i) {
      std::cout << "[" << eigen_values_c(j, i).real << ", "
                << eigen_values_c(j, i).imag << "j], ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "eigen_vectors_c = " << std::endl;
  for (size_t j = 0; j < eigen_vectors_c.cols(); ++j) {
    for (size_t i = 0; i < eigen_vectors_c.rows(); ++i) {
      std::cout << "[" << eigen_vectors_c(j, i).real << ", "
                << eigen_vectors_c(j, i).imag << "j], ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto A_c_comp = A_c.create_complex();
  auto eigen_values_matrix =
      Matrix<DefDiag, Complex<double>, 3>(eigen_values_c);
  auto result =
      A_c_comp * eigen_vectors_c - eigen_vectors_c * eigen_values_matrix;

  std::cout << "result = " << std::endl;
  for (size_t j = 0; j < result.cols(); ++j) {
    for (size_t i = 0; i < result.rows(); ++i) {
      std::cout << "[" << result(j, i).real << ", " << result(j, i).imag
                << "j], ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
