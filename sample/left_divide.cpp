/**
 * @file left_divide.cpp
 * @brief Demonstrates solving a linear system using custom matrix and solver
 * classes.
 *
 * This example creates several types of matrices (dense, diagonal, sparse)
 * using the PythonNumpy library, and solves a linear system of equations using
 * a linear algebra solver. The result is printed to the console.
 */
#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  auto Zero = make_DenseMatrixZeros<double, 3, 3>();

  auto A = make_DenseMatrix<3, 3>(1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 9.0, 8.0, 7.0);

  auto B = make_DiagMatrix<3>(1.0, 2.0, 3.0);

  auto C =
      make_SparseMatrix<SparseAvailable<ColumnAvailable<true, false, false>,
                                        ColumnAvailable<true, false, true>,
                                        ColumnAvailable<false, true, true>>>(
          1.0, 3.0, 8.0, 2.0, 4.0);

  auto solver = make_LinalgSolver<decltype(A), decltype(C)>();
  // Or, you can use "LinalgSolver_Type" to declare the type of
  // LinalgSolver.
  LinalgSolver_Type<decltype(A), decltype(C)> solver_t = solver;

  auto result = solver_t.solve(A, C);

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
