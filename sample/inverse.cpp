#include <iostream>

#include "python_numpy.hpp"

using namespace PythonNumpy;

int main() {

  // Matrix<DefDense, double, 3, 3> A({{1, 2, 3}, {5, 4, 6}, {9, 8, 7}});
  auto A = make_DenseMatrix<3, 3>(1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 9.0, 8.0, 7.0);

  // Matrix<DefDense, double, 3, 3> A_s({{1, 2, 3}, {5, 4, 6}, {5, 4, 6}});
  auto A_s =
      make_DenseMatrix<3, 3>(1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 5.0, 4.0, 6.0);

  // Matrix<DefDiag, double, 3> B({1, 2, 3});
  auto B = make_DiagMatrix<3>(1.0, 2.0, 3.0);

  auto C =
      make_SparseMatrix<SparseAvailable<ColumnAvailable<true, false, false>,
                                        ColumnAvailable<true, false, true>,
                                        ColumnAvailable<false, true, true>>>(
          1.0, 3.0, 8.0, 2.0, 4.0);

  auto solver = make_LinalgSolverInv<decltype(A)>();
  // Or, you can use "LinalgSolverInv_Type" to declare the type of
  // LinalgSolverInv.
  LinalgSolverInv_Type<decltype(A)> solver_t = solver;

  auto result = solver_t.inv(A);

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
