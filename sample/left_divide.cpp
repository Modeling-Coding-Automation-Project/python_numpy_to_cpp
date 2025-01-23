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

  static auto solver = make_LinalgSolver(A, C);
  auto result = solver.solve(A, C);

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
