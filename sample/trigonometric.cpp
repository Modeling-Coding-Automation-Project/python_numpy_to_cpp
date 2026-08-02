#include <iostream>

#include "python_numpy.hpp"

int main() {

  auto time = PythonNumpy::make_DenseMatrix<11, 1>(0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                   0.6, 0.7, 0.8, 0.9, 1.0);

  auto series_data = PythonNumpy::sin(2.0 * PythonNumpy::PI * time);

  std::cout << "Time:" << std::endl;
  for (std::size_t i = 0; i < time.ROWS; ++i) {
    std::cout << time(i) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Series Data:" << std::endl;
  for (std::size_t i = 0; i < series_data.ROWS; ++i) {
    std::cout << series_data(i) << std::endl;
  }

  return 0;
}
