#ifndef PYTHON_NUMPY_COMPLEX_HPP
#define PYTHON_NUMPY_COMPLEX_HPP

#include "base_matrix.hpp"

namespace PythonNumpy {

template <typename T> using Complex = Base::Matrix::Complex<T>;

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_COMPLEX_HPP
