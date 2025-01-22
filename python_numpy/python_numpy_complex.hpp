#ifndef __PYTHON_NUMPY_COMPLEX_HPP__
#define __PYTHON_NUMPY_COMPLEX_HPP__

#include "base_matrix.hpp"

namespace PythonNumpy {

template <typename T> using Complex = Base::Matrix::Complex<T>;

template <typename T> struct UnderlyingType {
  using Type = T;
};

template <typename U> struct UnderlyingType<Complex<U>> {
  using Type = U;
};

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_COMPLEX_HPP__
