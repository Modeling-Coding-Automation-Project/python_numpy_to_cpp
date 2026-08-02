#ifndef PYTHON_MATH_HPP_
#define PYTHON_MATH_HPP_

#include "base_matrix.hpp"

#include "python_numpy_augmented_matrix.hpp"
#include "python_numpy_base.hpp"

namespace PythonNumpy {

/* Constants */

constexpr double PI = Base::Matrix::PI;
constexpr double TWO_PI = Base::Matrix::TWO_PI;
constexpr double HALF_PI = Base::Matrix::HALF_PI;
constexpr double TAU = Base::Matrix::TAU;

constexpr double E = Base::Matrix::E;

constexpr double LN_2 = Base::Matrix::LN_2;
constexpr double LN_10 = Base::Matrix::LN_10;

} // namespace PythonNumpy

#endif // PYTHON_MATH_HPP_
