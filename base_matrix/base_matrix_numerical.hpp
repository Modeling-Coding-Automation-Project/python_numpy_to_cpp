/**
 * @file base_matrix_numerical.hpp
 * @brief Scaled scalar operations for fixed-storage iterative solvers.
 *
 * Inputs must be finite. The caller checks non-finite inputs and outputs.
 * Scaling protects intermediate squares; the final result must still fit T.
 */
#ifndef BASE_MATRIX_NUMERICAL_HPP_
#define BASE_MATRIX_NUMERICAL_HPP_

#include "base_matrix_complex.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {
namespace Numerical {

template <typename T> inline T positive_threshold(T value) {
  return value > static_cast<T>(0) ? value : static_cast<T>(1.0e-10);
}

// Only normalized, nonnegative sums are passed here (normally [1, 2*N]).
template <typename T> inline T normalized_sqrt(T value) {
  return value == static_cast<T>(0) ? static_cast<T>(0)
                                    : Base::Math::sqrt_newton_method<T, 3>(
                                          value, static_cast<T>(1.0e-20));
}

template <typename T> struct Scalar {
  using Real = T;
  static inline T conjugate(T value) { return value; }
  static inline T component_max(T value) { return Base::Math::abs(value); }
  static inline T squared_abs(T value) { return value * value; }
  static inline T magnitude(T value) { return Base::Math::abs(value); }
  static inline T divide(T numerator, T denominator, T floor) {
    return numerator / Base::Utility::avoid_zero_divide(denominator, floor);
  }
};

template <typename T> struct Scalar<Complex<T>> {
  using Real = T;
  static inline Complex<T> conjugate(const Complex<T> &value) {
    return Complex<T>(value.real, -value.imag);
  }
  static inline T component_max(const Complex<T> &value) {
    T real = Base::Math::abs(value.real);
    T imag = Base::Math::abs(value.imag);
    return real > imag ? real : imag;
  }
  static inline T squared_abs(const Complex<T> &value) {
    return value.real * value.real + value.imag * value.imag;
  }
  static inline T magnitude(const Complex<T> &value) {
    T scale = component_max(value);
    if (scale == static_cast<T>(0)) {
      return static_cast<T>(0);
    }
    // The scale is known nonzero. Do not clamp a valid tiny scale.
    Complex<T> normalized = value / scale;
    return scale * normalized_sqrt(squared_abs(normalized));
  }
  static inline Complex<T> divide(const Complex<T> &numerator,
                                  const Complex<T> &denominator, T floor) {
    T scale = component_max(denominator);
    if (scale == static_cast<T>(0)) {
      return numerator / floor;
    }
    Complex<T> normalized = denominator / scale;
    T square = squared_abs(normalized);
    // Clamp magnitude while preserving phase. The normalized square is [1, 2].
    if (scale < floor) {
      T length = normalized_sqrt(square);
      if (scale * length < floor) {
        scale = floor / length;
      }
    }
    // Divide before multiplying when scale is large. For a small scale,
    // postponing that division avoids overflowing an intermediate quotient.
    if (scale >= static_cast<T>(1)) {
      return ((numerator / scale) / square) * conjugate(normalized);
    }
    return ((numerator / square) * conjugate(normalized)) / scale;
  }
};

template <typename T, std::size_t N>
inline typename Scalar<T>::Real norm(const Vector<T, N> &value,
                                     std::size_t active_size = N) {
  using Real = typename Scalar<T>::Real;
  Real scale = static_cast<Real>(0);
  for (std::size_t i = 0; i < N; ++i) {
    if (i < active_size) {
      Real candidate = Scalar<T>::component_max(value[i]);
      if (candidate > scale) {
        scale = candidate;
      }
    }
  }
  if (scale == static_cast<Real>(0)) {
    return static_cast<Real>(0);
  }
  Real sum = static_cast<Real>(0);
  for (std::size_t i = 0; i < N; ++i) {
    if (i < active_size) {
      sum += Scalar<T>::squared_abs(value[i] / scale);
    }
  }
  return scale * normalized_sqrt(sum);
}

/** Real c and scalar s define [c s; -conj(s) c]. */
template <typename T>
inline void givens(const T &first, const T &second,
                   typename Scalar<T>::Real &cosine, T &sine) {
  using Real = typename Scalar<T>::Real;
  Real scale = Scalar<T>::component_max(first);
  Real second_scale = Scalar<T>::component_max(second);
  if (second_scale == static_cast<Real>(0)) {
    cosine = static_cast<Real>(1);
    sine = T();
    return;
  }
  if (second_scale > scale) {
    scale = second_scale;
  }
  T first_scaled = first / scale;
  T second_scaled = second / scale;
  Real first_norm = Scalar<T>::magnitude(first_scaled);
  Real length = normalized_sqrt(Scalar<T>::squared_abs(first_scaled) +
                                Scalar<T>::squared_abs(second_scaled));
  if (first_norm == static_cast<Real>(0)) {
    cosine = static_cast<Real>(0);
    sine = Scalar<T>::conjugate(second_scaled) / length;
  } else {
    T phase = first_scaled / first_norm;
    cosine = first_norm / length;
    sine = phase * (Scalar<T>::conjugate(second_scaled) / length);
  }
}

} // namespace Numerical
} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_NUMERICAL_HPP_
