/**
 * @file base_matrix_numerical.hpp
 * @brief Provides scaled real and complex operations for fixed-storage solvers.
 *
 * Scalar traits, vector norms, and Givens rotations share scaling operations
 * that avoid squaring unscaled inputs. Square roots are delegated to
 * Base::Math. Vector scans support both fixed for-loops and recursive template
 * expansion.
 *
 * Inputs must be finite. The caller checks non-finite inputs and outputs.
 * Scaling protects intermediate squares; the final result must still fit T.
 */
#ifndef BASE_MATRIX_NUMERICAL_HPP_
#define BASE_MATRIX_NUMERICAL_HPP_

#include "base_matrix_complex.hpp"
#include "base_matrix_reciprocal.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {
namespace Numerical {

// Fallback for a nonpositive denominator threshold supplied by the caller.
constexpr double DEFAULT_DIVISION_MIN_NUMERICAL = 1.0e-10;
// Denominator protection passed to the external sqrt for normalized sums.
constexpr double NORMALIZED_SQRT_DIVISION_MIN = 1.0e-20;

/**
 * @brief Replaces a nonpositive denominator threshold with the default value.
 *
 * @tparam T The real floating-point type of the threshold.
 * @param value The requested denominator threshold.
 * @return value when positive, or DEFAULT_DIVISION_MIN_NUMERICAL otherwise.
 */
template <typename T> inline T positive_threshold(T value) {
  return value > static_cast<T>(0)
             ? value
             : static_cast<T>(DEFAULT_DIVISION_MIN_NUMERICAL);
}

/**
 * @brief Computes the square root of a normalized, nonnegative sum.
 *
 * Zero is returned directly. Positive inputs, normally in [1, 2*N] for an
 * N-element norm, use the configured Base::Math square-root implementation.
 *
 * @tparam T The real floating-point type of the sum.
 * @param value The nonnegative sum after component scaling.
 * @return The square root, with accuracy determined by Base::Math.
 */
template <typename T> inline T normalized_sqrt(T value) {
  return value == static_cast<T>(0)
             ? static_cast<T>(0)
             : Base::Math::sqrt<T>(
                   value, static_cast<T>(NORMALIZED_SQRT_DIVISION_MIN));
}

/**
 * @brief Supplies real scalar operations used by the iterative solvers.
 *
 * @tparam T The real floating-point element type.
 */
template <typename T> struct Scalar {
  using Real = T;
  /**
   * @brief Returns the conjugate of a real value, which is the value itself.
   * @param value The input scalar.
   * @return The unchanged scalar.
   */
  static inline T conjugate(T value) { return value; }
  /**
   * @brief Returns the absolute value of the single real component.
   * @param value The input scalar.
   * @return The magnitude used for component scaling.
   */
  static inline T component_max(T value) { return Base::Math::abs(value); }
  /**
   * @brief Computes the squared magnitude without additional scaling.
   * @param value The input scalar whose square must be representable in T.
   * @return value multiplied by itself.
   */
  static inline T squared_abs(T value) { return value * value; }
  /**
   * @brief Computes the magnitude of a real scalar.
   * @param value The input scalar.
   * @return The absolute value of the scalar.
   */
  static inline T magnitude(T value) { return Base::Math::abs(value); }
  /**
   * @brief Divides real scalars using the configured denominator protection.
   * @param numerator The dividend.
   * @param denominator The divisor before protection.
   * @param floor The positive threshold passed to avoid_zero_divide.
   * @return The quotient using the protected denominator.
   */
  static inline T divide(T numerator, T denominator, T floor) {
    return numerator / Base::Utility::avoid_zero_divide(denominator, floor);
  }
};

/**
 * @brief Supplies scaled complex operations with a real magnitude type.
 *
 * @tparam T The real floating-point type of the complex components.
 */
template <typename T> struct Scalar<Complex<T>> {
  using Real = T;
  /**
   * @brief Computes the complex conjugate by negating the imaginary component.
   * @param value The input complex scalar.
   * @return The complex conjugate of value.
   */
  static inline Complex<T> conjugate(const Complex<T> &value) {
    return Complex<T>(value.real, -value.imag);
  }
  /**
   * @brief Finds the larger absolute component for subsequent scaling.
   * @param value The input complex scalar.
   * @return The maximum of the absolute real and imaginary components.
   */
  static inline T component_max(const Complex<T> &value) {
    T real = Base::Math::abs(value.real);
    T imag = Base::Math::abs(value.imag);
    return real > imag ? real : imag;
  }
  /**
   * @brief Computes the squared complex magnitude without additional scaling.
   * @param value The input whose component squares and sum must fit T.
   * @return The sum of the squared real and imaginary components.
   */
  static inline T squared_abs(const Complex<T> &value) {
    return value.real * value.real + value.imag * value.imag;
  }
  /**
   * @brief Computes the complex magnitude after scaling its components.
   *
   * Scaling bounds the intermediate squares; the final magnitude must still
   * be representable in T. The zero scalar is handled without division.
   *
   * @param value The input complex scalar.
   * @return The nonnegative magnitude using the Base::Math square root.
   */
  static inline T magnitude(const Complex<T> &value) {
    T scale = component_max(value);
    if (scale == static_cast<T>(0)) {
      return static_cast<T>(0);
    }
    // The scale is known nonzero. Do not clamp a valid tiny scale.
    Complex<T> normalized = value / scale;
    return scale * normalized_sqrt(squared_abs(normalized));
  }
  /**
   * @brief Divides complex scalars with magnitude-based denominator protection.
   *
   * A small nonzero denominator retains its phase while its magnitude is
   * raised to floor. A zero denominator is replaced by the positive real floor.
   * Scaling and the order of division limit intermediate overflow, but inputs
   * and the resulting quotient must remain finite and representable.
   *
   * @param numerator The complex dividend.
   * @param denominator The complex divisor before protection.
   * @param floor The positive minimum denominator magnitude.
   * @return The quotient using the scaled and protected denominator.
   */
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

/* Scaled vector norm */
namespace VectorNorm {

/**
 * @brief Scans active components in ascending index order for the norm scale.
 * @tparam T The real or complex vector element type.
 * @tparam N The fixed vector capacity.
 * @tparam Index The current component index, less than N.
 */
template <typename T, std::size_t N, std::size_t Index> struct ScaleLoop {
  using Real = typename Scalar<T>::Real;

  /**
   * @brief Updates the scale and recursively visits the remaining components.
   * @param value The vector to scan.
   * @param active_size The number of leading components to consider, up to N.
   * @param[in,out] scale The largest absolute component found so far.
   */
  static inline void compute(const Vector<T, N> &value, std::size_t active_size,
                             Real &scale) {
    if (Index < active_size) {
      Real candidate = Scalar<T>::component_max(value[Index]);
      if (candidate > scale) {
        scale = candidate;
      }
    }
    ScaleLoop<T, N, Index + 1>::compute(value, active_size, scale);
  }
};

/**
 * @brief Terminates the component scale scan at the fixed vector capacity.
 * @tparam T The real or complex vector element type.
 * @tparam N The fixed vector capacity and terminal index.
 */
template <typename T, std::size_t N> struct ScaleLoop<T, N, N> {
  using Real = typename Scalar<T>::Real;

  /**
   * @brief Ends the recursion without changing the accumulated scale.
   * @param value The unused input vector.
   * @param active_size The unused active component count.
   * @param scale The completed scale, left unchanged.
   */
  static inline void compute(const Vector<T, N> &value, std::size_t active_size,
                             Real &scale) {
    static_cast<void>(value);
    static_cast<void>(active_size);
    static_cast<void>(scale);
  }
};

/**
 * @brief Accumulates scaled squares in the same order as the for-loop path.
 * @tparam T The real or complex vector element type.
 * @tparam N The fixed vector capacity.
 * @tparam Index The current component index, less than N.
 */
template <typename T, std::size_t N, std::size_t Index> struct SquaredSumLoop {
  using Real = typename Scalar<T>::Real;

  /**
   * @brief Adds the current scaled square and visits the remaining components.
   * @param value The vector whose norm is being computed.
   * @param active_size The number of leading components to consider, up to N.
   * @param divide_by_scale The prepared division by the component scale.
   * @param[in,out] sum The accumulated scaled squared magnitude.
   */
  static inline void
  compute(const Vector<T, N> &value, std::size_t active_size,
          const Detail::RepeatedDivision<Real> &divide_by_scale, Real &sum) {
    if (Index < active_size) {
      sum += Scalar<T>::squared_abs(divide_by_scale.apply(value[Index]));
    }
    SquaredSumLoop<T, N, Index + 1>::compute(value, active_size,
                                             divide_by_scale, sum);
  }
};

/**
 * @brief Terminates the scaled-square accumulation at the vector capacity.
 * @tparam T The real or complex vector element type.
 * @tparam N The fixed vector capacity and terminal index.
 */
template <typename T, std::size_t N> struct SquaredSumLoop<T, N, N> {
  using Real = typename Scalar<T>::Real;

  /**
   * @brief Ends the recursion without changing the accumulated squared sum.
   * @param value The unused input vector.
   * @param active_size The unused active component count.
   * @param divide_by_scale The unused prepared division.
   * @param sum The completed sum, left unchanged.
   */
  static inline void
  compute(const Vector<T, N> &value, std::size_t active_size,
          const Detail::RepeatedDivision<Real> &divide_by_scale, Real &sum) {
    static_cast<void>(value);
    static_cast<void>(active_size);
    static_cast<void>(divide_by_scale);
    static_cast<void>(sum);
  }
};

template <typename T, std::size_t N>
inline typename Scalar<T>::Real component_scale(const Vector<T, N> &value,
                                                std::size_t active_size) {
  using Real = typename Scalar<T>::Real;
  Real scale = static_cast<Real>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < N; ++i) {
    if (i < active_size) {
      Real candidate = Scalar<T>::component_max(value[i]);
      if (candidate > scale) {
        scale = candidate;
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  ScaleLoop<T, N, 0>::compute(value, active_size, scale);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return scale;
}

template <typename T, std::size_t N>
inline typename Scalar<T>::Real scaled_squared_sum(
    const Vector<T, N> &value, std::size_t active_size,
    const Detail::RepeatedDivision<typename Scalar<T>::Real> &divide_by_scale) {
  using Real = typename Scalar<T>::Real;
  Real sum = static_cast<Real>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < N; ++i) {
    if (i < active_size) {
      sum += Scalar<T>::squared_abs(divide_by_scale.apply(value[i]));
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  SquaredSumLoop<T, N, 0>::compute(value, active_size, divide_by_scale, sum);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return sum;
}

} // namespace VectorNorm

/**
 * @brief Computes a scaled Euclidean norm over the leading vector components.
 *
 * A first scan finds the component scale, and a second scan accumulates
 * normalized squares. Zero or empty input returns zero without division.
 * This avoids squaring unscaled values; the final norm must still fit Real.
 *
 * @tparam T The real or complex vector element type.
 * @tparam N The fixed vector capacity.
 * @param value The input vector.
 * @param active_size The number of leading components to include, up to N.
 * @return The real norm using the configured Base::Math square-root accuracy.
 */
template <typename T, std::size_t N>
inline typename Scalar<T>::Real norm(const Vector<T, N> &value,
                                     std::size_t active_size = N) {
  using Real = typename Scalar<T>::Real;
  const Real scale = VectorNorm::component_scale(value, active_size);
  if (scale == static_cast<Real>(0)) {
    return static_cast<Real>(0);
  }
  const Detail::RepeatedDivision<Real> divide_by_scale(scale);
  const Real sum =
      VectorNorm::scaled_squared_sum(value, active_size, divide_by_scale);
  return scale * normalized_sqrt(sum);
}

/**
 * @brief Computes the reciprocal of the scaled Euclidean norm.
 *
 * Uses the same two-scan scaling as norm(), then evaluates
 * 1 / (scale * sqrt(sum)) as rsqrt(sum) / scale, so no separate square root
 * and division of the full norm is needed. Because sum is in [1, 2*N] after
 * scaling, rsqrt is always well-conditioned. A zero or empty vector returns
 * zero instead of dividing by zero.
 *
 * @tparam T The real or complex vector element type.
 * @tparam N The fixed vector capacity.
 * @param value The input vector.
 * @param active_size The number of leading components to include, up to N.
 * @return 1 / norm(value, active_size), or zero if the norm is zero.
 */
template <typename T, std::size_t N>
inline typename Scalar<T>::Real norm_inv(const Vector<T, N> &value,
                                         std::size_t active_size = N) {
  using Real = typename Scalar<T>::Real;
  const Real scale = VectorNorm::component_scale(value, active_size);
  if (scale == static_cast<Real>(0)) {
    return static_cast<Real>(0);
  }
  const Detail::RepeatedDivision<Real> divide_by_scale(scale);
  const Real sum =
      VectorNorm::scaled_squared_sum(value, active_size, divide_by_scale);
  return divide_by_scale.apply(Base::Math::rsqrt<Real>(
      sum, static_cast<Real>(NORMALIZED_SQRT_DIVISION_MIN)));
}

/**
 * @brief Computes scaled Givens coefficients for a real or complex pair.
 *
 * The rotation is [c s; -conj(s) c], where c is real. Its second component
 * is eliminated up to numerical error. A zero second input gives the identity
 * rotation. Square-root accuracy follows the configured Base::Math functions.
 *
 * @tparam T The real or complex scalar type.
 * @param first The first component of the vector to rotate.
 * @param second The second component of the vector to rotate.
 * @param[out] cosine The real coefficient c.
 * @param[out] sine The real or complex coefficient s.
 */
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
