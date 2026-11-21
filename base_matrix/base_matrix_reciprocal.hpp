/**
 * @file base_matrix_reciprocal.hpp
 * @brief Provides reusable division by a common denominator.
 *
 * Floating-point operations can compute a reciprocal before a loop and reuse
 * it for each dividend. The precision setting controls when these operations
 * use direct division for extreme denominators. Integer and user-defined
 * scalar types use their division operator.
 */
#ifndef BASE_MATRIX_RECIPROCAL_HPP_
#define BASE_MATRIX_RECIPROCAL_HPP_

#include <limits>
#include <type_traits>

namespace Base {
namespace Matrix {
namespace Detail {

/**
 * @brief Selects range checks for floating-point reciprocals.
 *
 * When true, direct division is used if the denominator or its reciprocal
 * falls outside the normal range. When false, the reciprocal is always cached
 * and can overflow or lose precision for extreme denominators.
 */
constexpr bool BASE_MATRIX_RECIPROCAL_KEEP_PRECISION = false;

/**
 * @brief Stores a common denominator for direct division.
 *
 * The second template argument selects the floating-point specializations.
 * The primary template preserves the scalar type's division semantics.
 *
 * @tparam T The denominator type.
 * @tparam KeepPrecision The precision setting for floating-point division.
 */
template <typename T, bool = std::is_floating_point<T>::value,
          bool KeepPrecision = BASE_MATRIX_RECIPROCAL_KEEP_PRECISION>
class RepeatedDivision {
public:
  /**
   * @brief Stores the denominator for subsequent divisions.
   * @param denominator The common divisor.
   */
  explicit RepeatedDivision(T denominator) : denominator_(denominator) {}

  /* Copy Constructor */
  RepeatedDivision(const RepeatedDivision &other)
      : denominator_(other.denominator_) {}

  /* Copy Assignment Operator */
  RepeatedDivision &operator=(const RepeatedDivision &other) {
    if (this != &other) {
      denominator_ = other.denominator_;
    }
    return *this;
  }

  /* Move Constructor */
  RepeatedDivision(RepeatedDivision &&other) noexcept
      : denominator_(other.denominator_) {}

  /* Move Assignment Operator */
  RepeatedDivision &operator=(RepeatedDivision &&other) noexcept {
    if (this != &other) {
      denominator_ = other.denominator_;
    }
    return *this;
  }

  /**
   * @brief Divides a value by the stored denominator.
   * @tparam U The dividend type, which supports division by T.
   * @param value The dividend.
   * @return The quotient computed by the division operator.
   */
  template <typename U> inline U apply(const U &value) const {
    return value / denominator_;
  }

private:
  T denominator_;
};

/**
 * @brief Reuses one floating-point reciprocal for all dividends.
 * @tparam T The real floating-point type of the denominator.
 */
template <typename T> class RepeatedDivision<T, true, false> {
public:
  /**
   * @brief Computes and stores the reciprocal of the denominator.
   * @param denominator The common divisor whose reciprocal is computed in T.
   */
  explicit RepeatedDivision(T denominator)
      : inverse_(static_cast<T>(1) / denominator) {}

  /* Copy Constructor */
  RepeatedDivision(const RepeatedDivision &other) : inverse_(other.inverse_) {}

  /* Copy Assignment Operator */
  RepeatedDivision &operator=(const RepeatedDivision &other) {
    if (this != &other) {
      inverse_ = other.inverse_;
    }
    return *this;
  }

  /* Move Constructor */
  RepeatedDivision(RepeatedDivision &&other) noexcept
      : inverse_(other.inverse_) {}

  /* Move Assignment Operator */
  RepeatedDivision &operator=(RepeatedDivision &&other) noexcept {
    if (this != &other) {
      inverse_ = other.inverse_;
    }
    return *this;
  }

  /**
   * @brief Multiplies a value by the stored reciprocal.
   * @tparam U The dividend type, which supports multiplication by T.
   * @param value The dividend.
   * @return The product of value and the stored reciprocal.
   */
  template <typename U> inline U apply(const U &value) const {
    return value * inverse_;
  }

private:
  T inverse_;
};

/**
 * @brief Reuses a floating-point reciprocal within the normal range.
 * @tparam T The real floating-point type of the denominator.
 */
template <typename T> class RepeatedDivision<T, true, true> {
public:
  /**
   * @brief Stores the denominator and caches its reciprocal when both are
   * finite and normal.
   * @param denominator The common divisor.
   */
  explicit RepeatedDivision(T denominator)
      : denominator_(denominator), inverse_(static_cast<T>(0)) {
    const T magnitude =
        denominator < static_cast<T>(0) ? -denominator : denominator;
    constexpr T min_normal = (std::numeric_limits<T>::min)();
    constexpr T max_denominator = static_cast<T>(1) / min_normal;
    // Cache the reciprocal only when both values are finite and normal.
    if (magnitude >= min_normal && magnitude <= max_denominator) {
      inverse_ = static_cast<T>(1) / denominator;
    }
  }

  /* Copy Constructor */
  RepeatedDivision(const RepeatedDivision &other)
      : denominator_(other.denominator_), inverse_(other.inverse_) {}

  /* Copy Assignment Operator */
  RepeatedDivision &operator=(const RepeatedDivision &other) {
    if (this != &other) {
      denominator_ = other.denominator_;
      inverse_ = other.inverse_;
    }
    return *this;
  }

  /* Move Constructor */
  RepeatedDivision(RepeatedDivision &&other) noexcept
      : denominator_(other.denominator_), inverse_(other.inverse_) {}

  /* Move Assignment Operator */
  RepeatedDivision &operator=(RepeatedDivision &&other) noexcept {
    if (this != &other) {
      denominator_ = other.denominator_;
      inverse_ = other.inverse_;
    }
    return *this;
  }

  /**
   * @brief Uses the cached reciprocal when available and divides directly
   * otherwise.
   * @tparam U The dividend type, which supports multiplication and division
   * by T.
   * @param value The dividend.
   * @return The quotient computed by multiplication or direct division.
   */
  template <typename U> inline U apply(const U &value) const {
    return inverse_ != static_cast<T>(0) ? value * inverse_
                                         : value / denominator_;
  }

private:
  T denominator_;
  T inverse_;
};

} // namespace Detail
} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_RECIPROCAL_HPP_
