#ifndef BASE_MATRIX_RECIPROCAL_HPP_
#define BASE_MATRIX_RECIPROCAL_HPP_

#include <limits>
#include <type_traits>

namespace Base {
namespace Matrix {
namespace Detail {

constexpr bool BASE_MATRIX_RECIPROCAL_KEEP_PRECISION = false;

// Keep the scalar's division semantics for integer and user-defined types.
template <typename T, bool = std::is_floating_point<T>::value>
class RepeatedDivision {
public:
  explicit RepeatedDivision(T denominator) : denominator_(denominator) {}

  /* Copy Constructor */
  RepeatedDivision(const RepeatedDivision &other)
      : denominator_(other.denominator_) {}

  RepeatedDivision<T, false> &operator=(const RepeatedDivision &other) {
    if (this != &other) {
      denominator_ = other.denominator_;
    }
    return *this;
  }

  /* Move Constructor */
  RepeatedDivision(RepeatedDivision &&other) noexcept
      : denominator_(other.denominator_) {}

  RepeatedDivision<T, false> &operator=(RepeatedDivision &&other) noexcept {
    if (this != &other) {
      denominator_ = other.denominator_;
    }
    return *this;
  }

  /* Function */
  template <typename U> inline U apply(const U &value) const {
    return value / denominator_;
  }

private:
  T denominator_;
};

template <typename T> class RepeatedDivision<T, true> {
public:
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
