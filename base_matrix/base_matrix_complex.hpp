#ifndef BASE_MATRIX_COMPLEX_HPP
#define BASE_MATRIX_COMPLEX_HPP

#include "base_math.hpp"
#include "base_matrix_utility.hpp"

#include <cstddef>
#include <utility>

namespace Base {
namespace Matrix {

template <typename T> class Complex {
public:
  Complex() : real(static_cast<T>(0)), imag(static_cast<T>(0)) {}

  Complex(const T &real) : real(real), imag(static_cast<T>(0)) {}

  Complex(const T &real, const T &imag) : real(real), imag(imag) {}

  /* Copy Constructor */
  Complex(const Complex<T> &other) : real(other.real), imag(other.imag) {}

  Complex<T> &operator=(const Complex<T> &other) {
    if (this != &other) {
      this->real = other.real;
      this->imag = other.imag;
    }
    return *this;
  }

  /* Move Constructor */
  Complex(Complex<T> &&other) noexcept
      : real(std::move(other.real)), imag(std::move(other.imag)) {}

  Complex<T> &operator=(Complex<T> &&other) noexcept {
    if (this != &other) {
      this->real = std::move(other.real);
      this->imag = std::move(other.imag);
    }
    return *this;
  }

  /* Method */
  Complex<T> operator+(const Complex<T> &a_comp) const {
    Complex<T> result;

    result.real = this->real + a_comp.real;
    result.imag = this->imag + a_comp.imag;

    return result;
  }

  Complex<T> operator-(const Complex<T> &a_comp) const {
    Complex<T> result;

    result.real = this->real - a_comp.real;
    result.imag = this->imag - a_comp.imag;

    return result;
  }

  Complex<T> operator-(void) const {
    Complex<T> result;

    result.real = -this->real;
    result.imag = -this->imag;

    return result;
  }

  Complex<T> operator*(const Complex<T> &a_comp) const {
    Complex<T> result;

    result.real = this->real * a_comp.real - this->imag * a_comp.imag;
    result.imag = this->real * a_comp.imag + this->imag * a_comp.real;

    return result;
  }

  void operator+=(const T &a) { this->real += a; }

  void operator-=(const T &a) { this->real -= a; }

  void operator*=(const T &a) {
    this->real *= a;
    this->imag *= a;
  }

  void operator+=(const Complex<T> &a_comp) {
    this->real += a_comp.real;
    this->imag += a_comp.imag;
  }

  void operator-=(const Complex<T> &a_comp) {
    this->real -= a_comp.real;
    this->imag -= a_comp.imag;
  }

  void operator*=(const Complex<T> &a_comp) {

    T real_temp = this->real * a_comp.real - this->imag * a_comp.imag;
    T imag_temp = this->real * a_comp.imag + this->imag * a_comp.real;

    this->real = real_temp;
    this->imag = imag_temp;
  }

  /* Property */
  T real = static_cast<T>(0);
  T imag = static_cast<T>(0);
};

/* Scalar */
template <typename T> Complex<T> operator+(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real + b;
  result.imag = a_comp.imag;

  return result;
}

template <typename T> Complex<T> operator+(T b, const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = b + a_comp.real;
  result.imag = a_comp.imag;

  return result;
}

template <typename T> Complex<T> operator-(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real - b;
  result.imag = a_comp.imag;

  return result;
}

template <typename T> Complex<T> operator-(T b, const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = b - a_comp.real;
  result.imag = -a_comp.imag;

  return result;
}

template <typename T> Complex<T> operator*(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real * b;
  result.imag = a_comp.imag * b;

  return result;
}

template <typename T> Complex<T> operator*(T b, const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = b * a_comp.real;
  result.imag = b * a_comp.imag;

  return result;
}

template <typename T> Complex<T> operator/(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real / b;
  result.imag = a_comp.imag / b;

  return result;
}

/* complex functions */
template <typename T>
Complex<T> complex_divide(const Complex<T> &a_comp, const Complex<T> &b_comp,
                          T division_min) {
  Complex<T> result;

  T denominator = avoid_zero_divide(
      b_comp.real * b_comp.real + b_comp.imag * b_comp.imag, division_min);

  result.real =
      (a_comp.real * b_comp.real + a_comp.imag * b_comp.imag) / denominator;
  result.imag =
      (a_comp.imag * b_comp.real - a_comp.real * b_comp.imag) / denominator;

  return result;
}

template <typename T>
Complex<T> complex_divide(T a, const Complex<T> &b_comp, T division_min) {
  Complex<T> result;

  T denominator = avoid_zero_divide(
      b_comp.real * b_comp.real + b_comp.imag * b_comp.imag, division_min);

  result.real = (a * b_comp.real) / denominator;
  result.imag = (-a * b_comp.imag) / denominator;

  return result;
}

template <typename T> T complex_abs(const Complex<T> &a_comp) {
  T result;

  result =
      Base::Math::sqrt(a_comp.real * a_comp.real + a_comp.imag * a_comp.imag);

  return result;
}

template <typename T> T complex_abs_sq(const Complex<T> &a_comp) {
  T result;

  result = a_comp.real * a_comp.real + a_comp.imag * a_comp.imag;

  return result;
}

template <typename T> T complex_phase(const Complex<T> &a_comp) {
  T result;

  result = Base::Math::atan2(a_comp.imag, a_comp.real);

  return result;
}

template <typename T> Complex<T> complex_conjugate(const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = a_comp.real;
  result.imag = -a_comp.imag;

  return result;
}

template <typename T> Complex<T> complex_sqrt(const Complex<T> &a_comp) {
  Complex<T> result;

  T a_abs =
      Base::Math::sqrt(a_comp.real * a_comp.real + a_comp.imag * a_comp.imag);

  result.real = Base::Math::sqrt((a_comp.real + a_abs) * static_cast<T>(0.5));

  if (a_comp.imag >= 0) {
    result.imag =
        Base::Math::sqrt((-a_comp.real + a_abs) * static_cast<T>(0.5));
  } else {
    result.imag =
        -Base::Math::sqrt((-a_comp.real + a_abs) * static_cast<T>(0.5));
  }

  return result;
}

template <typename T>
Complex<T> complex_sign(const Complex<T> &a_comp, T division_min) {
  Complex<T> result;

  T a_abs =
      Base::Math::sqrt(a_comp.real * a_comp.real + a_comp.imag * a_comp.imag);

  if (a_abs >= division_min) {
    result.real = a_comp.real / a_abs;
    result.imag = a_comp.imag / a_abs;
  } else {
    result.real = static_cast<T>(0);
    result.imag = static_cast<T>(0);
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPLEX_HPP
