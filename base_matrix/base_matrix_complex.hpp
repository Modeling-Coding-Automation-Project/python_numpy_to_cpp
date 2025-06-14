/**
 * @file base_matrix_complex.hpp
 * @brief Defines a templated Complex number class and related operations for
 * matrix computations.
 *
 * This file provides the implementation of a generic Complex<T> class within
 * the Base::Matrix namespace, supporting basic arithmetic, assignment, and
 * comparison operators for complex numbers. It also includes a set of utility
 * functions for complex arithmetic, such as division, absolute value, phase,
 * conjugate, square root, and sign. The code is designed to be used in
 * mathematical and matrix computation contexts, with additional type traits to
 * identify complex types.
 *
 * Classes:
 * - Complex<T>: A template class representing a complex number with real and
 * imaginary parts of type T. Provides constructors, copy/move semantics,
 * arithmetic operators, and compound assignment operators for complex number
 * manipulation.
 * - Is_Complex_Type<T>: Type trait to determine if a type is a specialization
 * of Complex<T>.
 */
#ifndef __BASE_MATRIX_COMPLEX_HPP__
#define __BASE_MATRIX_COMPLEX_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_utility.hpp"

#include <cstddef>
#include <utility>

namespace Base {
namespace Matrix {

/**
 * @brief A templated Complex number class.
 *
 * This class represents a complex number with real and imaginary parts of
 * type T. It supports basic arithmetic operations, assignment, and comparison
 * operators.
 *
 * @tparam T The type of the real and imaginary parts of the complex number.
 */
template <typename T> class Complex {
public:
  /* Type */
  using Value_Type = T;

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

  /**
   * @brief Adds two Complex numbers.
   *
   * This operator overloads the + operator to perform element-wise addition
   * of the real and imaginary parts of two Complex numbers.
   *
   * @param a_comp The Complex number to add to this instance.
   * @return A new Complex number representing the sum of this and a_comp.
   */
  inline Complex<T> operator+(const Complex<T> &a_comp) const {
    Complex<T> result;

    result.real = this->real + a_comp.real;
    result.imag = this->imag + a_comp.imag;

    return result;
  }

  /**
   * @brief Subtracts two Complex numbers.
   *
   * This operator overloads the - operator to perform element-wise subtraction
   * of the real and imaginary parts of two Complex numbers.
   *
   * @param a_comp The Complex number to subtract from this instance.
   * @return A new Complex number representing the difference of this and
   * a_comp.
   */
  inline Complex<T> operator-(const Complex<T> &a_comp) const {
    Complex<T> result;

    result.real = this->real - a_comp.real;
    result.imag = this->imag - a_comp.imag;

    return result;
  }

  /**
   * @brief Negates the Complex number.
   *
   * This operator overloads the unary - operator to negate both the real and
   * imaginary parts of the Complex number.
   *
   * @return A new Complex number representing the negation of this instance.
   */
  inline Complex<T> operator-(void) const {
    Complex<T> result;

    result.real = -this->real;
    result.imag = -this->imag;

    return result;
  }

  /**
   * @brief Multiplies two Complex numbers.
   *
   * This operator overloads the * operator to perform multiplication of two
   * Complex numbers using the formula (a + bi)(c + di) = (ac - bd) + (ad +
   * bc)i.
   *
   * @param a_comp The Complex number to multiply with this instance.
   * @return A new Complex number representing the product of this and a_comp.
   */
  inline Complex<T> operator*(const Complex<T> &a_comp) const {
    Complex<T> result;

    result.real = this->real * a_comp.real - this->imag * a_comp.imag;
    result.imag = this->real * a_comp.imag + this->imag * a_comp.real;

    return result;
  }

  /**
   * @brief Divides two Complex numbers.
   *
   * This operator overloads the / operator to perform division of two Complex
   * numbers using the formula (a + bi)/(c + di) = ((ac + bd)/(c^2 + d^2)) +
   * ((bc - ad)/(c^2 + d^2))i.
   *
   * @param a_comp The Complex number to divide this instance by.
   * @return A new Complex number representing the quotient of this and a_comp.
   */
  inline void operator+=(const T &a) { this->real += a; }

  /**
   * @brief Subtracts a scalar from the Complex number.
   *
   * This operator overloads the -= operator to perform element-wise
   * subtraction of a scalar from the real part of the Complex number.
   *
   * @param a The scalar to subtract from the real part of this instance.
   */
  inline void operator-=(const T &a) { this->real -= a; }

  /**
   * @brief Multiplies the Complex number by a scalar.
   *
   * This operator overloads the *= operator to perform element-wise
   * multiplication of the real and imaginary parts of the Complex number by a
   * scalar.
   *
   * @param a The scalar to multiply both the real and imaginary parts of this
   * instance.
   */
  inline void operator*=(const T &a) {
    this->real *= a;
    this->imag *= a;
  }

  /**
   * @brief Divides the Complex number by a scalar.
   *
   * This operator overloads the /= operator to perform element-wise division
   * of the real and imaginary parts of the Complex number by a scalar.
   *
   * @param a The scalar to divide both the real and imaginary parts of this
   * instance.
   */
  inline void operator+=(const Complex<T> &a_comp) {
    this->real += a_comp.real;
    this->imag += a_comp.imag;
  }

  /**
   * @brief Subtracts a Complex number from this instance.
   *
   * This operator overloads the -= operator to perform element-wise
   * subtraction of the real and imaginary parts of a Complex number from this
   * instance.
   *
   * @param a_comp The Complex number to subtract from this instance.
   */
  inline void operator-=(const Complex<T> &a_comp) {
    this->real -= a_comp.real;
    this->imag -= a_comp.imag;
  }

  /**
   * @brief Multiplies this Complex number by another Complex number.
   *
   * This operator overloads the *= operator to perform multiplication of two
   * Complex numbers using the formula (a + bi)(c + di) = (ac - bd) + (ad +
   * bc)i.
   *
   * @param a_comp The Complex number to multiply with this instance.
   */
  inline void operator*=(const Complex<T> &a_comp) {

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

/**
 * @brief Adds a scalar to a Complex number.
 *
 * This operator overloads the + operator to add a scalar to the real part of
 * a Complex number, leaving the imaginary part unchanged.
 *
 * @param a_comp The Complex number to add the scalar to.
 * @param b The scalar to add to the real part of a_comp.
 * @return A new Complex number representing the sum of a_comp and b.
 */
template <typename T>
inline Complex<T> operator+(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real + b;
  result.imag = a_comp.imag;

  return result;
}

/**
 * @brief Adds a Complex number to a scalar.
 *
 * This operator overloads the + operator to add a Complex number to a scalar,
 * effectively adding the scalar to the real part of the Complex number.
 *
 * @param b The scalar to add to the real part of a_comp.
 * @param a_comp The Complex number to which the scalar is added.
 * @return A new Complex number representing the sum of b and a_comp.
 */
template <typename T>
inline Complex<T> operator+(T b, const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = b + a_comp.real;
  result.imag = a_comp.imag;

  return result;
}

/**
 * @brief Subtracts a scalar from a Complex number.
 *
 * This operator overloads the - operator to subtract a scalar from the real
 * part of a Complex number, leaving the imaginary part unchanged.
 *
 * @param a_comp The Complex number from which the scalar is subtracted.
 * @param b The scalar to subtract from the real part of a_comp.
 * @return A new Complex number representing the difference of a_comp and b.
 */
template <typename T>
inline Complex<T> operator-(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real - b;
  result.imag = a_comp.imag;

  return result;
}

/**
 * @brief Subtracts a Complex number from a scalar.
 *
 * This operator overloads the - operator to subtract a Complex number from a
 * scalar, effectively subtracting the real part of the Complex number from the
 * scalar and negating the imaginary part.
 *
 * @param b The scalar from which the Complex number is subtracted.
 * @param a_comp The Complex number to subtract from b.
 * @return A new Complex number representing the difference of b and a_comp.
 */
template <typename T>
inline Complex<T> operator-(T b, const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = b - a_comp.real;
  result.imag = -a_comp.imag;

  return result;
}

/**
 * @brief Multiplies a Complex number by a scalar.
 *
 * This operator overloads the * operator to multiply both the real and
 * imaginary parts of a Complex number by a scalar.
 *
 * @param a_comp The Complex number to multiply by the scalar.
 * @param b The scalar to multiply both the real and imaginary parts of a_comp.
 * @return A new Complex number representing the product of a_comp and b.
 */
template <typename T>
inline Complex<T> operator*(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real * b;
  result.imag = a_comp.imag * b;

  return result;
}

/**
 * @brief Multiplies a scalar by a Complex number.
 *
 * This operator overloads the * operator to multiply both the real and
 * imaginary parts of a Complex number by a scalar.
 *
 * @param b The scalar to multiply both the real and imaginary parts of a_comp.
 * @param a_comp The Complex number to multiply by the scalar.
 * @return A new Complex number representing the product of b and a_comp.
 */
template <typename T>
inline Complex<T> operator*(T b, const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = b * a_comp.real;
  result.imag = b * a_comp.imag;

  return result;
}

/**
 * @brief Divides a Complex number by a scalar.
 *
 * This operator overloads the / operator to divide both the real and
 * imaginary parts of a Complex number by a scalar.
 *
 * @param a_comp The Complex number to divide by the scalar.
 * @param b The scalar to divide both the real and imaginary parts of a_comp.
 * @return A new Complex number representing the quotient of a_comp and b.
 */
template <typename T>
inline Complex<T> operator/(const Complex<T> &a_comp, T b) {
  Complex<T> result;

  result.real = a_comp.real / b;
  result.imag = a_comp.imag / b;

  return result;
}

/**
 * @brief Divides a scalar by a Complex number.
 *
 * This operator overloads the / operator to divide a scalar by a Complex
 * number, effectively dividing the scalar by the magnitude of the Complex
 * number and adjusting the real and imaginary parts accordingly.
 *
 * @param b The scalar to divide by the Complex number.
 * @param a_comp The Complex number by which the scalar is divided.
 * @return A new Complex number representing the quotient of b and a_comp.
 */
template <typename T>
inline bool operator<(const Complex<T> &a_comp, const Complex<T> &b_comp) {
  bool result = false;

  if ((a_comp.real + a_comp.imag) < (b_comp.real + b_comp.imag)) {
    result = true;
  } else {
    result = false;
  }

  return result;
}

/**
 * @brief Compares two Complex numbers for less than or equal to.
 *
 * This operator overloads the <= operator to compare the sum of the real and
 * imaginary parts of two Complex numbers.
 *
 * @param a_comp The first Complex number to compare.
 * @param b_comp The second Complex number to compare.
 * @return True if the sum of a_comp is less than or equal to the sum of
 * b_comp, false otherwise.
 */
template <typename T>
inline bool operator<=(const Complex<T> &a_comp, const Complex<T> &b_comp) {
  bool result = false;

  if ((a_comp.real + a_comp.imag) <= (b_comp.real + b_comp.imag)) {
    result = true;
  } else {
    result = false;
  }

  return result;
}

/**
 * @brief Compares two Complex numbers for greater than.
 *
 * This operator overloads the > operator to compare the sum of the real and
 * imaginary parts of two Complex numbers.
 *
 * @param a_comp The first Complex number to compare.
 * @param b_comp The second Complex number to compare.
 * @return True if the sum of a_comp is greater than the sum of b_comp, false
 * otherwise.
 */
template <typename T>
inline bool operator>(const Complex<T> &a_comp, const Complex<T> &b_comp) {
  bool result = false;

  if ((a_comp.real + a_comp.imag) > (b_comp.real + b_comp.imag)) {
    result = true;
  } else {
    result = false;
  }

  return result;
}

/**
 * @brief Compares two Complex numbers for greater than or equal to.
 *
 * This operator overloads the >= operator to compare the sum of the real and
 * imaginary parts of two Complex numbers.
 *
 * @param a_comp The first Complex number to compare.
 * @param b_comp The second Complex number to compare.
 * @return True if the sum of a_comp is greater than or equal to the sum of
 * b_comp, false otherwise.
 */
template <typename T>
inline bool operator>=(const Complex<T> &a_comp, const Complex<T> &b_comp) {
  bool result = false;

  if ((a_comp.real + a_comp.imag) >= (b_comp.real + b_comp.imag)) {
    result = true;
  } else {
    result = false;
  }

  return result;
}

/* complex functions */

/**
 * @brief Divides two Complex numbers.
 *
 * This function performs division of two Complex numbers using the formula
 * (a + bi)/(c + di) = ((ac + bd)/(c^2 + d^2)) + ((bc - ad)/(c^2 + d^2))i,
 * avoiding division by zero using a minimum value.
 *
 * @param a_comp The Complex number to be divided.
 * @param b_comp The Complex number to divide by.
 * @param division_min The minimum value to avoid division by zero.
 * @return A new Complex number representing the quotient of a_comp and b_comp.
 */
template <typename T>
inline Complex<T> complex_divide(const Complex<T> &a_comp,
                                 const Complex<T> &b_comp,
                                 const T &division_min) {
  Complex<T> result;

  T denominator = Base::Utility::avoid_zero_divide(
      b_comp.real * b_comp.real + b_comp.imag * b_comp.imag, division_min);

  result.real =
      (a_comp.real * b_comp.real + a_comp.imag * b_comp.imag) / denominator;
  result.imag =
      (a_comp.imag * b_comp.real - a_comp.real * b_comp.imag) / denominator;

  return result;
}

/**
 * @brief Divides a scalar by a Complex number.
 *
 * This function performs division of a scalar by a Complex number using the
 * formula (a)/(c + di) = (a*c)/(c^2 + d^2) - (a*d)/(c^2 + d^2)i, avoiding
 * division by zero using a minimum value.
 *
 * @param a The scalar to be divided.
 * @param b_comp The Complex number to divide by.
 * @param division_min The minimum value to avoid division by zero.
 * @return A new Complex number representing the quotient of a and b_comp.
 */
template <typename T>
inline Complex<T> complex_divide(T a, const Complex<T> &b_comp,
                                 const T &division_min) {
  Complex<T> result;

  T denominator = Base::Utility::avoid_zero_divide(
      b_comp.real * b_comp.real + b_comp.imag * b_comp.imag, division_min);

  result.real = (a * b_comp.real) / denominator;
  result.imag = (-a * b_comp.imag) / denominator;

  return result;
}

/**
 * @brief Divides a Complex number by a scalar.
 *
 * This function performs division of a Complex number by a scalar, effectively
 * dividing both the real and imaginary parts by the scalar.
 *
 * @param a_comp The Complex number to be divided.
 * @param b The scalar to divide both the real and imaginary parts of a_comp.
 * @return A new Complex number representing the quotient of a_comp and b.
 */
template <typename T> inline T complex_abs(const Complex<T> &a_comp) {
  T result;

  result = Base::Math::sqrt<T>(a_comp.real * a_comp.real +
                               a_comp.imag * a_comp.imag);

  return result;
}

/**
 * @brief Computes the squared absolute value of a Complex number.
 *
 * This function calculates the squared absolute value of a Complex number
 * using the formula |a + bi|^2 = a^2 + b^2.
 *
 * @param a_comp The Complex number whose squared absolute value is computed.
 * @return The squared absolute value of a_comp.
 */
template <typename T> inline T complex_abs_sq(const Complex<T> &a_comp) {
  T result;

  result = a_comp.real * a_comp.real + a_comp.imag * a_comp.imag;

  return result;
}

/**
 * @brief Computes the phase (angle) of a Complex number.
 *
 * This function calculates the phase of a Complex number using the
 * atan2 function, which computes the angle in radians between the positive
 * x-axis and the line to the point (real, imag).
 *
 * @param a_comp The Complex number whose phase is computed.
 * @return The phase of a_comp in radians.
 */
template <typename T> inline T complex_phase(const Complex<T> &a_comp) {
  T result;

  result = Base::Math::atan2(a_comp.imag, a_comp.real);

  return result;
}

/**
 * @brief Computes the complex conjugate of a Complex number.
 *
 * This function returns a new Complex number with the same real part and the
 * negated imaginary part of the input Complex number.
 *
 * @param a_comp The Complex number whose conjugate is computed.
 * @return A new Complex number representing the conjugate of a_comp.
 */
template <typename T>
inline Complex<T> complex_conjugate(const Complex<T> &a_comp) {
  Complex<T> result;

  result.real = a_comp.real;
  result.imag = -a_comp.imag;

  return result;
}

/**
 * @brief Computes the square root of a Complex number.
 *
 * This function calculates the square root of a Complex number using a
 * method that avoids division by zero, returning a new Complex number
 * representing the square root.
 *
 * @param a_comp The Complex number whose square root is computed.
 * @return A new Complex number representing the square root of a_comp.
 */
template <typename T> inline Complex<T> complex_sqrt(const Complex<T> &a_comp) {
  Complex<T> result;

  T a_abs = Base::Math::sqrt<T>(a_comp.real * a_comp.real +
                                a_comp.imag * a_comp.imag);

  result.real =
      Base::Math::sqrt<T>((a_comp.real + a_abs) * static_cast<T>(0.5));

  if (a_comp.imag >= 0) {
    result.imag =
        Base::Math::sqrt<T>((-a_comp.real + a_abs) * static_cast<T>(0.5));
  } else {
    result.imag =
        -Base::Math::sqrt<T>((-a_comp.real + a_abs) * static_cast<T>(0.5));
  }

  return result;
}

/**
 * @brief Computes the square root of a Complex number with a minimum division
 * value.
 *
 * This function calculates the square root of a Complex number, ensuring that
 * division by zero is avoided by using a specified minimum value for division.
 *
 * @param a_comp The Complex number whose square root is computed.
 * @param division_min The minimum value to avoid division by zero.
 * @return A new Complex number representing the square root of a_comp.
 */
template <typename T>
inline Complex<T> complex_sqrt(const Complex<T> &a_comp,
                               const T &division_min) {
  Complex<T> result;

  T a_abs = Base::Math::sqrt<T>(
      a_comp.real * a_comp.real + a_comp.imag * a_comp.imag, division_min);

  result.real = Base::Math::sqrt<T>((a_comp.real + a_abs) * static_cast<T>(0.5),
                                    division_min);

  if (a_comp.imag >= 0) {
    result.imag = Base::Math::sqrt<T>(
        (-a_comp.real + a_abs) * static_cast<T>(0.5), division_min);
  } else {
    result.imag = -Base::Math::sqrt<T>(
        (-a_comp.real + a_abs) * static_cast<T>(0.5), division_min);
  }

  return result;
}

/**
 * @brief Computes the sign of a Complex number.
 *
 * This function calculates the sign of a Complex number by normalizing it to
 * have a magnitude of 1, effectively returning a new Complex number with the
 * same direction as the input but with a magnitude of 1.
 *
 * @param a_comp The Complex number whose sign is computed.
 * @return A new Complex number representing the sign of a_comp.
 */
template <typename T>
inline Complex<T> complex_sign(const Complex<T> &a_comp,
                               const T &division_min) {
  Complex<T> result;

  T a_abs_r = Base::Math::rsqrt<T>(
      a_comp.real * a_comp.real + a_comp.imag * a_comp.imag, division_min);

  result.real = a_comp.real * a_abs_r;
  result.imag = a_comp.imag * a_abs_r;

  return result;
}

/* Tmeplate to judge Complex type or not */
// default
template <typename T> struct Is_Complex_Type : std::false_type {};

// is Complex
template <typename T> struct Is_Complex_Type<Complex<T>> : std::true_type {};

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_COMPLEX_HPP__
