/**
 * @file base_matrix_vector.hpp
 * @brief Provides a templated fixed-size vector class and related operations
 * for mathematical computations.
 *
 * This file defines the Base::Matrix::Vector class template and a set of
 * associated functions and operator overloads for performing mathematical
 * operations on vectors, including addition, subtraction, multiplication,
 * normalization, dot product, and norm calculations. The implementation
 * supports both real and complex vectors, and can be configured to use either
 * std::vector or std::array as the underlying storage, depending on the
 * compilation flags.
 *
 * The code also includes utility functions for extracting real and imaginary
 * parts from complex vectors, as well as specialized operations for column
 * vectors and normalization routines.
 *
 */
#ifndef __BASE_MATRIX_VECTOR_HPP__
#define __BASE_MATRIX_VECTOR_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_complex.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

#include <cmath>

namespace Base {
namespace Matrix {

/* Vector */

/**
 * @brief Templated fixed-size vector class for mathematical computations.
 *
 * This class provides a fixed-size vector implementation with various
 * mathematical operations, including dot product, normalization, and scalar
 * addition/subtraction. It can be configured to use either std::vector or
 * std::array as the underlying storage.
 *
 * @tparam T The type of the vector elements (e.g., float, double).
 * @tparam N The size of the vector.
 */
template <typename T, std::size_t N> class Vector {
public:
/* Constructor */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  Vector() : data(N, static_cast<T>(0)) {}

  Vector(const std::initializer_list<T> &input) : data(input) {}

  Vector(const std::vector<T> &input) : data(input) {}

#else // __BASE_MATRIX_USE_STD_VECTOR__

  Vector() : data{} {}

  Vector(const std::initializer_list<T> &input) : data{} {

    // This may cause runtime error if the size of values is larger than N.
    std::copy(input.begin(), input.end(), this->data.begin());
  }

  Vector(const std::array<T, N> &input) : data(input) {}

  Vector(const std::vector<T> &input) : data{} {

    // This may cause runtime error if the size of values is larger than N.
    std::copy(input.begin(), input.end(), this->data.begin());
  }

#endif // __BASE_MATRIX_USE_STD_VECTOR__

  /* Copy Constructor */
  Vector(const Vector<T, N> &vector) : data(vector.data) {}

  Vector<T, N> &operator=(const Vector<T, N> &vector) {
    if (this != &vector) {
      this->data = vector.data;
    }
    return *this;
  }

  /* Move Constructor */
  Vector(Vector<T, N> &&vector) noexcept : data(std::move(vector.data)) {}

  Vector<T, N> &operator=(Vector<T, N> &&vector) noexcept {
    if (this != &vector) {
      this->data = std::move(vector.data);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Accessor for vector elements.
   *
   * This function provides access to the vector elements using the subscript
   * operator. If the index is out of bounds, it returns the last element.
   *
   * @param index The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  T &operator[](std::size_t index) {
    if (index >= N) {
      index = N - 1;
    }
    return this->data[index];
  }

  /**
   * @brief Const accessor for vector elements.
   *
   * This function provides const access to the vector elements using the
   * subscript operator. If the index is out of bounds, it returns the last
   * element.
   *
   * @param index The index of the element to access.
   * @return A const reference to the element at the specified index.
   */
  const T &operator[](std::size_t index) const {
    if (index >= N) {
      index = N - 1;
    }
    return this->data[index];
  }

  /**
   * @brief Accessor for vector elements using the subscript operator.
   *
   * This function provides access to the vector elements using the subscript
   * operator. If the index is out of bounds, it returns the last element.
   *
   * @param index The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  constexpr std::size_t size() const { return N; }

  /**
   * @brief Accessor for the first element of the vector.
   *
   * This function returns a reference to the first element of the vector.
   *
   * @return A reference to the first element.
   */
  static inline Vector<T, N> Ones(void) {
    return Vector<T, N>(std::vector<T>(N, static_cast<T>(1)));
  }

  /* dot */

  /**
   * @brief Computes the dot product of two vectors.
   *
   * This function computes the dot product of two vectors using a recursive
   * template structure for efficiency.
   *
   * @tparam U The type of the vector elements (should match T).
   * @tparam P The size of the vectors (should match N).
   * @param a The first vector.
   * @param b The second vector.
   * @return The dot product of the two vectors.
   */
  template <typename U, std::size_t P, std::size_t P_idx> struct VectorDotCore {
    static T compute(const Vector<U, P> &a, const Vector<U, P> &b) {
      return a[P_idx] * b[P_idx] +
             VectorDotCore<U, P, P_idx - 1>::compute(a, b);
    }
  };

  /**
   * @brief Specialization for the base case of the recursive dot product
   * computation.
   *
   * This specialization handles the case when the index reaches 0, returning
   * the product of the first elements of both vectors.
   *
   * @tparam U The type of the vector elements (should match T).
   * @tparam P The size of the vectors (should match N).
   */
  template <typename U, std::size_t P> struct VectorDotCore<U, P, 0> {
    static T compute(const Vector<U, P> &a, const Vector<U, P> &b) {
      return a[0] * b[0];
    }
  };

  /**
   * @brief Computes the dot product of two vectors.
   *
   * This function provides a convenient interface for computing the dot
   * product of two vectors using the VectorDotCore structure.
   *
   * @tparam U The type of the vector elements (should match T).
   * @tparam P The size of the vectors (should match N).
   * @param a The first vector.
   * @param b The second vector.
   * @return The dot product of the two vectors.
   */
  template <typename U, std::size_t P>
  static inline T VECTOR_DOT(const Vector<U, P> &a, const Vector<U, P> &b) {
    return VectorDotCore<U, P, P - 1>::compute(a, b);
  }

  /**
   * @brief Computes the dot product of this vector with another vector.
   *
   * This function computes the dot product of this vector with another vector
   * using either a for loop or a recursive template structure, depending on the
   * compilation flags.
   *
   * @param other The other vector to compute the dot product with.
   * @return The dot product of the two vectors.
   */
  inline T dot(const Vector<T, N> &other) const {
    T result = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    for (std::size_t i = 0; i < N; ++i) {
      result += this->data[i] * other[i];
    }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    result = VECTOR_DOT<T, N>(*this, other);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return result;
  }

  /**
   * @brief Computes the dot product of this vector with another vector.
   *
   * This function computes the dot product of this vector with another vector
   * using either a for loop or a recursive template structure, depending on the
   * compilation flags.
   *
   * @param other The other vector to compute the dot product with.
   * @return The dot product of the two vectors.
   */
  inline T norm() const {

    T sum = this->dot(*this);

    return Base::Math::sqrt<T>(sum);
  }

  /**
   * @brief Computes the dot product of this vector with another vector.
   *
   * This function computes the dot product of this vector with another vector
   * using either a for loop or a recursive template structure, depending on the
   * compilation flags.
   *
   * @param other The other vector to compute the dot product with.
   * @param division_min The minimum value for division to avoid division by
   * zero.
   * @return The dot product of the two vectors.
   */
  inline T norm(const T &division_min) const {

    T sum = this->dot(*this);

    return Base::Math::sqrt<T>(sum, division_min);
  }

  /**
   * @brief Computes the inverse of the norm of this vector.
   *
   * This function computes the inverse of the norm of this vector, which is
   * useful for normalization.
   *
   * @return The inverse of the norm of this vector.
   */
  inline T norm_inv() const {

    T sum = this->dot(*this);

    return Base::Math::rsqrt<T>(sum);
  }

  /**
   * @brief Computes the inverse of the norm of this vector with a minimum
   * division value.
   *
   * This function computes the inverse of the norm of this vector, which is
   * useful for normalization, while ensuring that division by zero is avoided
   * by using a minimum division value.
   *
   * @param division_min The minimum value for division to avoid division by
   * zero.
   * @return The inverse of the norm of this vector.
   */
  inline T norm_inv(const T &division_min) const {

    T sum = this->dot(*this);

    return Base::Math::rsqrt<T>(sum, division_min);
  }

/* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> data;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, N> data;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/**
 * @brief Specialization of the Vector class for complex numbers.
 *
 * This specialization provides additional functionality for complex vectors,
 * including methods to extract real and imaginary parts.
 *
 * @tparam T The type of the complex vector elements (e.g.,
 * std::complex<float>).
 * @tparam N The size of the vector.
 */
template <typename T, std::size_t N> class ColVector : public Vector<T, N> {
public:
  /* Constructor */
  ColVector() : Vector<T, N>() {}

  ColVector(const Vector<T, N> &vec) : Vector<T, N>(vec) {}

  /* Copy Constructor */
  ColVector(const ColVector<T, N> &vec) : Vector<T, N>(vec) {}
  ColVector<T, N> &operator=(const ColVector<T, N> &vec) {
    if (this != &vec) {
      this->data = vec.data;
    }
    return *this;
  }

  /* Move Constructor */
  ColVector(ColVector<T, N> &&vec) noexcept : Vector<T, N>(std::move(vec)) {}
  ColVector<T, N> &operator=(ColVector<T, N> &&vec) noexcept {
    if (this != &vec) {
      this->data = std::move(vec.data);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Extracts the real part of the complex vector.
   *
   * This function returns a new vector containing the real parts of the
   * complex numbers in this vector.
   *
   * @return A vector containing the real parts of the complex numbers.
   */
  inline Vector<T, N> transpose() const {
    Vector<T, N> result;

    Base::Utility::copy<T, 0, N, 0, N, N>(this->data, result.data);

    return result;
  }
};

/* Normalize */
namespace VectorNormalize {

template <typename T, std::size_t N, std::size_t Index>
struct VectorNormalizeCore {
  /**
   * @brief Normalizes the vector by multiplying each element with the inverse
   * of the norm.
   *
   * This function recursively normalizes the vector by multiplying each
   * element with the inverse of the norm, starting from the last index and
   * moving towards the first.
   *
   * @param vec The vector to normalize.
   * @param norm_inv The inverse of the norm to multiply each element with.
   */
  static void compute(Vector<T, N> &vec, T norm_inv) {
    vec[Index] *= norm_inv;
    VectorNormalizeCore<T, N, Index - 1>::compute(vec, norm_inv);
  }
};

// Specialization to end the recursion
template <typename T, std::size_t N> struct VectorNormalizeCore<T, N, 0> {
  /**
   * @brief Normalizes the first element of the vector by multiplying it with
   * the inverse of the norm.
   *
   * This function is called when the recursion reaches the first index, and it
   * multiplies the first element of the vector with the inverse of the norm.
   *
   * @param vec The vector to normalize.
   * @param norm_inv The inverse of the norm to multiply the first element with.
   */
  static void compute(Vector<T, N> &vec, T norm_inv) { vec[0] *= norm_inv; }
};

/**
 * @brief Computes the normalization of a vector by multiplying each element
 * with the inverse of the norm.
 *
 * This function provides a convenient interface for normalizing a vector by
 * calling the VectorNormalizeCore structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector to normalize.
 * @param norm_inv The inverse of the norm to multiply each element with.
 */
template <typename T, std::size_t N>
inline void compute(Vector<T, N> &vec, T norm_inv) {
  VectorNormalizeCore<T, N, N - 1>::compute(vec, norm_inv);
}

} // namespace VectorNormalize

/**
 * @brief Normalizes a vector by multiplying each element with the inverse of
 * the norm.
 *
 * This function normalizes the vector by multiplying each element with the
 * inverse of the norm, which is computed using the dot product of the vector
 * with itself.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector to normalize.
 */
template <typename T, std::size_t N>
inline void vector_normalize(Vector<T, N> &vec) {
  T norm_inv = vec.norm_inv();

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorNormalize::compute<T, N>(vec, norm_inv);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/**
 * @brief Normalizes a vector by multiplying each element with the inverse of
 * the norm, using a minimum division value to avoid division by zero.
 *
 * This function normalizes the vector by multiplying each element with the
 * inverse of the norm, which is computed using the dot product of the vector
 * with itself, while ensuring that division by zero is avoided by using a
 * minimum division value.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector to normalize.
 * @param division_min The minimum value for division to avoid division by
 * zero.
 */
template <typename T, std::size_t N>
inline void vector_normalize(Vector<T, N> &vec, const T &division_min) {
  T norm_inv = vec.norm_inv(division_min);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorNormalize::compute<T, N>(vec, norm_inv);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/* Scalar Addition */
namespace VectorAddScalar {

// Vector Add Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Adds a scalar to each element of the vector.
   *
   * This function recursively adds a scalar to each element of the vector,
   * starting from the last index and moving towards the first.
   *
   * @param vec The vector to which the scalar is added.
   * @param scalar The scalar value to add to each element of the vector.
   * @param result The resulting vector after adding the scalar.
   */
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] + scalar;
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Adds a scalar to the first element of the vector.
   *
   * This function is called when the recursion reaches the first index, and it
   * adds the scalar to the first element of the vector.
   *
   * @param vec The vector to which the scalar is added.
   * @param scalar The scalar value to add to the first element of the vector.
   * @param result The resulting vector after adding the scalar.
   */
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] + scalar;
  }
};

/**
 * @brief Computes the addition of a scalar to each element of the vector.
 *
 * This function provides a convenient interface for adding a scalar to each
 * element of the vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector to which the scalar is added.
 * @param scalar The scalar value to add to each element of the vector.
 * @param result The resulting vector after adding the scalar.
 */
template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(vec, scalar, result);
}

} // namespace VectorAddScalar

/**
 * @brief Adds a scalar to each element of the vector.
 *
 * This function adds a scalar to each element of the vector, using either a
 * for loop or a recursive template structure, depending on the compilation
 * flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector to which the scalar is added.
 * @param scalar The scalar value to add to each element of the vector.
 * @return A new vector with the scalar added to each element.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator+(const Vector<T, N> &vec, const T &scalar) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = vec.data[i] + scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorAddScalar::compute<T, N>(vec, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/**
 * @brief Adds a scalar to each element of the vector.
 *
 * This function adds a scalar to each element of the vector, using either a
 * for loop or a recursive template structure, depending on the compilation
 * flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param scalar The scalar value to add to each element of the vector.
 * @param vec The vector to which the scalar is added.
 * @return A new vector with the scalar added to each element.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator+(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar + vec[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorAddScalar::compute<T, N>(vec, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Scalar Subtraction */
namespace VectorSubScalar {

// Vector Sub Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Subtracts a scalar from each element of the vector.
   *
   * This function recursively subtracts a scalar from each element of the
   * vector, starting from the last index and moving towards the first.
   *
   * @param vec The vector from which the scalar is subtracted.
   * @param scalar The scalar value to subtract from each element of the vector.
   * @param result The resulting vector after subtracting the scalar.
   */
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] - scalar;
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Subtracts a scalar from the first element of the vector.
   *
   * This function is called when the recursion reaches the first index, and it
   * subtracts the scalar from the first element of the vector.
   *
   * @param vec The vector from which the scalar is subtracted.
   * @param scalar The scalar value to subtract from the first element of the
   * vector.
   * @param result The resulting vector after subtracting the scalar.
   */
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] - scalar;
  }
};

/**
 * @brief Computes the subtraction of a scalar from each element of the vector.
 *
 * This function provides a convenient interface for subtracting a scalar from
 * each element of the vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector from which the scalar is subtracted.
 * @param scalar The scalar value to subtract from each element of the vector.
 * @param result The resulting vector after subtracting the scalar.
 */
template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &vec, const T &scalar,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(vec, scalar, result);
}

} // namespace VectorSubScalar

/**
 * @brief Subtracts a scalar from each element of the vector.
 *
 * This function subtracts a scalar from each element of the vector, using
 * either a for loop or a recursive template structure, depending on the
 * compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector from which the scalar is subtracted.
 * @param scalar The scalar value to subtract from each element of the vector.
 * @return A new vector with the scalar subtracted from each element.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator-(const Vector<T, N> &vec, const T &scalar) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = vec[i] - scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorSubScalar::compute<T, N>(vec, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace ScalarSubVector {

// Scalar Sub Vector Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Subtracts a vector from a scalar.
   *
   * This function recursively subtracts each element of the vector from the
   * scalar, starting from the last index and moving towards the first.
   *
   * @param scalar The scalar value from which the vector is subtracted.
   * @param vec The vector to subtract from the scalar.
   * @param result The resulting vector after subtracting the vector from the
   * scalar.
   */
  static void compute(T scalar, const Vector<T, N> &vec, Vector<T, N> &result) {
    result[N_idx] = scalar - vec[N_idx];
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Subtracts the first element of the vector from the scalar.
   *
   * This function is called when the recursion reaches the first index, and it
   * subtracts the first element of the vector from the scalar.
   *
   * @param scalar The scalar value from which the first element of the vector
   * is subtracted.
   * @param vec The vector whose first element is subtracted from the scalar.
   * @param result The resulting vector after subtracting the first element of
   * the vector from the scalar.
   */
  static void compute(T scalar, const Vector<T, N> &vec, Vector<T, N> &result) {
    result[0] = scalar - vec[0];
  }
};

/**
 * @brief Computes the subtraction of a vector from a scalar.
 *
 * This function provides a convenient interface for subtracting a vector from
 * a scalar using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param scalar The scalar value from which the vector is subtracted.
 * @param vec The vector to subtract from the scalar.
 * @param result The resulting vector after subtracting the vector from the
 * scalar.
 */
template <typename T, std::size_t N>
inline void compute(const T &scalar, const Vector<T, N> &vec,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(scalar, vec, result);
}

} // namespace ScalarSubVector

/**
 * @brief Subtracts a vector from a scalar.
 *
 * This function subtracts each element of the vector from the scalar, using
 * either a for loop or a recursive template structure, depending on the
 * compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param scalar The scalar value from which the vector is subtracted.
 * @param vec The vector to subtract from the scalar.
 * @return A new vector with the scalar subtracted from each element of the
 * vector.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator-(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar - vec[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  ScalarSubVector::compute<T, N>(scalar, vec, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Vector Addition */
namespace VectorAddVector {

// Vector Add Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Adds two vectors element-wise.
   *
   * This function recursively adds the corresponding elements of two vectors,
   * starting from the last index and moving towards the first.
   *
   * @param a The first vector to add.
   * @param b The second vector to add.
   * @param result The resulting vector after adding the two vectors.
   */
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] + b[N_idx];
    Core<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Adds the first elements of two vectors.
   *
   * This function is called when the recursion reaches the first index, and it
   * adds the first elements of the two vectors.
   *
   * @param a The first vector to add.
   * @param b The second vector to add.
   * @param result The resulting vector after adding the first elements of the
   * two vectors.
   */
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[0] = a[0] + b[0];
  }
};

/**
 * @brief Computes the addition of two vectors element-wise.
 *
 * This function provides a convenient interface for adding two vectors using
 * the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vectors.
 * @param a The first vector to add.
 * @param b The second vector to add.
 * @param result The resulting vector after adding the two vectors.
 */
template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(a, b, result);
}

} // namespace VectorAddVector

/**
 * @brief Adds two vectors element-wise.
 *
 * This function adds two vectors element-wise, using either a for loop or a
 * recursive template structure, depending on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vectors.
 * @param a The first vector to add.
 * @param b The second vector to add.
 * @return A new vector with the result of the addition.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator+(const Vector<T, N> &a, const Vector<T, N> &b) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = a[i] + b[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorAddVector::compute<T, N>(a, b, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Vector Subtraction */
namespace VectorSubVector {

// Vector Sub Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Subtracts two vectors element-wise.
   *
   * This function recursively subtracts the corresponding elements of two
   * vectors, starting from the last index and moving towards the first.
   *
   * @param a The first vector from which the second vector is subtracted.
   * @param b The second vector to subtract from the first vector.
   * @param result The resulting vector after subtracting the two vectors.
   */
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] - b[N_idx];
    Core<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Subtracts the first elements of two vectors.
   *
   * This function is called when the recursion reaches the first index, and it
   * subtracts the first element of the second vector from the first element of
   * the first vector.
   *
   * @param a The first vector from which the second vector is subtracted.
   * @param b The second vector to subtract from the first vector.
   * @param result The resulting vector after subtracting the two vectors.
   */
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[0] = a[0] - b[0];
  }
};

/**
 * @brief Computes the subtraction of two vectors element-wise.
 *
 * This function provides a convenient interface for subtracting two vectors
 * using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vectors.
 * @param a The first vector from which the second vector is subtracted.
 * @param b The second vector to subtract from the first vector.
 * @param result The resulting vector after subtracting the two vectors.
 */
template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(a, b, result);
}

} // namespace VectorSubVector

/**
 * @brief Subtracts two vectors element-wise.
 *
 * This function subtracts two vectors element-wise, using either a for loop or
 * a recursive template structure, depending on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vectors.
 * @param a The first vector from which the second vector is subtracted.
 * @param b The second vector to subtract from the first vector.
 * @return A new vector with the result of the subtraction.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator-(const Vector<T, N> &a, const Vector<T, N> &b) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = a[i] - b[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorSubVector::compute<T, N>(a, b, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Scalar Multiplication */
namespace VectorMultiplyScalar {

// Vector Multiply Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Multiplies each element of the vector by a scalar.
   *
   * This function recursively multiplies each element of the vector by a
   * scalar, starting from the last index and moving towards the first.
   *
   * @param vec The vector to multiply by the scalar.
   * @param scalar The scalar value to multiply each element of the vector with.
   * @param result The resulting vector after multiplying by the scalar.
   */
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] * scalar;
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Multiplies the first element of the vector by a scalar.
   *
   * This function is called when the recursion reaches the first index, and it
   * multiplies the first element of the vector by the scalar.
   *
   * @param vec The vector to multiply by the scalar.
   * @param scalar The scalar value to multiply the first element of the vector
   * with.
   * @param result The resulting vector after multiplying by the scalar.
   */
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] * scalar;
  }
};

/**
 * @brief Computes the multiplication of a vector by a scalar.
 *
 * This function provides a convenient interface for multiplying a vector by a
 * scalar using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector to multiply by the scalar.
 * @param scalar The scalar value to multiply each element of the vector with.
 * @param result The resulting vector after multiplying by the scalar.
 */
template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(vec, scalar, result);
}

} // namespace VectorMultiplyScalar

/**
 * @brief Multiplies a vector by a scalar.
 *
 * This function multiplies each element of the vector by a scalar, using
 * either a for loop or a recursive template structure, depending on the
 * compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The vector to multiply by the scalar.
 * @param scalar The scalar value to multiply each element of the vector with.
 * @return A new vector with each element multiplied by the scalar.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator*(const Vector<T, N> &vec, const T &scalar) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = vec[i] * scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorMultiplyScalar::compute<T, N>(vec, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/**
 * @brief Multiplies a scalar by a vector.
 *
 * This function multiplies each element of the vector by a scalar, using
 * either a for loop or a recursive template structure, depending on the
 * compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param scalar The scalar value to multiply each element of the vector with.
 * @param vec The vector to multiply by the scalar.
 * @return A new vector with each element multiplied by the scalar.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator*(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar * vec[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorMultiplyScalar::compute<T, N>(vec, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Vector Multiply */
namespace VectorMultiply {

template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Multiplies two vectors element-wise.
   *
   * This function recursively multiplies the corresponding elements of two
   * vectors, starting from the last index and moving towards the first.
   *
   * @param a The first vector to multiply.
   * @param b The second vector to multiply.
   * @param result The resulting vector after multiplying the two vectors.
   */
  static void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] * b[N_idx];
    Core<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Multiplies the first elements of two vectors.
   *
   * This function is called when the recursion reaches the first index, and it
   * multiplies the first elements of the two vectors.
   *
   * @param a The first vector to multiply.
   * @param b The second vector to multiply.
   * @param result The resulting vector after multiplying the first elements of
   * the two vectors.
   */
  static void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                      Vector<T, N> &result) {
    result[0] = a[0] * b[0];
  }
};

/**
 * @brief Computes the multiplication of two vectors element-wise.
 *
 * This function provides a convenient interface for multiplying two vectors
 * using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vectors.
 * @param a The first vector to multiply.
 * @param b The second vector to multiply.
 * @param result The resulting vector after multiplying the two vectors.
 */
template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(a, b, result);
}

} // namespace VectorMultiply

/**
 * @brief Multiplies two vectors element-wise.
 *
 * This function multiplies two vectors element-wise, using either a for loop or
 * a recursive template structure, depending on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vectors.
 * @param a The first vector to multiply.
 * @param b The second vector to multiply.
 * @return A new vector with the result of the multiplication.
 */
template <typename T, std::size_t N>
inline Vector<T, N> operator*(const Vector<T, N> &a, const Vector<T, N> &b) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = a[i] * b[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorMultiply::compute<T, N>(a, b, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Complex Norm */
namespace ComplexVectorNorm {

// Vector Multiply Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Computes the squared norm of a complex vector.
   *
   * This function recursively computes the squared norm of a complex vector,
   * starting from the last index and moving towards the first.
   *
   * @param vec_comp The complex vector whose norm is computed.
   * @return The squared norm of the complex vector.
   */
  static T compute(const Vector<Complex<T>, N> &vec_comp) {
    return vec_comp[N_idx].real * vec_comp[N_idx].real +
           vec_comp[N_idx].imag * vec_comp[N_idx].imag +
           Core<T, N, N_idx - 1>::compute(vec_comp);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Computes the squared norm of the first element of a complex vector.
   *
   * This function is called when the recursion reaches the first index, and it
   * computes the squared norm of the first element of the complex vector.
   *
   * @param vec_comp The complex vector whose norm is computed.
   * @return The squared norm of the first element of the complex vector.
   */
  static T compute(const Vector<Complex<T>, N> &vec_comp) {
    return vec_comp[0].real * vec_comp[0].real +
           vec_comp[0].imag * vec_comp[0].imag;
  }
};

/**
 * @brief Computes the squared norm of a complex vector.
 *
 * This function provides a convenient interface for computing the squared norm
 * of a complex vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec_comp The complex vector whose norm is computed.
 * @return The squared norm of the complex vector.
 */
template <typename T, std::size_t N>
inline T compute(const Vector<Complex<T>, N> &vec_comp) {
  return Core<T, N, N - 1>::compute(vec_comp);
}

} // namespace ComplexVectorNorm

/**
 * @brief Computes the norm of a complex vector.
 *
 * This function computes the norm of a complex vector, using either a for loop
 * or a recursive template structure, depending on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec_comp The complex vector whose norm is computed.
 * @return The norm of the complex vector.
 */
template <typename T, std::size_t N>
inline T complex_vector_norm(const Vector<Complex<T>, N> &vec_comp) {
  T sum = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  sum = ComplexVectorNorm::compute<T, N>(vec_comp);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Base::Math::sqrt<T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(
      sum);
}

/**
 * @brief Computes the norm of a complex vector with a minimum division value.
 *
 * This function computes the norm of a complex vector, using either a for loop
 * or a recursive template structure, depending on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec_comp The complex vector whose norm is computed.
 * @param division_min The minimum value for division to avoid division by zero.
 * @return The norm of the complex vector.
 */
template <typename T, std::size_t N>
inline T complex_vector_norm(const Vector<Complex<T>, N> &vec_comp,
                             const T &division_min) {
  T sum = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  sum = ComplexVectorNorm::compute<T, N>(vec_comp);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Base::Math::sqrt<T>(sum, division_min);
}

/**
 * @brief Computes the inverse norm of a complex vector.
 *
 * This function computes the inverse norm of a complex vector, using either a
 * for loop or a recursive template structure, depending on the compilation
 * flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec_comp The complex vector whose inverse norm is computed.
 * @return The inverse norm of the complex vector.
 */
template <typename T, std::size_t N>
inline T complex_vector_norm_inv(const Vector<Complex<T>, N> &vec_comp) {
  T sum = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  sum = ComplexVectorNorm::compute<T, N>(vec_comp);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Base::Math::rsqrt<T>(sum);
}

/**
 * @brief Computes the inverse norm of a complex vector with a minimum division
 * value.
 *
 * This function computes the inverse norm of a complex vector, using either a
 * for loop or a recursive template structure, depending on the compilation
 * flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec_comp The complex vector whose inverse norm is computed.
 * @param division_min The minimum value for division to avoid division by zero.
 * @return The inverse norm of the complex vector.
 */
template <typename T, std::size_t N>
inline T complex_vector_norm_inv(const Vector<Complex<T>, N> &vec_comp,
                                 const T &division_min) {
  T sum = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  sum = ComplexVectorNorm::compute<T, N>(vec_comp);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Base::Math::rsqrt<T>(sum, division_min);
}

/* Complex Normalize */
namespace ComplexVectorNormalize {

template <typename T, std::size_t N, std::size_t Index> struct Core {
  /**
   * @brief Normalizes a complex vector by multiplying each element by the
   * inverse norm.
   *
   * This function recursively normalizes each element of the complex vector,
   * starting from the last index and moving towards the first.
   *
   * @param vec The complex vector to normalize.
   * @param norm_inv The inverse norm value to multiply each element by.
   */
  static void compute(Vector<Complex<T>, N> &vec, T norm_inv) {
    vec[Index] *= norm_inv;
    Core<T, N, Index - 1>::compute(vec, norm_inv);
  }
};

// Specialization to end the recursion
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Normalizes the first element of a complex vector by multiplying it
   * by the inverse norm.
   *
   * This function is called when the recursion reaches the first index, and it
   * multiplies the first element of the complex vector by the inverse norm.
   *
   * @param vec The complex vector to normalize.
   * @param norm_inv The inverse norm value to multiply the first element by.
   */
  static void compute(Vector<Complex<T>, N> &vec, T norm_inv) {
    vec[0] *= norm_inv;
  }
};

/**
 * @brief Computes the normalization of a complex vector.
 *
 * This function provides a convenient interface for normalizing a complex
 * vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The complex vector to normalize.
 * @param norm_inv The inverse norm value to multiply each element by.
 */
template <typename T, std::size_t N>
inline void compute(Vector<Complex<T>, N> &vec, T norm_inv) {
  Core<T, N, N - 1>::compute(vec, norm_inv);
}

} // namespace ComplexVectorNormalize

/**
 * @brief Normalizes a complex vector by multiplying each element by the inverse
 * norm.
 *
 * This function normalizes a complex vector, using either a for loop or a
 * recursive template structure, depending on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The complex vector to normalize.
 */
template <typename T, std::size_t N>
inline void complex_vector_normalize(Vector<Complex<T>, N> &vec) {
  T norm_inv = complex_vector_norm_inv(vec);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  ComplexVectorNormalize::compute<T, N>(vec, norm_inv);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/**
 * @brief Normalizes a complex vector by multiplying each element by the inverse
 * norm with a minimum division value.
 *
 * This function normalizes a complex vector, using either a for loop or a
 * recursive template structure, depending on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param vec The complex vector to normalize.
 * @param division_min The minimum value for division to avoid division by zero.
 */
template <typename T, std::size_t N>
inline void complex_vector_normalize(Vector<Complex<T>, N> &vec,
                                     const T &division_min) {
  T norm_inv = complex_vector_norm_inv(vec, division_min);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  ComplexVectorNormalize::compute<T, N>(vec, norm_inv);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/* Get Real and Imaginary Vector from Complex Vector */
namespace GetRealFromComplexVector {

// Get Real from Complex Vector Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Extracts the real part from a complex vector and stores it in a
   * vector.
   *
   * This function recursively extracts the real part of each element of a
   * complex vector, starting from the last index and moving towards the first.
   *
   * @param To_vector The vector to store the real parts.
   * @param From_vector The complex vector from which the real parts are
   * extracted.
   */
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  /**
   * @brief Extracts the real part from a complex vector and stores it in an
   * array.
   *
   * This function recursively extracts the real part of each element of a
   * complex vector, starting from the last index and moving towards the first.
   *
   * @param To_vector The array to store the real parts.
   * @param From_vector The complex vector from which the real parts are
   * extracted.
   */
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  /**
   * @brief Extracts the real part from a complex vector and stores it in a
   * Vector.
   *
   * This function recursively extracts the real part of each element of a
   * complex vector, starting from the last index and moving towards the first.
   *
   * @param To_vector The Vector to store the real parts.
   * @param From_vector The complex vector from which the real parts are
   * extracted.
   */
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Extracts the real part from the first element of a complex vector
   * and stores it in a vector.
   *
   * This function is called when the recursion reaches the first index, and it
   * extracts the real part of the first element of the complex vector.
   *
   * @param To_vector The vector to store the real part.
   * @param From_vector The complex vector from which the real part is
   * extracted.
   */
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[0] = From_vector[0].real;
  }

  /**
   * @brief Extracts the real part from the first element of a complex vector
   * and stores it in an array.
   *
   * This function is called when the recursion reaches the first index, and it
   * extracts the real part of the first element of the complex vector.
   *
   * @param To_vector The array to store the real part.
   * @param From_vector The complex vector from which the real part is
   * extracted.
   */
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].real;
  }

  /**
   * @brief Extracts the real part from the first element of a complex vector
   * and stores it in a Vector.
   *
   * This function is called when the recursion reaches the first index, and it
   * extracts the real part of the first element of the complex vector.
   *
   * @param To_vector The Vector to store the real part.
   * @param From_vector The complex vector from which the real part is
   * extracted.
   */
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].real;
  }
};

/**
 * @brief Computes the real part of a complex vector and stores it in a vector,
 * array, or Vector.
 *
 * This function provides a convenient interface for extracting the real part
 * of a complex vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param To_vector The vector, array, or Vector to store the real parts.
 * @param From_vector The complex vector from which the real parts are
 * extracted.
 */
template <typename T, std::size_t N>
inline void compute(std::vector<T> &To_vector,
                    const std::vector<Complex<T>> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

/**
 * @brief Computes the real part of a complex vector and stores it in an array
 * or Vector.
 *
 * This function provides a convenient interface for extracting the real part
 * of a complex vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param To_vector The array or Vector to store the real parts.
 * @param From_vector The complex vector from which the real parts are
 * extracted.
 */
template <typename T, std::size_t N>
inline void compute(std::array<T, N> &To_vector,
                    const std::array<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

/**
 * @brief Computes the real part of a complex vector and stores it in a Vector.
 *
 * This function provides a convenient interface for extracting the real part
 * of a complex vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param To_vector The Vector to store the real parts.
 * @param From_vector The complex vector from which the real parts are
 * extracted.
 */
template <typename T, std::size_t N>
inline void compute(Vector<T, N> &To_vector,
                    const Vector<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

} // namespace GetRealFromComplexVector

namespace GetImagFromComplexVector {

// Get Imag from Complex Vector Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  /**
   * @brief Extracts the imaginary part from a complex vector and stores it in a
   * vector.
   *
   * This function recursively extracts the imaginary part of each element of a
   * complex vector, starting from the last index and moving towards the first.
   *
   * @param To_vector The vector to store the imaginary parts.
   * @param From_vector The complex vector from which the imaginary parts are
   * extracted.
   */
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  /**
   * @brief Extracts the imaginary part from a complex vector and stores it in
   * an array.
   *
   * This function recursively extracts the imaginary part of each element of a
   * complex vector, starting from the last index and moving towards the first.
   *
   * @param To_vector The array to store the imaginary parts.
   * @param From_vector The complex vector from which the imaginary parts are
   * extracted.
   */
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  /**
   * @brief Extracts the imaginary part from a complex vector and stores it in a
   * Vector.
   *
   * This function recursively extracts the imaginary part of each element of a
   * complex vector, starting from the last index and moving towards the first.
   *
   * @param To_vector The Vector to store the imaginary parts.
   * @param From_vector The complex vector from which the imaginary parts are
   * extracted.
   */
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Extracts the imaginary part from the first element of a complex
   * vector and stores it in a vector.
   *
   * This function is called when the recursion reaches the first index, and it
   * extracts the imaginary part of the first element of the complex vector.
   *
   * @param To_vector The vector to store the imaginary part.
   * @param From_vector The complex vector from which the imaginary part is
   * extracted.
   */
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[0] = From_vector[0].imag;
  }

  /**
   * @brief Extracts the imaginary part from the first element of a complex
   * vector and stores it in an array.
   *
   * This function is called when the recursion reaches the first index, and it
   * extracts the imaginary part of the first element of the complex vector.
   *
   * @param To_vector The array to store the imaginary part.
   * @param From_vector The complex vector from which the imaginary part is
   * extracted.
   */
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].imag;
  }

  /**
   * @brief Extracts the imaginary part from the first element of a complex
   * vector and stores it in a Vector.
   *
   * This function is called when the recursion reaches the first index, and it
   * extracts the imaginary part of the first element of the complex vector.
   *
   * @param To_vector The Vector to store the imaginary part.
   * @param From_vector The complex vector from which the imaginary part is
   * extracted.
   */
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].imag;
  }
};

/**
 * @brief Computes the imaginary part of a complex vector and stores it in a
 * vector, array, or Vector.
 *
 * This function provides a convenient interface for extracting the imaginary
 * part of a complex vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param To_vector The vector, array, or Vector to store the imaginary parts.
 * @param From_vector The complex vector from which the imaginary parts are
 * extracted.
 */
template <typename T, std::size_t N>
inline void compute(std::vector<T> &To_vector,
                    const std::vector<Complex<T>> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

/**
 * @brief Computes the imaginary part of a complex vector and stores it in an
 * array or Vector.
 *
 * This function provides a convenient interface for extracting the imaginary
 * part of a complex vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param To_vector The array or Vector to store the imaginary parts.
 * @param From_vector The complex vector from which the imaginary parts are
 * extracted.
 */
template <typename T, std::size_t N>
inline void compute(std::array<T, N> &To_vector,
                    const std::array<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

/**
 * @brief Computes the imaginary part of a complex vector and stores it in a
 * Vector.
 *
 * This function provides a convenient interface for extracting the imaginary
 * part of a complex vector using the Core structure.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param To_vector The Vector to store the imaginary parts.
 * @param From_vector The complex vector from which the imaginary parts are
 * extracted.
 */
template <typename T, std::size_t N>
inline void compute(Vector<T, N> &To_vector,
                    const Vector<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

} // namespace GetImagFromComplexVector

#ifdef __BASE_MATRIX_USE_STD_VECTOR__

/**
 * @brief Extracts the real part from a complex vector and stores it in a
 * std::vector.
 *
 * This function extracts the real part of each element of a complex vector,
 * using either a for loop or a recursive template structure, depending on the
 * compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param From_vector The complex vector from which the real parts are
 * extracted.
 * @return A new std::vector containing the real parts of the complex vector.
 */
template <typename T, std::size_t N>
inline std::vector<T> get_real_vector_from_complex_vector(
    const std::vector<Complex<T>> &From_vector) {

  std::vector<T> To_vector(N);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetRealFromComplexVector::compute<T, N>(To_vector, From_vector);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_vector;
}

/**
 * @brief Extracts the imaginary part from a complex vector and stores it in a
 * std::vector.
 *
 * This function extracts the imaginary part of each element of a complex
 * vector, using either a for loop or a recursive template structure, depending
 * on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param From_vector The complex vector from which the imaginary parts are
 * extracted.
 * @return A new std::vector containing the imaginary parts of the complex
 * vector.
 */
template <typename T, std::size_t N>
inline std::vector<T> get_imag_vector_from_complex_vector(
    const std::vector<Complex<T>> &From_vector) {

  std::vector<T> To_vector(N);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetImagFromComplexVector::compute<T, N>(To_vector, From_vector);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_vector;
}

#else // __BASE_MATRIX_USE_STD_VECTOR__

/**
 * @brief Extracts the real part from a complex vector and stores it in an
 * std::array.
 *
 * This function extracts the real part of each element of a complex vector,
 * using either a for loop or a recursive template structure, depending on the
 * compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param From_vector The complex vector from which the real parts are
 * extracted.
 * @return A new std::array containing the real parts of the complex vector.
 */
template <typename T, std::size_t N>
inline std::array<T, N> get_real_vector_from_complex_vector(
    const std::array<Complex<T>, N> &From_vector) {

  std::array<T, N> To_vector;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetRealFromComplexVector::compute<T, N>(To_vector, From_vector);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_vector;
}

/**
 * @brief Extracts the imaginary part from a complex vector and stores it in an
 * std::array.
 *
 * This function extracts the imaginary part of each element of a complex
 * vector, using either a for loop or a recursive template structure, depending
 * on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param From_vector The complex vector from which the imaginary parts are
 * extracted.
 * @return A new std::array containing the imaginary parts of the complex
 * vector.
 */
template <typename T, std::size_t N>
inline std::array<T, N> get_imag_vector_from_complex_vector(
    const std::array<Complex<T>, N> &From_vector) {

  std::array<T, N> To_vector;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetImagFromComplexVector::compute<T, N>(To_vector, From_vector);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_vector;
}

#endif // __BASE_MATRIX_USE_STD_VECTOR__

/**
 * @brief Extracts the real part from a complex vector and stores it in a
 * Vector.
 *
 * This function extracts the real part of each element of a complex vector,
 * using either a for loop or a recursive template structure, depending on the
 * compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param From_vector The complex vector from which the real parts are
 * extracted.
 * @return A new Vector containing the real parts of the complex vector.
 */
template <typename T, std::size_t N>
inline Vector<T, N>
get_real_vector_from_complex_vector(const Vector<Complex<T>, N> &From_vector) {

  Vector<T, N> To_vector;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetRealFromComplexVector::compute<T, N>(To_vector, From_vector);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_vector;
}

/**
 * @brief Extracts the imaginary part from a complex vector and stores it in a
 * Vector.
 *
 * This function extracts the imaginary part of each element of a complex
 * vector, using either a for loop or a recursive template structure, depending
 * on the compilation flags.
 *
 * @tparam T The type of the vector elements.
 * @tparam N The size of the vector.
 * @param From_vector The complex vector from which the imaginary parts are
 * extracted.
 * @return A new Vector containing the imaginary parts of the complex vector.
 */
template <typename T, std::size_t N>
inline Vector<T, N>
get_imag_vector_from_complex_vector(const Vector<Complex<T>, N> &From_vector) {

  Vector<T, N> To_vector;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetImagFromComplexVector::compute<T, N>(To_vector, From_vector);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_vector;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_VECTOR_HPP__
