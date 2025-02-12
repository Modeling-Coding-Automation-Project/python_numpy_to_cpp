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
template <typename T, std::size_t N> class Vector {
public:
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

  /* Function */
  T &operator[](std::size_t index) {
    if (index >= N) {
      index = N - 1;
    }
    return this->data[index];
  }

  const T &operator[](std::size_t index) const {
    if (index >= N) {
      index = N - 1;
    }
    return this->data[index];
  }

  constexpr std::size_t size() const { return N; }

  static inline Vector<T, N> Ones(void) {
    return Vector<T, N>(std::vector<T>(N, static_cast<T>(1)));
  }

  /* dot */
  template <typename U, std::size_t P, std::size_t P_idx> struct VectorDotCore {
    static T compute(const Vector<U, P> &a, const Vector<U, P> &b) {
      return a[P_idx] * b[P_idx] +
             VectorDotCore<U, P, P_idx - 1>::compute(a, b);
    }
  };

  // Termination condition: P_idx == 0
  template <typename U, std::size_t P> struct VectorDotCore<U, P, 0> {
    static T compute(const Vector<U, P> &a, const Vector<U, P> &b) {
      return a[0] * b[0];
    }
  };

  template <typename U, std::size_t P>
  static inline T VECTOR_DOT(const Vector<U, P> &a, const Vector<U, P> &b) {
    return VectorDotCore<U, P, P - 1>::compute(a, b);
  }

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

  inline T norm() const {

    T sum = this->dot(*this);

    return Base::Math::sqrt<T>(sum);
  }

  inline T norm(const T &division_min) const {

    T sum = this->dot(*this);

    return Base::Math::sqrt<T>(sum, division_min);
  }

  inline T norm_inv() const {

    T sum = this->dot(*this);

    return Base::Math::rsqrt<T>(sum);
  }

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

template <typename T, std::size_t N> class ColVector : public Vector<T, N> {
public:
  ColVector() : Vector<T, N>() {}

  ColVector(const Vector<T, N> &vec) : Vector<T, N>(vec) {}

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
  static void compute(Vector<T, N> &vec, T norm_inv) {
    vec[Index] *= norm_inv;
    VectorNormalizeCore<T, N, Index - 1>::compute(vec, norm_inv);
  }
};

// Specialization to end the recursion
template <typename T, std::size_t N> struct VectorNormalizeCore<T, N, 0> {
  static void compute(Vector<T, N> &vec, T norm_inv) { vec[0] *= norm_inv; }
};

template <typename T, std::size_t N>
inline void compute(Vector<T, N> &vec, T norm_inv) {
  VectorNormalizeCore<T, N, N - 1>::compute(vec, norm_inv);
}

} // namespace VectorNormalize

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
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] + scalar;
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] + scalar;
  }
};

template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(vec, scalar, result);
}

} // namespace VectorAddScalar

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
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] - scalar;
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] - scalar;
  }
};

template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &vec, const T &scalar,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(vec, scalar, result);
}

} // namespace VectorSubScalar

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
  static void compute(T scalar, const Vector<T, N> &vec, Vector<T, N> &result) {
    result[N_idx] = scalar - vec[N_idx];
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(T scalar, const Vector<T, N> &vec, Vector<T, N> &result) {
    result[0] = scalar - vec[0];
  }
};

template <typename T, std::size_t N>
inline void compute(const T &scalar, const Vector<T, N> &vec,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(scalar, vec, result);
}

} // namespace ScalarSubVector

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
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] + b[N_idx];
    Core<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[0] = a[0] + b[0];
  }
};

template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(a, b, result);
}

} // namespace VectorAddVector

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
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] - b[N_idx];
    Core<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[0] = a[0] - b[0];
  }
};

template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(a, b, result);
}

} // namespace VectorSubVector

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
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] * scalar;
    Core<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] * scalar;
  }
};

template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(vec, scalar, result);
}

} // namespace VectorMultiplyScalar

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
  static void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] * b[N_idx];
    Core<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                      Vector<T, N> &result) {
    result[0] = a[0] * b[0];
  }
};

template <typename T, std::size_t N>
inline void compute(const Vector<T, N> &a, const Vector<T, N> &b,
                    Vector<T, N> &result) {
  Core<T, N, N - 1>::compute(a, b, result);
}

} // namespace VectorMultiply

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
  static T compute(const Vector<Complex<T>, N> &vec_comp) {
    return vec_comp[N_idx].real * vec_comp[N_idx].real +
           vec_comp[N_idx].imag * vec_comp[N_idx].imag +
           Core<T, N, N_idx - 1>::compute(vec_comp);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static T compute(const Vector<Complex<T>, N> &vec_comp) {
    return vec_comp[0].real * vec_comp[0].real +
           vec_comp[0].imag * vec_comp[0].imag;
  }
};

template <typename T, std::size_t N>
inline T compute(const Vector<Complex<T>, N> &vec_comp) {
  return Core<T, N, N - 1>::compute(vec_comp);
}

} // namespace ComplexVectorNorm

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
  static void compute(Vector<Complex<T>, N> &vec, T norm_inv) {
    vec[Index] *= norm_inv;
    Core<T, N, Index - 1>::compute(vec, norm_inv);
  }
};

// Specialization to end the recursion
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static void compute(Vector<Complex<T>, N> &vec, T norm_inv) {
    vec[0] *= norm_inv;
  }
};

template <typename T, std::size_t N>
inline void compute(Vector<Complex<T>, N> &vec, T norm_inv) {
  Core<T, N, N - 1>::compute(vec, norm_inv);
}

} // namespace ComplexVectorNormalize

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
  // For std::vector
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  // For std::array
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  // For Vector
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  // For std::vector
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[0] = From_vector[0].real;
  }

  // For std::array
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].real;
  }

  // For Vector
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].real;
  }
};

template <typename T, std::size_t N>
inline void compute(std::vector<T> &To_vector,
                    const std::vector<Complex<T>> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

template <typename T, std::size_t N>
inline void compute(std::array<T, N> &To_vector,
                    const std::array<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

template <typename T, std::size_t N>
inline void compute(Vector<T, N> &To_vector,
                    const Vector<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

} // namespace GetRealFromComplexVector

namespace GetImagFromComplexVector {

// Get Imag from Complex Vector Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx> struct Core {
  // For std::vector
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  // For std::array
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }

  // For Vector
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    Core<T, N, N_idx - 1>::compute(To_vector, From_vector);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  // For std::vector
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[0] = From_vector[0].imag;
  }

  // For std::array
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].imag;
  }

  // For Vector
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[0] = From_vector[0].imag;
  }
};

template <typename T, std::size_t N>
inline void compute(std::vector<T> &To_vector,
                    const std::vector<Complex<T>> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

template <typename T, std::size_t N>
inline void compute(std::array<T, N> &To_vector,
                    const std::array<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

template <typename T, std::size_t N>
inline void compute(Vector<T, N> &To_vector,
                    const Vector<Complex<T>, N> &From_vector) {
  Core<T, N, N - 1>::compute(To_vector, From_vector);
}

} // namespace GetImagFromComplexVector

#ifdef __BASE_MATRIX_USE_STD_VECTOR__

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
