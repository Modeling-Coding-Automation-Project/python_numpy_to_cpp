#ifndef BASE_MATRIX_VECTOR_HPP
#define BASE_MATRIX_VECTOR_HPP

#include "base_math.hpp"
#include "base_matrix_complex.hpp"
#include "base_matrix_macros.hpp"

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
#ifdef BASE_MATRIX_USE_STD_VECTOR

  Vector() : data(N, static_cast<T>(0)) {}

  Vector(const std::initializer_list<T> &input) : data(input) {}

  Vector(const std::vector<T> &input) : data(input) {}

#else

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

#endif

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

  inline T dot(const Vector<T, N> &other) const {
    T result = static_cast<T>(0);
    for (std::size_t i = 0; i < N; ++i) {
      result += this->data[i] * other[i];
    }
    return result;
  }

  inline T norm() const {
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < N; ++i) {
      sum += this->data[i] * this->data[i];
    }

    return Base::Math::sqrt_base_math<
        T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum);
  }

  inline T norm(const T &division_min) const {
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < N; ++i) {
      sum += this->data[i] * this->data[i];
    }

    return Base::Math::sqrt_base_math<
        T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum, division_min);
  }

  inline T norm_inv() const {
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < N; ++i) {
      sum += this->data[i] * this->data[i];
    }

    return Base::Math::rsqrt_base_math<
        T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum);
  }

  inline T norm_inv(const T &division_min) const {
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < N; ++i) {
      sum += this->data[i] * this->data[i];
    }

    return Base::Math::rsqrt_base_math<
        T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum, division_min);
  }

/* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> data;
#else
  std::array<T, N> data;
#endif
};

template <typename T, std::size_t N> class ColVector : public Vector<T, N> {
public:
  ColVector() : Vector<T, N>() {}

  ColVector(const Vector<T, N> &vec) : Vector<T, N>(vec) {}

  inline Vector<T, N> transpose() const {
    Vector<T, N> result;

    std::copy(this->data.begin(), this->data.end(), result.data.begin());

    return result;
  }
};

/* Normalize */
template <typename T, std::size_t N>
inline void vector_normalize(Vector<T, N> &vec) {
  T norm_inv = vec.norm_inv();
  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }
}

template <typename T, std::size_t N>
inline void vector_normalize(Vector<T, N> &vec, const T &division_min) {
  T norm_inv = vec.norm_inv(division_min);
  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }
}

/* Scalar Addition */
// Vector Add Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct VectorAddScalarCore {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] + scalar;
    VectorAddScalarCore<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct VectorAddScalarCore<T, N, 0> {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] + scalar;
  }
};

template <typename T, std::size_t N>
static inline void COMPILED_VECTOR_ADD_SCALAR(const Vector<T, N> &vec, T scalar,
                                              Vector<T, N> &result) {
  VectorAddScalarCore<T, N, N - 1>::compute(vec, scalar, result);
}

template <typename T, std::size_t N>
inline Vector<T, N> operator+(const Vector<T, N> &vec, const T &scalar) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = vec.data[i] + scalar;
  }

#else

  Base::Matrix::COMPILED_VECTOR_ADD_SCALAR<T, N>(vec, scalar, result);

#endif

  return result;
}

template <typename T, std::size_t N>
inline Vector<T, N> operator+(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar + vec[i];
  }

#else

  Base::Matrix::COMPILED_VECTOR_ADD_SCALAR<T, N>(vec, scalar, result);

#endif

  return result;
}

/* Scalar Subtraction */
// Vector Sub Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct VectorSubScalarCore {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] - scalar;
    VectorSubScalarCore<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct VectorSubScalarCore<T, N, 0> {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] - scalar;
  }
};

template <typename T, std::size_t N>
static inline void COMPILED_VECTOR_SUB_SCALAR(const Vector<T, N> &vec,
                                              const T &scalar,
                                              Vector<T, N> &result) {
  VectorSubScalarCore<T, N, N - 1>::compute(vec, scalar, result);
}

template <typename T, std::size_t N>
inline Vector<T, N> operator-(const Vector<T, N> &vec, const T &scalar) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = vec[i] - scalar;
  }

#else

  Base::Matrix::COMPILED_VECTOR_SUB_SCALAR<T, N>(vec, scalar, result);

#endif

  return result;
}

// Scalar Sub Vector Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct ScalarSubVectorCore {
  static void compute(T scalar, const Vector<T, N> &vec, Vector<T, N> &result) {
    result[N_idx] = scalar - vec[N_idx];
    ScalarSubVectorCore<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct ScalarSubVectorCore<T, N, 0> {
  static void compute(T scalar, const Vector<T, N> &vec, Vector<T, N> &result) {
    result[0] = scalar - vec[0];
  }
};

template <typename T, std::size_t N>
static inline void COMPILED_SCALAR_SUB_VECTOR(const T &scalar,
                                              const Vector<T, N> &vec,
                                              Vector<T, N> &result) {
  ScalarSubVectorCore<T, N, N - 1>::compute(scalar, vec, result);
}

template <typename T, std::size_t N>
inline Vector<T, N> operator-(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar - vec[i];
  }

#else

  Base::Matrix::COMPILED_SCALAR_SUB_VECTOR<T, N>(scalar, vec, result);

#endif

  return result;
}

/* Vector Addition */
// Vector Add Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct VectorAddVectorCore {
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] + b[N_idx];
    VectorAddVectorCore<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct VectorAddVectorCore<T, N, 0> {
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[0] = a[0] + b[0];
  }
};

template <typename T, std::size_t N>
static inline void COMPILED_VECTOR_ADD_VECTOR(const Vector<T, N> &a,
                                              const Vector<T, N> &b,
                                              Vector<T, N> &result) {
  VectorAddVectorCore<T, N, N - 1>::compute(a, b, result);
}

template <typename T, std::size_t N>
inline Vector<T, N> operator+(const Vector<T, N> &a, const Vector<T, N> &b) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = a[i] + b[i];
  }

#else

  Base::Matrix::COMPILED_VECTOR_ADD_VECTOR<T, N>(a, b, result);

#endif

  return result;
}

/* Vector Subtraction */
// Vector Sub Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct VectorSubVectorCore {
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[N_idx] = a[N_idx] - b[N_idx];
    VectorSubVectorCore<T, N, N_idx - 1>::compute(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct VectorSubVectorCore<T, N, 0> {
  static void compute(const Vector<T, N> &a, const Vector<T, N> b,
                      Vector<T, N> &result) {
    result[0] = a[0] - b[0];
  }
};

template <typename T, std::size_t N>
static inline void COMPILED_VECTOR_SUB_VECTOR(const Vector<T, N> &a,
                                              const Vector<T, N> &b,
                                              Vector<T, N> &result) {
  VectorSubVectorCore<T, N, N - 1>::compute(a, b, result);
}

template <typename T, std::size_t N>
inline Vector<T, N> operator-(const Vector<T, N> &a, const Vector<T, N> &b) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = a[i] - b[i];
  }

#else

  Base::Matrix::COMPILED_VECTOR_SUB_VECTOR<T, N>(a, b, result);

#endif

  return result;
}

/* Scalar Multiplication */
// Vector Multiply Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct VectorMultiplyScalarCore {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[N_idx] = vec[N_idx] * scalar;
    VectorMultiplyScalarCore<T, N, N_idx - 1>::compute(vec, scalar, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct VectorMultiplyScalarCore<T, N, 0> {
  static void compute(const Vector<T, N> &vec, T scalar, Vector<T, N> &result) {
    result[0] = vec[0] * scalar;
  }
};

template <typename T, std::size_t N>
static inline void COMPILED_VECTOR_MULTIPLY_SCALAR(const Vector<T, N> &vec,
                                                   T scalar,
                                                   Vector<T, N> &result) {
  VectorMultiplyScalarCore<T, N, N - 1>::compute(vec, scalar, result);
}

template <typename T, std::size_t N>
inline Vector<T, N> operator*(const Vector<T, N> &vec, const T &scalar) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = vec[i] * scalar;
  }

#else

  Base::Matrix::COMPILED_VECTOR_MULTIPLY_SCALAR<T, N>(vec, scalar, result);

#endif

  return result;
}

template <typename T, std::size_t N>
inline Vector<T, N> operator*(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar * vec[i];
  }

#else

  Base::Matrix::COMPILED_VECTOR_MULTIPLY_SCALAR<T, N>(vec, scalar, result);

#endif

  return result;
}

/* Vector Multiply */
template <typename T, std::size_t N, std::size_t N_idx>
struct VectorMultiplyCore {
  static void multiply(const Vector<T, N> &a, const Vector<T, N> &b,
                       Vector<T, N> &result) {
    result[N_idx] = a[N_idx] * b[N_idx];
    VectorMultiplyCore<T, N, N_idx - 1>::multiply(a, b, result);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct VectorMultiplyCore<T, N, 0> {
  static void multiply(const Vector<T, N> &a, const Vector<T, N> &b,
                       Vector<T, N> &result) {
    result[0] = a[0] * b[0];
  }
};

template <typename T, std::size_t N>
static inline void VECTOR_MULTIPLY(const Vector<T, N> &a, const Vector<T, N> &b,
                                   Vector<T, N> &result) {
  VectorMultiplyCore<T, N, N - 1>::multiply(a, b, result);
}

template <typename T, std::size_t N>
inline Vector<T, N> operator*(const Vector<T, N> &a, const Vector<T, N> &b) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    result[i] = a[i] * b[i];
  }

#else

  Base::Matrix::VECTOR_MULTIPLY<T, N>(a, b, result);

#endif

  return result;
}

/* Complex Norm */
// Vector Multiply Scalar Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct ComplexVectorNormCore {
  static T compute(const Vector<Complex<T>, N> &vec_comp) {
    return vec_comp[N_idx].real * vec_comp[N_idx].real +
           vec_comp[N_idx].imag * vec_comp[N_idx].imag +
           ComplexVectorNormCore<T, N, N_idx - 1>::compute(vec_comp);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N> struct ComplexVectorNormCore<T, N, 0> {
  static T compute(const Vector<Complex<T>, N> &vec_comp) {
    return vec_comp[0].real * vec_comp[0].real +
           vec_comp[0].imag * vec_comp[0].imag;
  }
};

template <typename T, std::size_t N>
static inline T
COMPILED_COMPLEX_VECTOR_NORM(const Vector<Complex<T>, N> &vec_comp) {
  return ComplexVectorNormCore<T, N, N - 1>::compute(vec_comp);
}

template <typename T, std::size_t N>
inline T complex_vector_norm(const Vector<Complex<T>, N> &vec_comp) {
  T sum = static_cast<T>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else

  sum = Base::Matrix::COMPILED_COMPLEX_VECTOR_NORM<T, N>(vec_comp);

#endif

  return Base::Math::sqrt_base_math<
      T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum);
}

template <typename T, std::size_t N>
inline T complex_vector_norm(const Vector<Complex<T>, N> &vec_comp,
                             const T &division_min) {
  T sum = static_cast<T>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else

  sum = Base::Matrix::COMPILED_COMPLEX_VECTOR_NORM<T, N>(vec_comp);

#endif

  return Base::Math::sqrt_base_math<
      T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum, division_min);
}

template <typename T, std::size_t N>
inline T complex_vector_norm_inv(const Vector<Complex<T>, N> &vec_comp) {
  T sum = static_cast<T>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else

  sum = Base::Matrix::COMPILED_COMPLEX_VECTOR_NORM<T, N>(vec_comp);

#endif

  return Base::Math::rsqrt_base_math<
      T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum);
}

template <typename T, std::size_t N>
inline T complex_vector_norm_inv(const Vector<Complex<T>, N> &vec_comp,
                                 const T &division_min) {
  T sum = static_cast<T>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

#else

  sum = Base::Matrix::COMPILED_COMPLEX_VECTOR_NORM<T, N>(vec_comp);

#endif

  return Base::Math::rsqrt_base_math<
      T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(sum, division_min);
}

/* Complex Normalize */
template <typename T, std::size_t N>
inline void complex_vector_normalize(Vector<Complex<T>, N> &vec) {
  T norm_inv = complex_vector_norm_inv(vec);
  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }
}

template <typename T, std::size_t N>
inline void complex_vector_normalize(Vector<Complex<T>, N> &vec,
                                     const T &division_min) {
  T norm_inv = complex_vector_norm_inv(vec, division_min);
  for (std::size_t i = 0; i < N; ++i) {
    vec[i] *= norm_inv;
  }
}

/* Get Real and Imaginary Vector from Complex Vector */
// Get Real from Complex Vector Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct GetRealFromComplexVectorCore {
  // For std::vector
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    GetRealFromComplexVectorCore<T, N, N_idx - 1>::compute(To_vector,
                                                           From_vector);
  }

  // For std::array
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    GetRealFromComplexVectorCore<T, N, N_idx - 1>::compute(To_vector,
                                                           From_vector);
  }

  // For Vector
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].real;
    GetRealFromComplexVectorCore<T, N, N_idx - 1>::compute(To_vector,
                                                           From_vector);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N>
struct GetRealFromComplexVectorCore<T, N, 0> {
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

// Get Imag from Complex Vector Core Template: N_idx < N
template <typename T, std::size_t N, std::size_t N_idx>
struct GetImagFromComplexVectorCore {
  // For std::vector
  static void compute(std::vector<T> &To_vector,
                      const std::vector<Complex<T>> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    GetImagFromComplexVectorCore<T, N, N_idx - 1>::compute(To_vector,
                                                           From_vector);
  }

  // For std::array
  static void compute(std::array<T, N> &To_vector,
                      const std::array<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    GetImagFromComplexVectorCore<T, N, N_idx - 1>::compute(To_vector,
                                                           From_vector);
  }

  // For Vector
  static void compute(Vector<T, N> &To_vector,
                      const Vector<Complex<T>, N> &From_vector) {
    To_vector[N_idx] = From_vector[N_idx].imag;
    GetImagFromComplexVectorCore<T, N, N_idx - 1>::compute(To_vector,
                                                           From_vector);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t N>
struct GetImagFromComplexVectorCore<T, N, 0> {
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

#ifdef BASE_MATRIX_USE_STD_VECTOR

template <typename T, std::size_t N>
inline std::vector<T> get_real_vector_from_complex_vector(
    const std::vector<Complex<T>> &From_vector) {

  std::vector<T> To_vector(N);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }

#else

  GetRealFromComplexVectorCore<T, N, N - 1>::compute(To_vector, From_vector);

#endif

  return To_vector;
}

template <typename T, std::size_t N>
inline std::vector<T> get_imag_vector_from_complex_vector(
    const std::vector<Complex<T>> &From_vector) {

  std::vector<T> To_vector(N);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }

#else

  GetImagFromComplexVectorCore<T, N, N - 1>::compute(To_vector, From_vector);

#endif

  return To_vector;
}

#else

template <typename T, std::size_t N>
inline std::array<T, N> get_real_vector_from_complex_vector(
    const std::array<Complex<T>, N> &From_vector) {

  std::array<T, N> To_vector;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }

#else

  GetRealFromComplexVectorCore<T, N, N - 1>::compute(To_vector, From_vector);

#endif

  return To_vector;
}

template <typename T, std::size_t N>
inline std::array<T, N> get_imag_vector_from_complex_vector(
    const std::array<Complex<T>, N> &From_vector) {

  std::array<T, N> To_vector;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }

#else

  GetImagFromComplexVectorCore<T, N, N - 1>::compute(To_vector, From_vector);

#endif

  return To_vector;
}

#endif

template <typename T, std::size_t N>
inline Vector<T, N>
get_real_vector_from_complex_vector(const Vector<Complex<T>, N> &From_vector) {

  Vector<T, N> To_vector;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }

#else

  GetRealFromComplexVectorCore<T, N, N - 1>::compute(To_vector, From_vector);

#endif

  return To_vector;
}

template <typename T, std::size_t N>
inline Vector<T, N>
get_imag_vector_from_complex_vector(const Vector<Complex<T>, N> &From_vector) {

  Vector<T, N> To_vector;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }

#else

  GetImagFromComplexVectorCore<T, N, N - 1>::compute(To_vector, From_vector);

#endif

  return To_vector;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_VECTOR_HPP
