#ifndef BASE_MATRIX_VECTOR_HPP
#define BASE_MATRIX_VECTOR_HPP

#include "base_matrix_complex.hpp"
#include "base_matrix_utility.hpp"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

/* Vector */
template <typename T, std::size_t N> class Vector {
public:
  Vector() : data(N, static_cast<T>(0)) {}

  Vector(const std::initializer_list<T> &input) : data(input) {}

  Vector(const std::vector<T> &input) : data(input) {}

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

  std::size_t size() const { return N; }

  static Vector<T, N> Ones(void) {
    return Vector<T, N>(std::vector<T>(N, static_cast<T>(1)));
  }

  Vector<T, N> operator+(const T &scalar) const {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = this->data[i] + scalar;
    }
    return result;
  }

  Vector<T, N> operator+(const Vector<T, N> &vec) const {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = this->data[i] + vec[i];
    }
    return result;
  }

  Vector<T, N> operator-(const T &scalar) const {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = this->data[i] - scalar;
    }
    return result;
  }

  Vector<T, N> operator-(const Vector<T, N> &vec) const {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = this->data[i] - vec[i];
    }
    return result;
  }

  Vector<T, N> operator*(const T &scalar) const {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = this->data[i] * scalar;
    }
    return result;
  }

  Vector<T, N> operator*(const Vector<T, N> &vec) const {
    Vector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = this->data[i] * vec[i];
    }
    return result;
  }

  T dot(const Vector<T, N> &other) const {
    T result = static_cast<T>(0);
    for (std::size_t i = 0; i < N; ++i) {
      result += this->data[i] * other[i];
    }
    return result;
  }

  T norm() const {
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < N; ++i) {
      sum += this->data[i] * this->data[i];
    }
    return std::sqrt(sum);
  }

  /* Variable */
  std::vector<T> data;
};

template <typename T, std::size_t N> class ColVector : public Vector<T, N> {
public:
  ColVector() : Vector<T, N>() {}

  ColVector(const Vector<T, N> &vec) : Vector<T, N>(vec) {}

  Vector<T, N> transpose() const {
    Vector<T, N> result;

    std::memcpy(&result[0], &this->data[0], N * sizeof(result[0]));

    return result;
  }
};

template <typename T, std::size_t N>
Vector<T, N> operator+(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar + vec[i];
  }
  return result;
}

template <typename T, std::size_t N>
Vector<T, N> operator-(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar - vec[i];
  }
  return result;
}

template <typename T, std::size_t N>
Vector<T, N> operator*(const T &scalar, const Vector<T, N> &vec) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar * vec[i];
  }
  return result;
}

template <typename T, std::size_t N>
T complex_vector_norm(const Vector<Complex<T>, N> &vec_comp) {
  T sum = static_cast<T>(0);

  for (std::size_t i = 0; i < N; ++i) {
    sum += vec_comp[i].real * vec_comp[i].real +
           vec_comp[i].imag * vec_comp[i].imag;
  }

  return std::sqrt(sum);
}

template <typename T>
void get_real_vector_from_complex_vector(
    std::vector<T> &To_vector, const std::vector<Complex<T>> &From_vector,
    std::size_t N) {

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }
}

template <typename T, std::size_t N>
void get_real_vector_from_complex_vector(
    Vector<T, N> &To_vector, const Vector<Complex<T>, N> &From_vector) {

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].real;
  }
}

template <typename T>
void get_imag_vector_from_complex_vector(
    std::vector<T> &To_vector, const std::vector<Complex<T>> &From_vector,
    std::size_t N) {

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }
}

template <typename T, std::size_t N>
void get_imag_vector_from_complex_vector(
    Vector<T, N> &To_vector, const Vector<Complex<T>, N> &From_vector) {

  for (std::size_t i = 0; i < N; ++i) {
    To_vector[i] = From_vector[i].imag;
  }
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_VECTOR_HPP
