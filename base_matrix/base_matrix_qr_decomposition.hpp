#ifndef BASE_MATRIX_QR_DECOMPOSITION_HPP
#define BASE_MATRIX_QR_DECOMPOSITION_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"
#include <cmath>
#include <cstddef>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N> class QRDecomposition {
public:
  QRDecomposition() : _division_min(static_cast<T>(0)) {}

  QRDecomposition(const Matrix<T, M, N> &A, T division_min)
      : _Q(Matrix<T, M, M>::identity()), _R(A), _division_min(division_min) {
    this->_decompose();
  }

  /* Copy Constructor */
  QRDecomposition(const QRDecomposition<T, M, N> &other)
      : _Q(other._Q), _R(other._R), _division_min(other._division_min) {}

  QRDecomposition<T, M, N> &operator=(const QRDecomposition<T, M, N> &other) {
    if (this != &other) {
      this->_Q = other._Q;
      this->_R = other._R;
      this->_division_min = other._division_min;
    }

    return *this;
  }

  /* Move Constructor */
  QRDecomposition(QRDecomposition<T, M, N> &&other) noexcept
      : _Q(std::move(other._Q)), _R(std::move(other._R)),
        _division_min(other._division_min) {}

  QRDecomposition<T, M, N> &operator=(QRDecomposition<T, M, N> &&other) {
    if (this != &other) {
      this->_Q = std::move(other._Q);
      this->_R = std::move(other._R);
      this->_division_min = other._division_min;
    }

    return *this;
  }

  /* Function */
  Matrix<T, M, M> get_Q() const { return this->_Q; }
  Matrix<T, M, N> get_R() const { return this->_R; }

private:
  /* Variable */
  Matrix<T, M, M> _Q;
  Matrix<T, M, N> _R;
  T _division_min;

  /* Function */
  void _decompose() {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {

        if (!near_zero(this->_R(i, j), this->_division_min)) {
          this->_givensRotation(i, j);
        }
      }
    }
  }

  void _givensRotation(std::size_t i, std::size_t j) {
    T c;
    T s;

    if (std::abs(this->_R(i, j)) > std::abs(this->_R(j, j))) {
      T t = -this->_R(j, j) /
            avoid_zero_divide(this->_R(i, j), this->_division_min);
      s = static_cast<T>(1) / std::sqrt(static_cast<T>(1) + t * t);
      c = s * t;

      for (std::size_t k = j; k < N; ++k) {
        T x = this->_R(j, k);
        T y = this->_R(i, k);
        this->_R(j, k) = c * x - s * y;
        this->_R(i, k) = s * x + c * y;
      }

      for (std::size_t k = 0; k < M; ++k) {
        T u = this->_Q(k, j);
        T v = this->_Q(k, i);
        this->_Q(k, j) = c * u - s * v;
        this->_Q(k, i) = s * u + c * v;
      }
    } else {
      T t = -this->_R(i, j) /
            avoid_zero_divide(this->_R(j, j), this->_division_min);
      c = static_cast<T>(1) / std::sqrt(static_cast<T>(1) + t * t);
      s = c * t;

      for (std::size_t k = j; k < N; ++k) {
        T x = this->_R(j, k);
        T y = this->_R(i, k);
        this->_R(j, k) = c * x - s * y;
        this->_R(i, k) = s * x + c * y;
      }

      for (std::size_t k = 0; k < M; ++k) {
        T u = this->_Q(k, j);
        T v = this->_Q(k, i);
        this->_Q(k, j) = c * u - s * v;
        this->_Q(k, i) = s * u + c * v;
      }
    }
  }
};

template <typename T, std::size_t M> class QRDecompositionDiag {
public:
  QRDecompositionDiag() : _division_min(static_cast<T>(0)) {}

  QRDecompositionDiag(const DiagMatrix<T, M> &A, T division_min)
      : _Q(DiagMatrix<T, M>::identity()), _R(A), _division_min(division_min) {}

  DiagMatrix<T, M> get_Q() const { return this->_Q; }
  DiagMatrix<T, M> get_R() const { return this->_R; }

private:
  DiagMatrix<T, M> _Q;
  DiagMatrix<T, M> _R;
  T _division_min;
};

template <typename T, std::size_t M, std::size_t N, std::size_t V>
class QRDecompositionSparse {
public:
  QRDecompositionSparse() : _division_min(static_cast<T>(0)) {}

  QRDecompositionSparse(const SparseMatrix<T, M, N, V> &A, T division_min)
      : _Q(Matrix<T, M, M>::identity()), _R(A.create_dense()),
        _division_min(division_min) {
    this->_decompose(A);
  }

  Matrix<T, M, M> get_Q() const { return this->_Q; }
  Matrix<T, M, N> get_R() const { return this->_R; }

private:
  Matrix<T, M, M> _Q;
  Matrix<T, M, N> _R;
  T _division_min;

  void _decompose(const SparseMatrix<T, M, N, V> &A) {
    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t k = A.row_pointers[i]; k < A.row_pointers[i + 1]; k++) {
        if ((i >= A.row_indices[k] + 1) &&
            (!near_zero(A.values[k], this->_division_min))) {
          this->_givensRotation(i, A.row_indices[k]);
        }
      }
    }
  }

  void _givensRotation(std::size_t i, std::size_t j) {
    T c;
    T s;

    if (std::abs(this->_R(i, j)) > std::abs(this->_R(j, j))) {
      T t = -this->_R(j, j) /
            avoid_zero_divide(this->_R(i, j), this->_division_min);
      s = static_cast<T>(1) / std::sqrt(static_cast<T>(1) + t * t);
      c = s * t;

      for (std::size_t k = j; k < N; ++k) {
        T x = this->_R(j, k);
        T y = this->_R(i, k);
        this->_R(j, k) = c * x - s * y;
        this->_R(i, k) = s * x + c * y;
      }

      for (std::size_t k = 0; k < M; ++k) {
        T u = this->_Q(k, j);
        T v = this->_Q(k, i);
        this->_Q(k, j) = c * u - s * v;
        this->_Q(k, i) = s * u + c * v;
      }
    } else {
      T t = -this->_R(i, j) /
            avoid_zero_divide(this->_R(j, j), this->_division_min);
      c = static_cast<T>(1) / std::sqrt(static_cast<T>(1) + t * t);
      s = c * t;

      for (std::size_t k = j; k < N; ++k) {
        T x = this->_R(j, k);
        T y = this->_R(i, k);
        this->_R(j, k) = c * x - s * y;
        this->_R(i, k) = s * x + c * y;
      }

      for (std::size_t k = 0; k < M; ++k) {
        T u = this->_Q(k, j);
        T v = this->_Q(k, i);
        this->_Q(k, j) = c * u - s * v;
        this->_Q(k, i) = s * u + c * v;
      }
    }
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_QR_DECOMPOSITION_HPP
