#ifndef __BASE_MATRIX_QR_DECOMPOSITION_HPP__
#define __BASE_MATRIX_QR_DECOMPOSITION_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_QR = 1.0e-10;

/* Given's Rotation */
template <typename T, std::size_t M, std::size_t N>
static inline void
qr_givensRotation(std::size_t i, std::size_t j, Matrix<T, M, M> &Q_matrix,
                  Matrix<T, M, N> &R_matrix, T division_min) {
  T c;
  T s;

  if (Base::Math::abs(R_matrix(i, j)) > Base::Math::abs(R_matrix(j, j))) {
    T t = -R_matrix(j, j) /
          Base::Utility::avoid_zero_divide(R_matrix(i, j), division_min);
    s = Base::Math::rsqrt<T>(static_cast<T>(1) + t * t, division_min);
    c = s * t;

    for (std::size_t k = j; k < N; ++k) {
      T x = R_matrix(j, k);
      T y = R_matrix(i, k);
      R_matrix(j, k) = c * x - s * y;
      R_matrix(i, k) = s * x + c * y;
    }

    for (std::size_t k = 0; k < M; ++k) {
      T u = Q_matrix(k, j);
      T v = Q_matrix(k, i);
      Q_matrix(k, j) = c * u - s * v;
      Q_matrix(k, i) = s * u + c * v;
    }
  } else {
    T t = -R_matrix(i, j) /
          Base::Utility::avoid_zero_divide(R_matrix(j, j), division_min);
    c = Base::Math::rsqrt<T>(static_cast<T>(1) + t * t, division_min);
    s = c * t;

    for (std::size_t k = j; k < N; ++k) {
      T x = R_matrix(j, k);
      T y = R_matrix(i, k);
      R_matrix(j, k) = c * x - s * y;
      R_matrix(i, k) = s * x + c * y;
    }

    for (std::size_t k = 0; k < M; ++k) {
      T u = Q_matrix(k, j);
      T v = Q_matrix(k, i);
      Q_matrix(k, j) = c * u - s * v;
      Q_matrix(k, i) = s * u + c * v;
    }
  }
}

template <typename T, std::size_t M, std::size_t N> class QRDecomposition {
public:
  QRDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        _Q_matrix(Matrix<T, M, M>::identity()) {}

  /* Copy Constructor */
  QRDecomposition(const QRDecomposition<T, M, N> &other)
      : division_min(other.division_min), _Q_matrix(other._Q_matrix),
        _R_matrix(other._R_matrix) {}

  QRDecomposition<T, M, N> &operator=(const QRDecomposition<T, M, N> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = other._Q_matrix;
      this->_R_matrix = other._R_matrix;
    }

    return *this;
  }

  /* Move Constructor */
  QRDecomposition(QRDecomposition<T, M, N> &&other) noexcept
      : division_min(other.division_min), _Q_matrix(std::move(other._Q_matrix)),
        _R_matrix(std::move(other._R_matrix)) {}

  QRDecomposition<T, M, N> &
  operator=(QRDecomposition<T, M, N> &&other) noexcept {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = std::move(other._Q_matrix);
      this->_R_matrix = std::move(other._R_matrix);
    }

    return *this;
  }

  /* Function */
  inline void solve(const Matrix<T, M, N> &A) {
    this->_R_matrix = A;
    this->_decompose();
  }

  inline Matrix<T, M, M> get_Q() const { return this->_Q_matrix; }

  inline Matrix<T, M, N> get_R() const { return this->_R_matrix; }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  Matrix<T, M, M> _Q_matrix;
  Matrix<T, M, N> _R_matrix;

protected:
  /* Function */
  inline void _decompose() {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {

        if (!Base::Utility::near_zero(this->_R_matrix(i, j),
                                      this->division_min)) {
          this->_givensRotation(i, j);
        }
      }
    }
  }

  inline void _givensRotation(std::size_t i, std::size_t j) {
    Base::Matrix::qr_givensRotation(i, j, this->_Q_matrix, this->_R_matrix,
                                    this->division_min);
  }
};

template <typename T, std::size_t M> class QRDecompositionDiag {
public:
  /* Constructor */
  QRDecompositionDiag()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        _Q_matrix(DiagMatrix<T, M>::identity()) {}

public:
  /* Function */
  inline void solve(const DiagMatrix<T, M> &A) { this->_R_matrix = A; }

  inline DiagMatrix<T, M> get_Q() const { return this->_Q_matrix; }

  inline DiagMatrix<T, M> get_R() const { return this->_R_matrix; }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  DiagMatrix<T, M> _Q_matrix;
  DiagMatrix<T, M> _R_matrix;
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
class QRDecompositionSparse {
public:
  /* Constructor */
  QRDecompositionSparse()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        _Q_matrix(Matrix<T, M, M>::identity()) {}

public:
  /* Function */
  inline void
  solve(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
    this->_R_matrix = Base::Matrix::output_dense_matrix(A);
    this->_decompose(A);
  }

  inline Matrix<T, M, M> get_Q() const { return this->_Q_matrix; }

  inline Matrix<T, M, N> get_R() const { return this->_R_matrix; }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  Matrix<T, M, M> _Q_matrix;
  Matrix<T, M, N> _R_matrix;

protected:
  /* Function */
  inline void _decompose(
      const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t k = RowPointers_A::list[i];
           k < RowPointers_A::list[i + 1]; k++) {
        if ((i >= RowIndices_A::list[k] + 1) &&
            (!Base::Utility::near_zero(A.values[k], this->division_min))) {
          this->_givensRotation(i, RowIndices_A::list[k]);
        }
      }
    }
  }

  inline void _givensRotation(std::size_t i, std::size_t j) {
    Base::Matrix::qr_givensRotation(i, j, this->_Q_matrix, this->_R_matrix,
                                    this->division_min);
  }
};

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_QR_DECOMPOSITION_HPP__
