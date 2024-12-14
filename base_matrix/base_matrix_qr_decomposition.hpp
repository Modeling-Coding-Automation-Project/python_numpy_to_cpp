#ifndef BASE_MATRIX_QR_DECOMPOSITION_HPP
#define BASE_MATRIX_QR_DECOMPOSITION_HPP

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
    s = Base::Math::rsqrt_base_math<
        T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(
        static_cast<T>(1) + t * t, division_min);
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
    c = Base::Math::rsqrt_base_math<
        T, Base::Math::SQRT_REPEAT_NUMBER_MOSTLY_ACCURATE>(
        static_cast<T>(1) + t * t, division_min);
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
  QRDecomposition() : _division_min(static_cast<T>(0)) {}

  QRDecomposition(const Matrix<T, M, N> &A, T division_min)
      : _Q_matrix(Matrix<T, M, M>::identity()), _R_matrix(A),
        _division_min(division_min) {
    this->_decompose();
  }

  /* Copy Constructor */
  QRDecomposition(const QRDecomposition<T, M, N> &other)
      : _Q_matrix(other._Q_matrix), _R_matrix(other._R_matrix),
        _division_min(other._division_min) {}

  QRDecomposition<T, M, N> &operator=(const QRDecomposition<T, M, N> &other) {
    if (this != &other) {
      this->_Q_matrix = other._Q_matrix;
      this->_R_matrix = other._R_matrix;
      this->_division_min = other._division_min;
    }

    return *this;
  }

  /* Move Constructor */
  QRDecomposition(QRDecomposition<T, M, N> &&other) noexcept
      : _Q_matrix(std::move(other._Q_matrix)),
        _R_matrix(std::move(other._R_matrix)),
        _division_min(other._division_min) {}

  QRDecomposition<T, M, N> &
  operator=(QRDecomposition<T, M, N> &&other) noexcept {
    if (this != &other) {
      this->_Q_matrix = std::move(other._Q_matrix);
      this->_R_matrix = std::move(other._R_matrix);
      this->_division_min = other._division_min;
    }

    return *this;
  }

  /* Function */
  inline Matrix<T, M, M> get_Q() const { return this->_Q_matrix; }
  inline Matrix<T, M, N> get_R() const { return this->_R_matrix; }

private:
  /* Variable */
  Matrix<T, M, M> _Q_matrix;
  Matrix<T, M, N> _R_matrix;
  T _division_min;

  /* Function */
  inline void _decompose() {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {

        if (!Base::Utility::near_zero(this->_R_matrix(i, j),
                                      this->_division_min)) {
          this->_givensRotation(i, j);
        }
      }
    }
  }

  inline void _givensRotation(std::size_t i, std::size_t j) {
    Base::Matrix::qr_givensRotation(i, j, this->_Q_matrix, this->_R_matrix,
                                    this->_division_min);
  }
};

template <typename T, std::size_t M> class QRDecompositionDiag {
public:
  QRDecompositionDiag() : _division_min(static_cast<T>(0)) {}

  QRDecompositionDiag(const DiagMatrix<T, M> &A, T division_min)
      : _Q_matrix(DiagMatrix<T, M>::identity()), _R_matrix(A),
        _division_min(division_min) {}

  inline DiagMatrix<T, M> get_Q() const { return this->_Q_matrix; }
  inline DiagMatrix<T, M> get_R() const { return this->_R_matrix; }

private:
  DiagMatrix<T, M> _Q_matrix;
  DiagMatrix<T, M> _R_matrix;
  T _division_min;
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
class QRDecompositionSparse {
public:
  QRDecompositionSparse() : _division_min(static_cast<T>(0)) {}

  QRDecompositionSparse(
      const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
      T division_min)
      : _Q_matrix(Matrix<T, M, M>::identity()), _R_matrix(A.create_dense()),
        _division_min(division_min) {
    this->_decompose(A);
  }

  inline Matrix<T, M, M> get_Q() const { return this->_Q_matrix; }
  inline Matrix<T, M, N> get_R() const { return this->_R_matrix; }

private:
  Matrix<T, M, M> _Q_matrix;
  Matrix<T, M, N> _R_matrix;
  T _division_min;

  inline void _decompose(
      const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t k = RowPointers_A::list[i];
           k < RowPointers_A::list[i + 1]; k++) {
        if ((i >= RowIndices_A::list[k] + 1) &&
            (!Base::Utility::near_zero(A.values[k], this->_division_min))) {
          this->_givensRotation(i, RowIndices_A::list[k]);
        }
      }
    }
  }

  inline void _givensRotation(std::size_t i, std::size_t j) {
    Base::Matrix::qr_givensRotation(i, j, this->_Q_matrix, this->_R_matrix,
                                    this->_division_min);
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_QR_DECOMPOSITION_HPP
