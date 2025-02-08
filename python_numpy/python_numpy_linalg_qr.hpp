#ifndef __PYTHON_NUMPY_LINALG_QR_HPP__
#define __PYTHON_NUMPY_LINALG_QR_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_QR = 1.0e-10;

template <typename T, std::size_t M, std::size_t N> class LinalgSolverQR {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

public:
  /* Constructor */
  LinalgSolverQR() { this->_QR_decomposer.division_min = this->_division_min; }

  /* Copy Constructor */
  LinalgSolverQR(const LinalgSolverQR<T, M, N> &other)
      : _QR_decomposer(other._QR_decomposer),
        _R_triangular(other._R_triangular), _division_min(other._division_min) {
  }

  LinalgSolverQR<T, M, N> &operator=(const LinalgSolverQR<T, M, N> &other) {
    if (this != &other) {
      this->_QR_decomposer = other._QR_decomposer;
      this->_R_triangular = other._R_triangular;
      this->_division_min = other._division_min;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverQR(LinalgSolverQR<T, M, N> &&other) noexcept
      : _QR_decomposer(std::move(other._QR_decomposer)),
        _R_triangular(std::move(other._R_triangular)),
        _division_min(std::move(other._division_min)) {}

  LinalgSolverQR<T, M, N> &operator=(LinalgSolverQR<T, M, N> &&other) noexcept {
    if (this != &other) {
      this->_QR_decomposer = std::move(other._QR_decomposer);
      this->_R_triangular = std::move(other._R_triangular);
      this->_division_min = std::move(other._division_min);
    }
    return *this;
  }

  /* Solve function */
  inline void solve(const Matrix<DefDense, T, M, N> &A) {
    this->_QR_decomposer.solve(A.matrix);
  }

  /* Get Q, R */
  inline auto get_R(void)
      -> Matrix<DefSparse, T, M, N,
                CreateSparseAvailableFromIndicesAndPointers<
                    N, Base::Matrix::UpperTriangularRowIndices<M, N>,
                    Base::Matrix::UpperTriangularRowPointers<M, N>>> const {

    Base::Matrix::TriangularSparse<T, M, N>::set_values_upper(
        this->_R_triangular, this->_QR_decomposer.get_R());

    return Matrix<DefSparse, T, M, N,
                  CreateSparseAvailableFromIndicesAndPointers<
                      N, Base::Matrix::UpperTriangularRowIndices<M, N>,
                      Base::Matrix::UpperTriangularRowPointers<M, N>>>(
        this->_R_triangular);
  }

  inline auto get_Q(void) -> Matrix<DefDense, T, M, M> const {
    return Matrix<DefDense, T, M, M>(this->_QR_decomposer.get_Q());
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = N;

private:
  /* Properties */
  Base::Matrix::QRDecomposition<T, M, N> _QR_decomposer;
  Base::Matrix::CompiledSparseMatrix<
      T, M, N, Base::Matrix::UpperTriangularRowIndices<M, N>,
      Base::Matrix::UpperTriangularRowPointers<M, N>>
      _R_triangular = Base::Matrix::TriangularSparse<T, M, N>::create_upper();

  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_QR);
};

template <typename T, std::size_t M> class LinalgSolverQRDiag {
public:
  /* Constructor */
  LinalgSolverQRDiag() {}

  /* Copy Constructor */
  LinalgSolverQRDiag(const LinalgSolverQRDiag<T, M> &other) : _R(other._R) {}

  LinalgSolverQRDiag<T, M> &operator=(const LinalgSolverQRDiag<T, M> &other) {
    if (this != &other) {
      this->_R = other._R;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverQRDiag(LinalgSolverQRDiag<T, M> &&other) noexcept
      : _R(std::move(other._R)) {}

  LinalgSolverQRDiag<T, M> &
  operator=(LinalgSolverQRDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->_R = std::move(other._R);
    }
    return *this;
  }

public:
  /* Function */
  inline void solve(const Matrix<DefDiag, T, M> &A) { this->_R = A; }

  inline auto get_R(void) -> Matrix<DefDiag, T, M> const { return this->_R; }

  inline auto get_Q(void) -> Matrix<DefDiag, T, M> const {
    return Matrix<DefDiag, T, M>::identity();
  }

private:
  /* Properties */
  Matrix<DefDiag, T, M> _R;
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
class LinalgSolverQRSparse {
public:
  /* Constructor */
  LinalgSolverQRSparse() {
    this->_QR_decomposer.division_min = this->_division_min;
  }

  /* Copy Constructor */
  LinalgSolverQRSparse(
      const LinalgSolverQRSparse<T, M, N, SparseAvailable> &other)
      : _QR_decomposer(other._QR_decomposer),
        _R_triangular(other._R_triangular), _division_min(other._division_min) {
  }

  LinalgSolverQRSparse<T, M, N, SparseAvailable> &
  operator=(const LinalgSolverQRSparse<T, M, N, SparseAvailable> &other) {
    if (this != &other) {
      this->_QR_decomposer = other._QR_decomposer;
      this->_R_triangular = other._R_triangular;
      this->_division_min = other._division_min;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverQRSparse(
      LinalgSolverQRSparse<T, M, N, SparseAvailable> &&other) noexcept
      : _QR_decomposer(std::move(other._QR_decomposer)),
        _R_triangular(std::move(other._R_triangular)),
        _division_min(std::move(other._division_min)) {}

  LinalgSolverQRSparse<T, M, N, SparseAvailable> &
  operator=(LinalgSolverQRSparse<T, M, N, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_QR_decomposer = std::move(other._QR_decomposer);
      this->_R_triangular = std::move(other._R_triangular);
      this->_division_min = std::move(other._division_min);
    }
    return *this;
  }

  /* Solve function */
  inline void solve(const Matrix<DefSparse, T, M, N, SparseAvailable> &A) {
    this->_QR_decomposer.solve(A.matrix);
  }

  /* Get Q, R */
  inline auto get_R(void)
      -> Matrix<DefSparse, T, M, N,
                CreateSparseAvailableFromIndicesAndPointers<
                    N, Base::Matrix::UpperTriangularRowIndices<M, N>,
                    Base::Matrix::UpperTriangularRowPointers<M, N>>> const {

    Base::Matrix::TriangularSparse<T, M, N>::set_values_upper(
        this->_R_triangular, this->_QR_decomposer.get_R());

    return Base::Matrix::CompiledSparseMatrix<
        T, M, N, Base::Matrix::UpperTriangularRowIndices<M, M>,
        Base::Matrix::UpperTriangularRowPointers<M, N>>(this->_R_triangular);
  }

  inline auto get_Q(void) -> Matrix<DefDense, T, M, M> const {
    return Matrix<DefDense, T, M, M>(this->_QR_decomposer.get_Q());
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = N;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  Base::Matrix::QRDecompositionSparse<
      T, M, N, Base::Matrix::RowIndicesFromSparseAvailable<SparseAvailable>,
      Base::Matrix::RowPointersFromSparseAvailable<SparseAvailable>>
      _QR_decomposer;

  Base::Matrix::CompiledSparseMatrix<
      T, M, N, Base::Matrix::UpperTriangularRowIndices<M, N>,
      Base::Matrix::UpperTriangularRowPointers<M, N>>
      _R_triangular = Base::Matrix::TriangularSparse<T, M, N>::create_upper();

  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_QR);
};

/* make LinalgSolverQR */
template <
    typename A_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverQR(void)
    -> LinalgSolverQR<typename A_Type::Value_Type, A_Type::COLS, A_Type::ROWS> {

  return LinalgSolverQR<typename A_Type::Value_Type, A_Type::COLS,
                        A_Type::ROWS>();
}

template <typename A_Type, typename std::enable_if<
                               Is_Diag_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverQR(void)
    -> LinalgSolverQRDiag<typename A_Type::Value_Type, A_Type::COLS> {

  return LinalgSolverQRDiag<typename A_Type::Value_Type, A_Type::COLS>();
}

template <
    typename A_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverQR(void)
    -> LinalgSolverQRSparse<typename A_Type::Value_Type, A_Type::COLS,
                            A_Type::ROWS,
                            typename A_Type::SparseAvailable_Type> {

  return LinalgSolverQRSparse<typename A_Type::Value_Type, A_Type::COLS,
                              A_Type::ROWS,
                              typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolverQR Type */
template <typename A_Type>
using LinalgSolverQR_Type = decltype(make_LinalgSolverQR<A_Type>());

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_QR_HPP__
