#ifndef __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__
#define __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

template <typename T, std::size_t M, typename SparseAvailable>
class LinalgSolverCholesky {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using SparseAvailable_Type = SparseAvailable;

public:
  /* Constructor */
  LinalgSolverCholesky() {}

  LinalgSolverCholesky(const Matrix<DefDense, T, M, M> &A) { this->solve(A); }

  LinalgSolverCholesky(const Matrix<DefDiag, T, M> &A) { this->solve(A); }

  LinalgSolverCholesky(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->solve(A);
  }

  /* Copy Constructor */
  LinalgSolverCholesky(const LinalgSolverCholesky<T, M, SparseAvailable> &other)
      : _cholesky_decomposed_matrix(other._cholesky_decomposed_matrix),
        _cholesky_decomposed_triangular(other._cholesky_decomposed_triangular),
        _zero_div_flag(other._zero_div_flag) {}

  LinalgSolverCholesky<T, M, SparseAvailable> &
  operator=(const LinalgSolverCholesky<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->_cholesky_decomposed_matrix = other._cholesky_decomposed_matrix;
      this->_cholesky_decomposed_triangular =
          other._cholesky_decomposed_triangular;
      this->_zero_div_flag = other._zero_div_flag;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverCholesky(
      LinalgSolverCholesky<T, M, SparseAvailable> &&other) noexcept
      : _cholesky_decomposed_matrix(
            std::move(other._cholesky_decomposed_matrix)),
        _cholesky_decomposed_triangular(
            std::move(other._cholesky_decomposed_triangular)),
        _zero_div_flag(std::move(other._zero_div_flag)) {}

  LinalgSolverCholesky<T, M, SparseAvailable> &
  operator=(LinalgSolverCholesky<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_cholesky_decomposed_matrix =
          std::move(other._cholesky_decomposed_matrix);
      this->_cholesky_decomposed_triangular =
          std::move(other._cholesky_decomposed_triangular);
      this->_zero_div_flag = std::move(other._zero_div_flag);
    }
    return *this;
  }

  /* Solve function */
  inline auto solve(const Matrix<DefDense, T, M, M> &A)
      -> Matrix<DefSparse, T, M, M,
                CreateSparseAvailableFromIndicesAndPointers<
                    M, UpperTriangularRowIndices<M, M>,
                    UpperTriangularRowPointers<M, M>>> {

    this->_cholesky_decomposed_matrix =
        Base::Matrix::cholesky_decomposition<T, M>(
            A.matrix, this->_cholesky_decomposed_matrix, this->_zero_div_flag);

    Base::Matrix::TriangularSparse<T, M, M>::set_values_upper(
        this->_cholesky_decomposed_triangular,
        this->_cholesky_decomposed_matrix);

    return Matrix<DefSparse, T, M, M,
                  CreateSparseAvailableFromIndicesAndPointers<
                      M, UpperTriangularRowIndices<M, M>,
                      UpperTriangularRowPointers<M, M>>>(
        this->_cholesky_decomposed_triangular);
  }

  inline auto solve(const Matrix<DefDiag, T, M> &A) -> Matrix<DefDiag, T, M> {

    Base::Matrix::DiagMatrix<T, M> Diag(this->_cholesky_decomposed_matrix(0));

    Diag = Base::Matrix::cholesky_decomposition_diag<T, M>(
        A.matrix, Diag, this->_zero_div_flag);

    this->_cholesky_decomposed_matrix(0) = Diag.data;

    return Matrix<DefDiag, T, M>(Diag);
  }

  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable> &A)
      -> Matrix<DefDense, T, M, M> {

    this->_cholesky_decomposed_matrix =
        Base::Matrix::cholesky_decomposition_sparse<T, M>(
            A.matrix, this->_cholesky_decomposed_matrix, this->_zero_div_flag);

    return Matrix<DefDense, T, M, M>(this->_cholesky_decomposed_matrix);
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  Base::Matrix::Matrix<T, M, M> _cholesky_decomposed_matrix;
  Base::Matrix::CompiledSparseMatrix<T, M, M, UpperTriangularRowIndices<M, M>,
                                     UpperTriangularRowPointers<M, M>>
      _cholesky_decomposed_triangular =
          Base::Matrix::TriangularSparse<T, M, M>::create_upper();
  bool _zero_div_flag = false;
};

/* make LinalgSolverCholesky */
template <typename T, std::size_t M,
          typename SparseAvailable = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgSolverCholesky(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolverCholesky<T, M, SparseAvailable> {

  return LinalgSolverCholesky<T, M, SparseAvailable>(A);
}

template <typename T, std::size_t M,
          typename SparseAvailable = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgSolverCholesky(const Matrix<DefDiag, T, M> &A)
    -> LinalgSolverCholesky<T, M, SparseAvailable> {

  return LinalgSolverCholesky<T, M, SparseAvailable>(A);
}

template <typename T, std::size_t M, typename SparseAvailable>
inline auto
make_LinalgSolverCholesky(const Matrix<DefSparse, T, M, M, SparseAvailable> &A)
    -> LinalgSolverCholesky<T, M, SparseAvailable> {

  return LinalgSolverCholesky<T, M, SparseAvailable>(A);
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__
