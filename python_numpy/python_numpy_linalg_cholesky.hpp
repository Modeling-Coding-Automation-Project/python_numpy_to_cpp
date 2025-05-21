#ifndef __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__
#define __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

constexpr double DEFAULT_DIVISION_MIN_LINALG_CHOLESKY = 1.0e-10;

template <typename A_Type> class LinalgSolverCholesky {
public:
  /* Type */
  using Value_Type = typename A_Type::Value_Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

  using SparseAvailable_Type = typename A_Type::SparseAvailable_Type;

  using UpperTriangular_SparseAvailable_Type =
      CreateSparseAvailableFromIndicesAndPointers<
          A_Type::COLS, UpperTriangularRowIndices<A_Type::COLS, A_Type::COLS>,
          UpperTriangularRowPointers<A_Type::COLS, A_Type::COLS>>;

protected:
  /* Type */
  using _T = typename A_Type::Value_Type;

  using _CholeskyTriangularRowIndices =
      UpperTriangularRowIndices<A_Type::COLS, A_Type::COLS>;
  using _CholeskyTriangularRowPointers =
      UpperTriangularRowPointers<A_Type::COLS, A_Type::COLS>;

public:
  /* Constructor */
  LinalgSolverCholesky()
      : _cholesky_decomposed_matrix(),
        _cholesky_decomposed_triangular(
            Base::Matrix::TriangularSparse<_T, A_Type::COLS,
                                           A_Type::COLS>::create_upper()),
        _zero_div_flag(false) {}

  /* Copy Constructor */
  LinalgSolverCholesky(const LinalgSolverCholesky<A_Type> &other)
      : division_min(other.division_min),
        _cholesky_decomposed_matrix(other._cholesky_decomposed_matrix),
        _cholesky_decomposed_triangular(other._cholesky_decomposed_triangular),
        _zero_div_flag(other._zero_div_flag) {}

  LinalgSolverCholesky<A_Type> &
  operator=(const LinalgSolverCholesky<A_Type> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_cholesky_decomposed_matrix = other._cholesky_decomposed_matrix;
      this->_cholesky_decomposed_triangular =
          other._cholesky_decomposed_triangular;
      this->_zero_div_flag = other._zero_div_flag;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverCholesky(LinalgSolverCholesky<A_Type> &&other) noexcept
      : division_min(std::move(other.division_min)),
        _cholesky_decomposed_matrix(
            std::move(other._cholesky_decomposed_matrix)),
        _cholesky_decomposed_triangular(
            std::move(other._cholesky_decomposed_triangular)),
        _zero_div_flag(std::move(other._zero_div_flag)) {}

  LinalgSolverCholesky<A_Type> &
  operator=(LinalgSolverCholesky<A_Type> &&other) noexcept {
    if (this != &other) {
      this->division_min = std::move(other.division_min);
      this->_cholesky_decomposed_matrix =
          std::move(other._cholesky_decomposed_matrix);
      this->_cholesky_decomposed_triangular =
          std::move(other._cholesky_decomposed_triangular);
      this->_zero_div_flag = std::move(other._zero_div_flag);
    }
    return *this;
  }

  /* Solve function */
  inline auto solve(const Matrix<DefDense, _T, A_Type::COLS, A_Type::COLS> &A)
      -> Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                UpperTriangular_SparseAvailable_Type> {

    this->_cholesky_decomposed_matrix =
        Base::Matrix::cholesky_decomposition<_T, A_Type::COLS>(
            A.matrix, this->_cholesky_decomposed_matrix, this->division_min,
            this->_zero_div_flag);

    Base::Matrix::TriangularSparse<_T, A_Type::COLS, A_Type::COLS>::
        set_values_upper(this->_cholesky_decomposed_triangular,
                         this->_cholesky_decomposed_matrix);

    return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                  UpperTriangular_SparseAvailable_Type>(
        this->_cholesky_decomposed_triangular);
  }

  inline auto solve(const Matrix<DefDiag, _T, A_Type::COLS> &A)
      -> Matrix<DefDiag, _T, A_Type::COLS> {

    Base::Matrix::DiagMatrix<_T, A_Type::COLS> Diag(
        this->_cholesky_decomposed_matrix(0));

    Diag = Base::Matrix::cholesky_decomposition_diag<_T, A_Type::COLS>(
        A.matrix, Diag, this->_zero_div_flag);

    this->_cholesky_decomposed_matrix(0) = Diag.data;

    return Matrix<DefDiag, _T, A_Type::COLS>(Diag);
  }

  inline auto solve(const Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                                 SparseAvailable_Type> &A)
      -> Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                UpperTriangular_SparseAvailable_Type> {

    this->_cholesky_decomposed_matrix =
        Base::Matrix::cholesky_decomposition_sparse<_T, A_Type::COLS>(
            A.matrix, this->_cholesky_decomposed_matrix, this->division_min,
            this->_zero_div_flag);

    Base::Matrix::TriangularSparse<_T, A_Type::COLS, A_Type::COLS>::
        set_values_upper(this->_cholesky_decomposed_triangular,
                         this->_cholesky_decomposed_matrix);

    return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                  UpperTriangular_SparseAvailable_Type>(
        this->_cholesky_decomposed_triangular);
  }

public:
  /* Function */
  inline bool get_zero_div_flag() const { return this->_zero_div_flag; }

  inline void set_division_min(const _T &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = A_Type::COLS;
  static constexpr std::size_t ROWS = A_Type::ROWS;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<_T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

public:
  /* Variable */
  _T division_min = static_cast<_T>(DEFAULT_DIVISION_MIN_LINALG_CHOLESKY);

protected:
  /* Variable */
  Base::Matrix::Matrix<_T, A_Type::COLS, A_Type::COLS>
      _cholesky_decomposed_matrix;

  Base::Matrix::CompiledSparseMatrix<_T, A_Type::COLS, A_Type::COLS,
                                     _CholeskyTriangularRowIndices,
                                     _CholeskyTriangularRowPointers>
      _cholesky_decomposed_triangular;

  bool _zero_div_flag;
};

/* make LinalgSolverCholesky */
template <typename A_Type>
inline auto make_LinalgSolverCholesky(void) -> LinalgSolverCholesky<A_Type> {

  return LinalgSolverCholesky<A_Type>();
}

/* LinalgSolverCholesky Type */
template <typename A_Type>
using LinalgSolverCholesky_Type = LinalgSolverCholesky<A_Type>;

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__
