#ifndef __PYTHON_NUMPY_LINALG_LU_HPP__
#define __PYTHON_NUMPY_LINALG_LU_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_LU = 1.0e-10;

template <typename A_Type> class LinalgSolverLU {
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

  using LowerTriangular_SparseAvailable_Type =
      CreateSparseAvailableFromIndicesAndPointers<
          A_Type::COLS, LowerTriangularRowIndices<A_Type::COLS, A_Type::COLS>,
          LowerTriangularRowPointers<A_Type::COLS, A_Type::COLS>>;

private:
  /* Type */
  using _T = typename A_Type::Value_Type;

public:
  /* Constructor */
  template <
      typename U = A_Type,
      typename std::enable_if<Is_Dense_Matrix<U>::value>::type * = nullptr>
  LinalgSolverLU() {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<_T, A_Type::COLS>();
  }

  template <typename U = A_Type,
            typename std::enable_if<Is_Diag_Matrix<U>::value>::type * = nullptr>
  LinalgSolverLU() {}

  template <
      typename U = A_Type,
      typename std::enable_if<Is_Sparse_Matrix<U>::value>::type * = nullptr>
  LinalgSolverLU() {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<_T, A_Type::COLS>();
  }

  /* Copy Constructor */
  LinalgSolverLU(const LinalgSolverLU<A_Type> &other)
      : _LU_decomposer(other._LU_decomposer),
        _L_triangular(other._L_triangular), _U_triangular(other._U_triangular) {
  }

  LinalgSolverLU<A_Type> &operator=(const LinalgSolverLU<A_Type> &other) {
    if (this != &other) {
      this->_LU_decomposer = other._LU_decomposer;
      this->_L_triangular = other._L_triangular;
      this->_U_triangular = other._U_triangular;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverLU(LinalgSolverLU<A_Type> &&other) noexcept
      : _LU_decomposer(std::move(other._LU_decomposer)),
        _L_triangular(std::move(other._L_triangular)),
        _U_triangular(std::move(other._U_triangular)) {}

  LinalgSolverLU<A_Type> &operator=(LinalgSolverLU<A_Type> &&other) noexcept {
    if (this != &other) {

      this->_LU_decomposer = std::move(other._LU_decomposer);
      this->_L_triangular = std::move(other._L_triangular);
      this->_U_triangular = std::move(other._U_triangular);
    }
    return *this;
  }

  /* Solve function */
  inline void solve(const Matrix<DefDense, _T, A_Type::COLS, A_Type::COLS> &A) {
    this->_LU_decomposer.solve(A.matrix);
  }

  inline void solve(const Matrix<DefDiag, _T, A_Type::COLS> &A) {
    this->_LU_decomposer =
        Base::Matrix::LUDecomposition<_T, A_Type::COLS>(A.matrix);
  }

  inline void solve(const Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                                 SparseAvailable_Type> &A) {

    auto A_dense = A.matrix.create_dense();
    this->_LU_decomposer.solve(A_dense);
  }

  /* Get */
  inline auto get_L() -> Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                                LowerTriangular_SparseAvailable_Type> const {

    Base::Matrix::TriangularSparse<_T, A_Type::COLS, A_Type::COLS>::
        set_values_lower(this->_L_triangular, this->_LU_decomposer.get_L());

    return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                  LowerTriangular_SparseAvailable_Type>(this->_L_triangular);
  }

  inline auto get_U() -> Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                                UpperTriangular_SparseAvailable_Type> const {

    Base::Matrix::TriangularSparse<_T, A_Type::COLS, A_Type::COLS>::
        set_values_upper(this->_U_triangular, this->_LU_decomposer.get_U());

    return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                  UpperTriangular_SparseAvailable_Type>(this->_U_triangular);
  }

  inline _T get_det() { return this->_LU_decomposer.get_determinant(); }

  /* Set */
  inline void set_division_min(const _T &division_min) {
    this->_LU_decomposer.division_min = division_min;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = A_Type::COLS;
  static constexpr std::size_t ROWS = A_Type::ROWS;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<_T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  Base::Matrix::LUDecomposition<_T, A_Type::COLS> _LU_decomposer;
  Base::Matrix::CompiledSparseMatrix<
      _T, A_Type::COLS, A_Type::COLS,
      Base::Matrix::LowerTriangularRowIndices<A_Type::COLS, A_Type::COLS>,
      Base::Matrix::LowerTriangularRowPointers<A_Type::COLS, A_Type::COLS>>
      _L_triangular =
          Base::Matrix::TriangularSparse<_T, A_Type::COLS,
                                         A_Type::COLS>::create_lower();

  Base::Matrix::CompiledSparseMatrix<
      _T, A_Type::COLS, A_Type::COLS,
      Base::Matrix::UpperTriangularRowIndices<A_Type::COLS, A_Type::COLS>,
      Base::Matrix::UpperTriangularRowPointers<A_Type::COLS, A_Type::COLS>>
      _U_triangular =
          Base::Matrix::TriangularSparse<_T, A_Type::COLS,
                                         A_Type::COLS>::create_upper();
};

/* make LinalgSolverLU */
template <typename A_Type>
inline auto make_LinalgSolverLU(void) -> LinalgSolverLU<A_Type> {

  return LinalgSolverLU<A_Type>();
}

/* LinalgSolverLU Type */
template <typename A_Type> using LinalgSolverLU_Type = LinalgSolverLU<A_Type>;

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_LU_HPP__
