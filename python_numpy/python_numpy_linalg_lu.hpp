#ifndef __PYTHON_NUMPY_LINALG_LU_HPP__
#define __PYTHON_NUMPY_LINALG_LU_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_LU = 1.0e-10;

template <typename T, std::size_t M, typename SparseAvailable>
class LinalgSolverLU {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using SparseAvailable_Type = SparseAvailable;

  using UpperTriangular_SparseAvailable_Type =
      CreateSparseAvailableFromIndicesAndPointers<
          M, UpperTriangularRowIndices<M, M>, UpperTriangularRowPointers<M, M>>;

  using LowerTriangular_SparseAvailable_Type =
      CreateSparseAvailableFromIndicesAndPointers<
          M, LowerTriangularRowIndices<M, M>, LowerTriangularRowPointers<M, M>>;

public:
  /* Constructor */
  LinalgSolverLU() {}

  /* Copy Constructor */
  LinalgSolverLU(const LinalgSolverLU<T, M, SparseAvailable> &other)
      : division_min(other.division_min), _LU_decomposer(other._LU_decomposer),
        _L_triangular(other._L_triangular), _U_triangular(other._U_triangular) {
  }

  LinalgSolverLU<T, M, SparseAvailable> &
  operator=(const LinalgSolverLU<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_LU_decomposer = other._LU_decomposer;
      this->_L_triangular = other._L_triangular;
      this->_U_triangular = other._U_triangular;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverLU(LinalgSolverLU<T, M, SparseAvailable> &&other) noexcept
      : division_min(std::move(other.division_min)),
        _LU_decomposer(std::move(other._LU_decomposer)),
        _L_triangular(std::move(other._L_triangular)),
        _U_triangular(std::move(other._U_triangular)) {}

  LinalgSolverLU<T, M, SparseAvailable> &
  operator=(LinalgSolverLU<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->division_min = std::move(other.division_min);
      this->_LU_decomposer = std::move(other._LU_decomposer);
      this->_L_triangular = std::move(other._L_triangular);
      this->_U_triangular = std::move(other._U_triangular);
    }
    return *this;
  }

  /* Solve function */
  inline void solve(const Matrix<DefDense, T, M, M> &A) {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<T, M>();
    this->_LU_decomposer.division_min = this->division_min;
    this->_LU_decomposer.solve(A.matrix);
  }

  inline void solve(const Matrix<DefDiag, T, M> &A) {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<T, M>(A.matrix);
  }

  inline void solve(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    auto A_dense = A.matrix.create_dense();

    this->_LU_decomposer = Base::Matrix::LUDecomposition<T, M>();
    this->_LU_decomposer.division_min = this->division_min;
    this->_LU_decomposer.solve(A_dense);
  }

  /* Get */
  inline auto get_L() -> Matrix<DefSparse, T, M, M,
                                LowerTriangular_SparseAvailable_Type> const {

    Base::Matrix::TriangularSparse<T, M, M>::set_values_lower(
        this->_L_triangular, this->_LU_decomposer.get_L());

    return Matrix<DefSparse, T, M, M, LowerTriangular_SparseAvailable_Type>(
        this->_L_triangular);
  }

  inline auto get_U() -> Matrix<DefSparse, T, M, M,
                                UpperTriangular_SparseAvailable_Type> const {

    Base::Matrix::TriangularSparse<T, M, M>::set_values_upper(
        this->_U_triangular, this->_LU_decomposer.get_U());

    return Matrix<DefSparse, T, M, M, UpperTriangular_SparseAvailable_Type>(
        this->_U_triangular);
  }

  inline T get_det() { return this->_LU_decomposer.get_determinant(); }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

public:
  /* Variable */
  T division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_LU);

private:
  /* Variable */
  Base::Matrix::LUDecomposition<T, M> _LU_decomposer;
  Base::Matrix::CompiledSparseMatrix<
      T, M, M, Base::Matrix::LowerTriangularRowIndices<M, M>,
      Base::Matrix::LowerTriangularRowPointers<M, M>>
      _L_triangular = Base::Matrix::TriangularSparse<T, M, M>::create_lower();
  Base::Matrix::CompiledSparseMatrix<
      T, M, M, Base::Matrix::UpperTriangularRowIndices<M, M>,
      Base::Matrix::UpperTriangularRowPointers<M, M>>
      _U_triangular = Base::Matrix::TriangularSparse<T, M, M>::create_upper();
};

/* make LinalgSolverLU */
template <typename A_Type>
inline auto make_LinalgSolverLU(void)
    -> LinalgSolverLU<typename A_Type::Value_Type, A_Type::COLS,
                      typename A_Type::SparseAvailable_Type> {

  return LinalgSolverLU<typename A_Type::Value_Type, A_Type::COLS,
                        typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolverLU Type */
template <typename A_Type>
using LinalgSolverLU_Type =
    LinalgSolverLU<typename A_Type::Value_Type, A_Type::COLS,
                   typename A_Type::SparseAvailable_Type>;

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_LU_HPP__
