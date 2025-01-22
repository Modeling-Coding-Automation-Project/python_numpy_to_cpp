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
  using SparseAvailable_Type = SparseAvailable;

public:
  /* Constructor */
  LinalgSolverLU() {}

  LinalgSolverLU(const Matrix<DefDense, T, M, M> &A) { this->solve(A); }

  LinalgSolverLU(const Matrix<DefDiag, T, M> &A) { this->solve(A); }

  LinalgSolverLU(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->solve(A);
  }

  /* Copy Constructor */
  LinalgSolverLU(const LinalgSolverLU<T, M, SparseAvailable> &other)
      : _LU_decomposer(other._LU_decomposer) {}

  LinalgSolverLU<T, M, SparseAvailable> &
  operator=(const LinalgSolverLU<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->_LU_decomposer = other._LU_decomposer;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverLU(LinalgSolverLU<T, M, SparseAvailable> &&other) noexcept
      : _LU_decomposer(std::move(other._LU_decomposer)) {}

  LinalgSolverLU<T, M, SparseAvailable> &
  operator=(LinalgSolverLU<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_LU_decomposer = std::move(other._LU_decomposer);
    }
    return *this;
  }

  /* Solve function */
  inline void solve(const Matrix<DefDense, T, M, M> &A) {
    this->_LU_decomposer =
        Base::Matrix::LUDecomposition<T, M>(A.matrix, this->_division_min);
  }

  inline void solve(const Matrix<DefDiag, T, M> &A) {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<T, M>(A.matrix);
  }

  inline void solve(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<T, M>(
        A.matrix.create_dense(), this->_division_min);
  }

  /* Get */
  inline auto get_L() -> Matrix<DefSparse, T, M, M,
                                CreateSparseAvailableFromIndicesAndPointers<
                                    M, LowerTriangularRowIndices<M, M>,
                                    LowerTriangularRowPointers<M, M>>> const {

    Base::Matrix::TriangularSparse<T, M, M>::set_values_lower(
        this->_L_triangular, this->_LU_decomposer.get_L());

    return Matrix<DefSparse, T, M, M,
                  CreateSparseAvailableFromIndicesAndPointers<
                      M, LowerTriangularRowIndices<M, M>,
                      LowerTriangularRowPointers<M, M>>>(this->_L_triangular);
  }

  inline auto get_U() -> Matrix<DefSparse, T, M, M,
                                CreateSparseAvailableFromIndicesAndPointers<
                                    M, UpperTriangularRowIndices<M, M>,
                                    UpperTriangularRowPointers<M, M>>> const {

    Base::Matrix::TriangularSparse<T, M, M>::set_values_upper(
        this->_U_triangular, this->_LU_decomposer.get_U());

    return Matrix<DefSparse, T, M, M,
                  CreateSparseAvailableFromIndicesAndPointers<
                      M, UpperTriangularRowIndices<M, M>,
                      UpperTriangularRowPointers<M, M>>>(this->_U_triangular);
  }

  inline T get_det() { return this->_LU_decomposer.get_determinant(); }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

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

  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_LU);
};

/* make LinalgSolverLU */
template <typename T, std::size_t M,
          typename SparseAvailable = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgSolverLU(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolverLU<T, M, SparseAvailable> {

  return LinalgSolverLU<T, M, SparseAvailable>(A);
}

template <typename T, std::size_t M,
          typename SparseAvailable = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgSolverLU(const Matrix<DefDiag, T, M> &A)
    -> LinalgSolverLU<T, M, SparseAvailable> {

  return LinalgSolverLU<T, M, SparseAvailable>(A);
}

template <typename T, std::size_t M, typename SparseAvailable>
inline auto
make_LinalgSolverLU(const Matrix<DefSparse, T, M, M, SparseAvailable> &A)
    -> LinalgSolverLU<T, M, SparseAvailable> {

  return LinalgSolverLU<T, M, SparseAvailable>(A);
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_LU_HPP__
