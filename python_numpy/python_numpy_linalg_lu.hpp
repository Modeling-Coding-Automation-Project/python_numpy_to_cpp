#ifndef PYTHON_NUMPY_LINALG_LU_HPP
#define PYTHON_NUMPY_LINALG_LU_HPP

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include <cstddef>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_LU = 1.0e-10;

template <typename T, std::size_t M, std::size_t V> class LinalgSolverLU {
public:
  /* Constructor */
  LinalgSolverLU() {}

  LinalgSolverLU(const Matrix<DefDense, T, M, M> &A) { this->solve(A); }

  LinalgSolverLU(const Matrix<DefDiag, T, M> &A) { this->solve(A); }

  LinalgSolverLU(const Matrix<DefSparse, T, M, M, V> &A) { this->solve(A); }

  /* Copy Constructor */
  LinalgSolverLU(const LinalgSolverLU<T, M, V> &other)
      : _LU_decomposer(other._LU_decomposer) {}

  LinalgSolverLU<T, M, V> &operator=(const LinalgSolverLU<T, M, V> &other) {
    if (this != &other) {
      this->_LU_decomposer = other._LU_decomposer;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverLU(LinalgSolverLU<T, M, V> &&other) noexcept
      : _LU_decomposer(std::move(other._LU_decomposer)) {}

  LinalgSolverLU<T, M, V> &operator=(LinalgSolverLU<T, M, V> &&other) noexcept {
    if (this != &other) {
      this->_LU_decomposer = std::move(other._LU_decomposer);
    }
    return *this;
  }

  /* Solve function */
  void solve(const Matrix<DefDense, T, M, M> &A) {
    this->_LU_decomposer =
        Base::Matrix::LUDecomposition<T, M>(A.matrix, this->_division_min);
  }

  void solve(const Matrix<DefDiag, T, M> &A) {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<T, M>(A.matrix);
  }

  void solve(const Matrix<DefSparse, T, M, M, V> &A) {
    this->_LU_decomposer = Base::Matrix::LUDecomposition<T, M>(
        A.matrix.create_dense(), this->_division_min);
  }

  /* Get */
  auto get_L()
      -> Matrix<DefSparse, T, M, M,
                Base::Matrix::CalculateTriangularSize<M, M>::value> const {

    Base::Matrix::TriangularSparse<T, M, M>::set_values_lower(
        this->_L_triangular, this->_LU_decomposer.get_L());

    return Matrix<DefSparse, T, M, M,
                  Base::Matrix::CalculateTriangularSize<M, M>::value>(
        this->_L_triangular);
  }

  auto get_U()
      -> Matrix<DefSparse, T, M, M,
                Base::Matrix::CalculateTriangularSize<M, M>::value> const {

    Base::Matrix::TriangularSparse<T, M, M>::set_values_upper(
        this->_U_triangular, this->_LU_decomposer.get_U());

    return Matrix<DefSparse, T, M, M,
                  Base::Matrix::CalculateTriangularSize<M, M>::value>(
        this->_U_triangular);
  }

  T get_det() { return this->_LU_decomposer.get_determinant(); }

private:
  /* Variable */
  Base::Matrix::LUDecomposition<T, M> _LU_decomposer;
  Base::Matrix::SparseMatrix<T, M, M,
                             Base::Matrix::CalculateTriangularSize<M, M>::value>
      _L_triangular = Base::Matrix::TriangularSparse<T, M, M>::create_lower();
  Base::Matrix::SparseMatrix<T, M, M,
                             Base::Matrix::CalculateTriangularSize<M, M>::value>
      _U_triangular = Base::Matrix::TriangularSparse<T, M, M>::create_upper();

  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_LU);
};

/* make LinalgSolverLU */
template <typename T, std::size_t M, std::size_t V = 1>
auto make_LinalgSolverLU(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolverLU<T, M, V> {

  return LinalgSolverLU<T, M, V>(A);
}

template <typename T, std::size_t M, std::size_t V = 1>
auto make_LinalgSolverLU(const Matrix<DefDiag, T, M> &A)
    -> LinalgSolverLU<T, M, V> {

  return LinalgSolverLU<T, M, V>(A);
}

template <typename T, std::size_t M, std::size_t V>
auto make_LinalgSolverLU(const Matrix<DefSparse, T, M, M, V> &A)
    -> LinalgSolverLU<T, M, V> {

  return LinalgSolverLU<T, M, V>(A);
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_LINALG_LU_HPP
