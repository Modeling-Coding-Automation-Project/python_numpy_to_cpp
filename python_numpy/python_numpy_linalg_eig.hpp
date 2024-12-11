#ifndef PYTHON_NUMPY_LINALG_EIG_HPP
#define PYTHON_NUMPY_LINALG_EIG_HPP

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_EIG = 1.0e-20;
const std::size_t DEFAULT_ITERATION_MAX_LINALG_EIG = 10;

/* Able to handle only real number */
template <typename T, std::size_t M> class LinalgSolverEigReal {
public:
  /* Constructor */
  LinalgSolverEigReal() {}

  LinalgSolverEigReal(const Matrix<DefDense, T, M, M> &A) {
    this->_Eigen_solver = Base::Matrix::EigenSolverReal<T, M>(
        A.matrix, DEFAULT_ITERATION_MAX_LINALG_EIG, this->_division_min);
  }

  /* Copy Constructor */
  LinalgSolverEigReal(const LinalgSolverEigReal<T, M> &other)
      : _Eigen_solver(other._Eigen_solver), _division_min(other._division_min) {
  }

  LinalgSolverEigReal<T, M> &operator=(const LinalgSolverEigReal<T, M> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
      this->_division_min = other._division_min;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigReal(LinalgSolverEigReal<T, M> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)),
        _division_min(std::move(other._division_min)) {}

  LinalgSolverEigReal<T, M> &
  operator=(LinalgSolverEigReal<T, M> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
      this->_division_min = std::move(other._division_min);
    }
    return *this;
  }

  /* Solve function */
  inline void solve_eigen_values(const Matrix<DefDense, T, M, M> &A) {
    this->_Eigen_solver.solve_eigen_values(A.matrix);
  }

  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  inline void solve_eigen_vectors(const Matrix<DefDense, T, M, M> &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix);
  }

  /* Get */
  inline auto get_eigen_values(void) -> Matrix<DefDense, T, M, 1> {
    return Matrix<DefDense, T, M, 1>(
        Base::Matrix::Matrix<T, M, 1>(this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void) -> Matrix<DefDense, T, M, M> {
    return Matrix<DefDense, T, M, M>(this->_Eigen_solver.get_eigen_vectors());
  }

  inline std::size_t get_iteration_max(void) {
    return this->_Eigen_solver.iteration_max;
  }

  /* Set */
  inline void set_iteration_max(std::size_t iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

private:
  /* Variable */
  Base::Matrix::EigenSolverReal<T, M> _Eigen_solver;
  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_EIG);
};

template <typename T, std::size_t M> class LinalgSolverEigRealDiag {
public:
  /* Constructor */
  LinalgSolverEigRealDiag() {}

  LinalgSolverEigRealDiag(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  void solve_eigen_values(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  void solve_eigen_vectors(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  /* Copy Constructor */
  LinalgSolverEigRealDiag(const LinalgSolverEigRealDiag<T, M> &other)
      : _eigen_values(other._eigen_values) {}

  LinalgSolverEigRealDiag<T, M> &
  operator=(const LinalgSolverEigRealDiag<T, M> &other) {
    if (this != &other) {
      this->_eigen_values = other._eigen_values;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigRealDiag(LinalgSolverEigRealDiag<T, M> &&other) noexcept
      : _eigen_values(std::move(other._eigen_values)) {}

  LinalgSolverEigRealDiag<T, M> &
  operator=(LinalgSolverEigRealDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->_eigen_values = std::move(other._eigen_values);
    }
    return *this;
  }

  /* Get */
  inline auto get_eigen_values(void) -> Matrix<DefDense, T, M, 1> {
    return Matrix<DefDense, T, M, 1>(this->_eigen_values);
  }

  inline auto get_eigen_vectors(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(Base::Matrix::DiagMatrix<T, M>::identity());
  }

private:
  /* Variable */
  Base::Matrix::Matrix<T, M, 1> _eigen_values;
};

template <typename T, std::size_t M, typename SparseAvailable>
class LinalgSolverEigRealSparse {
public:
  /* Constructor */
  LinalgSolverEigRealSparse() {}

  LinalgSolverEigRealSparse(
      const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->solve_eigen_values(A);
  }

  /* Copy Constructor */
  LinalgSolverEigRealSparse(
      const LinalgSolverEigRealSparse<T, M, SparseAvailable> &other)
      : _Eigen_solver(other._Eigen_solver), _division_min(other._division_min),
        _iteration_max(other._iteration_max) {}

  LinalgSolverEigRealSparse<T, M, SparseAvailable> &
  operator=(const LinalgSolverEigRealSparse<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
      this->_division_min = other._division_min;
      this->_iteration_max = other._iteration_max;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigRealSparse(
      LinalgSolverEigRealSparse<T, M, SparseAvailable> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)),
        _division_min(std::move(other._division_min)),
        _iteration_max(std::move(other._iteration_max)) {}

  LinalgSolverEigRealSparse<T, M, SparseAvailable> &
  operator=(LinalgSolverEigRealSparse<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
      this->_division_min = std::move(other._division_min);
      this->_iteration_max = std::move(other._iteration_max);
    }
    return *this;
  }

  /* Solve method */
  inline void
  solve_eigen_values(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->_Eigen_solver = Base::Matrix::EigenSolverReal<T, M>(
        A.matrix.create_dense(), this->_iteration_max, this->_division_min);
  }

  inline void
  solve_eigen_vectors(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix.create_dense());
  }

  /* Get */
  inline auto get_eigen_values(void) -> Matrix<DefDense, T, M, 1> {
    return Matrix<DefDense, T, M, 1>(
        Base::Matrix::Matrix<T, M, 1>(this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void) -> Matrix<DefDense, T, M, M> {
    return Matrix<DefDense, T, M, M>(this->_Eigen_solver.get_eigen_vectors());
  }

private:
  /* Variable */
  Base::Matrix::EigenSolverReal<T, M> _Eigen_solver;
  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_EIG);
  std::size_t _iteration_max =
      static_cast<std::size_t>(DEFAULT_ITERATION_MAX_LINALG_EIG);
};

/* make LinalgSolverEig */
template <typename T, std::size_t M>
inline auto make_LinalgSolverEigReal(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolverEigReal<T, M> {

  return LinalgSolverEigReal<T, M>(A);
}

template <typename T, std::size_t M>
inline auto make_LinalgSolverEigReal(const Matrix<DefDiag, T, M> &A)
    -> LinalgSolverEigRealDiag<T, M> {

  return LinalgSolverEigRealDiag<T, M>(A);
}

template <typename T, std::size_t M, typename SparseAvailable>
inline auto
make_LinalgSolverEigReal(const Matrix<DefSparse, T, M, M, SparseAvailable> &A)
    -> LinalgSolverEigRealSparse<T, M, SparseAvailable> {

  return LinalgSolverEigRealSparse<T, M, SparseAvailable>(A);
}

/* Able to handle complex number */
template <typename T, std::size_t M> class LinalgSolverEig {
public:
  /* Constructor */
  LinalgSolverEig() {}

  LinalgSolverEig(const Matrix<DefDense, T, M, M> &A) {
    this->solve_eigen_values(A);
  }

  /* Copy Constructor */
  LinalgSolverEig(const LinalgSolverEig<T, M> &other)
      : _Eigen_solver(other._Eigen_solver), _division_min(other._division_min),
        _iteration_max(other._iteration_max) {}

  LinalgSolverEig<T, M> &operator=(const LinalgSolverEig<T, M> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
      this->_division_min = other._division_min;
      this->_iteration_max = other._iteration_max;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEig(LinalgSolverEig<T, M> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)),
        _division_min(std::move(other._division_min)),
        _iteration_max(std::move(other._iteration_max)) {}

  LinalgSolverEig<T, M> &operator=(LinalgSolverEig<T, M> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
      this->_division_min = std::move(other._division_min);
      this->_iteration_max = std::move(other._iteration_max);
    }
    return *this;
  }

  /* Solve method */
  inline void solve_eigen_values(const Matrix<DefDense, T, M, M> &A) {
    this->_Eigen_solver = Base::Matrix::EigenSolverComplex<T, M>(
        A.matrix, this->_iteration_max, this->_division_min);
  }

  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  inline void solve_eigen_vectors(const Matrix<DefDense, T, M, M> &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix);
  }

  /* Get */
  inline auto get_eigen_values(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, 1> {
    return Matrix<DefDense, Base::Matrix::Complex<T>, M, 1>(
        Base::Matrix::Matrix<Base::Matrix::Complex<T>, M, 1>(
            this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, M> {
    return Matrix<DefDense, Base::Matrix::Complex<T>, M, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

  inline std::size_t get_iteration_max(void) {
    return this->_Eigen_solver.iteration_max;
  }

  /* Set */
  inline void set_iteration_max(std::size_t iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

  inline void set_iteration_max_for_eigen_vector(
      std::size_t iteration_max_for_eigen_vector) {
    this->_Eigen_solver.iteration_max_for_eigen_vector =
        iteration_max_for_eigen_vector;
  }

private:
  /* Variable */
  Base::Matrix::EigenSolverComplex<T, M> _Eigen_solver;
  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_EIG);
  std::size_t _iteration_max =
      static_cast<std::size_t>(DEFAULT_ITERATION_MAX_LINALG_EIG);
};

template <typename T, std::size_t M> class LinalgSolverEigDiag {
public:
  /* Constructor */
  LinalgSolverEigDiag() {}

  LinalgSolverEigDiag(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  /* Copy Constructor */
  LinalgSolverEigDiag(const LinalgSolverEigDiag<T, M> &other)
      : _eigen_values(other._eigen_values) {}

  LinalgSolverEigDiag<T, M> &operator=(const LinalgSolverEigDiag<T, M> &other) {
    if (this != &other) {
      this->_eigen_values = other._eigen_values;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigDiag(LinalgSolverEigDiag<T, M> &&other) noexcept
      : _eigen_values(std::move(other._eigen_values)) {}

  LinalgSolverEigDiag<T, M> &
  operator=(LinalgSolverEigDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->_eigen_values = std::move(other._eigen_values);
    }
    return *this;
  }

  /* Solve method */
  inline void solve_eigen_values(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  inline void solve_eigen_vectors(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  /* Get */
  inline auto get_eigen_values(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, 1> {
    return Matrix<DefDense, Base::Matrix::Complex<T>, M, 1>(
        this->_eigen_values);
  }

  inline auto get_eigen_vectors(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, M> {
    return Matrix<DefDense, Base::Matrix::Complex<T>, M, M>(
        Base::Matrix::DiagMatrix<T, M>::identity());
  }

private:
  /* Variable */
  Base::Matrix::Matrix<Base::Matrix::Complex<T>, M, 1> _eigen_values;
};

template <typename T, std::size_t M, typename SparseAvailable>
class LinalgSolverEigSparse {
public:
  /* Constructor */
  LinalgSolverEigSparse() {}

  LinalgSolverEigSparse(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->solve_eigen_values(A);
  }

  /* Copy Constructor */
  LinalgSolverEigSparse(
      const LinalgSolverEigSparse<T, M, SparseAvailable> &other)
      : _Eigen_solver(other._Eigen_solver), _division_min(other._division_min),
        _iteration_max(other._iteration_max) {}

  LinalgSolverEigSparse<T, M, SparseAvailable> &
  operator=(const LinalgSolverEigSparse<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
      this->_division_min = other._division_min;
      this->_iteration_max = other._iteration_max;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigSparse(
      LinalgSolverEigSparse<T, M, SparseAvailable> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)),
        _division_min(std::move(other._division_min)),
        _iteration_max(std::move(other._iteration_max)) {}

  LinalgSolverEigSparse<T, M, SparseAvailable> &
  operator=(LinalgSolverEigSparse<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
      this->_division_min = std::move(other._division_min);
      this->_iteration_max = std::move(other._iteration_max);
    }
    return *this;
  }

  /* Solve method */
  inline void
  solve_eigen_values(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->_Eigen_solver = Base::Matrix::EigenSolverComplex<T, M>(
        A.matrix.create_dense(), this->_iteration_max, this->_division_min);
  }

  inline void
  solve_eigen_vectors(const Matrix<DefSparse, T, M, M, SparseAvailable> &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix.create_dense());
  }

  /* Get */
  inline auto get_eigen_values(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, 1> {
    return Matrix<DefDense, Base::Matrix::Complex<T>, M, 1>(
        Base::Matrix::Matrix<Base::Matrix::Complex<T>, M, 1>(
            this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, M> {
    return Matrix<DefDense, Base::Matrix::Complex<T>, M, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

private:
  /* Variable */
  Base::Matrix::EigenSolverComplex<T, M> _Eigen_solver;
  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_EIG);
  std::size_t _iteration_max =
      static_cast<std::size_t>(DEFAULT_ITERATION_MAX_LINALG_EIG);
};

/* make LinalgSolverEig */
template <typename T, std::size_t M>
inline auto make_LinalgSolverEig(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolverEig<T, M> {

  return LinalgSolverEig<T, M>(A);
}

template <typename T, std::size_t M>
inline auto make_LinalgSolverEig(const Matrix<DefDiag, T, M> &A)
    -> LinalgSolverEigDiag<T, M> {

  return LinalgSolverEigDiag<T, M>(A);
}

template <typename T, std::size_t M, typename SparseAvailable>
inline auto
make_LinalgSolverEig(const Matrix<DefSparse, T, M, M, SparseAvailable> &A)
    -> LinalgSolverEigSparse<T, M, SparseAvailable> {

  return LinalgSolverEigSparse<T, M, SparseAvailable>(A);
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_LINALG_EIG_HPP
