#ifndef __PYTHON_NUMPY_LINALG_EIG_HPP__
#define __PYTHON_NUMPY_LINALG_EIG_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_EIG = 1.0e-20;
const std::size_t DEFAULT_ITERATION_MAX_LINALG_EIG = 10;

namespace ForLinalgSolverEigReal {

template <typename T, std::size_t M>
using EigenValues_Type = Matrix<DefDense, T, M, 1>;

template <typename T, std::size_t M>
using EigenVectors_Type = Matrix<DefDense, T, M, M>;

} // namespace ForLinalgSolverEigReal

/* Linalg solver for Real Eigen values and vectors of Dense Matrix */
template <typename T, std::size_t M> class LinalgSolverEigRealDense {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDense, T, M, M>;
  using EigenSolver_Type = Base::Matrix::EigenSolverReal<T, M>;

public:
  /* Constructor */
  LinalgSolverEigRealDense() {}

  /* Copy Constructor */
  LinalgSolverEigRealDense(const LinalgSolverEigRealDense<T, M> &other)
      : _Eigen_solver(other._Eigen_solver) {}

  LinalgSolverEigRealDense<T, M> &
  operator=(const LinalgSolverEigRealDense<T, M> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigRealDense(LinalgSolverEigRealDense<T, M> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)) {}

  LinalgSolverEigRealDense<T, M> &
  operator=(LinalgSolverEigRealDense<T, M> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
    }
    return *this;
  }

  /* Solve function */
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(A.matrix);
  }

  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix);
  }

  /* Get */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEigReal::EigenValues_Type<T, M> {
    return ForLinalgSolverEigReal::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<T, M, 1>(this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEigReal::EigenVectors_Type<T, M> {
    return ForLinalgSolverEigReal::EigenVectors_Type<T, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

  inline std::size_t get_iteration_max(void) {
    return this->_Eigen_solver.iteration_max;
  }

  /* Set */
  inline void set_iteration_max(std::size_t iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

  inline void set_division_min(const T &division_min_in) {
    this->_Eigen_solver.division_min = division_min_in;
  }

  inline void set_small_value(const T &small_value_in) {
    this->_Eigen_solver.small_value = small_value_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  EigenSolver_Type _Eigen_solver;
};

/* Linalg solver for Real Eigen values and vectors of Diag Matrix */
template <typename T, std::size_t M> class LinalgSolverEigRealDiag {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDiag, T, M>;
  using EigenValues_Type = Matrix<DefDense, T, M, 1>;
  using EigenVectors_Type = Matrix<DefDiag, T, M>;

public:
  /* Constructor */
  LinalgSolverEigRealDiag() {}

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

public:
  /* Function */
  inline void solve_eigen_values(const A_Type &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  inline void solve_eigen_vectors(const A_Type &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  /* Get */
  inline auto get_eigen_values(void) -> EigenValues_Type {
    return EigenValues_Type(this->_eigen_values);
  }

  inline auto get_eigen_vectors(void) -> EigenVectors_Type {
    return EigenVectors_Type(Base::Matrix::DiagMatrix<T, M>::identity());
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  Base::Matrix::Matrix<T, M, 1> _eigen_values;
};

/* Linalg solver for Real Eigen values and vectors of Sparse Matrix */
template <typename T, std::size_t M, typename SparseAvailable>
class LinalgSolverEigRealSparse {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefSparse, T, M, M, SparseAvailable>;
  using EigenSolver_Type = Base::Matrix::EigenSolverReal<T, M>;

public:
  /* Constructor */
  LinalgSolverEigRealSparse() {}

  /* Copy Constructor */
  LinalgSolverEigRealSparse(
      const LinalgSolverEigRealSparse<T, M, SparseAvailable> &other)
      : _Eigen_solver(other._Eigen_solver) {}

  LinalgSolverEigRealSparse<T, M, SparseAvailable> &
  operator=(const LinalgSolverEigRealSparse<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigRealSparse(
      LinalgSolverEigRealSparse<T, M, SparseAvailable> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)) {}

  LinalgSolverEigRealSparse<T, M, SparseAvailable> &
  operator=(LinalgSolverEigRealSparse<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
    }
    return *this;
  }

  /* Solve method */
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  /* Get */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEigReal::EigenValues_Type<T, M> {
    return ForLinalgSolverEigReal::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<T, M, 1>(this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEigReal::EigenVectors_Type<T, M> {
    return ForLinalgSolverEigReal::EigenVectors_Type<T, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

  /* Set */
  inline void set_iteration_max(std::size_t iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

  inline void set_division_min(const T &division_min_in) {
    this->_Eigen_solver.division_min = division_min_in;
  }

  inline void set_small_value(const T &small_value_in) {
    this->_Eigen_solver.small_value = small_value_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  Base::Matrix::EigenSolverReal<T, M> _Eigen_solver;
};

/* make LinalgSolverEig Real */
template <
    typename A_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEigReal()
    -> LinalgSolverEigRealDense<typename A_Type::Value_Complex_Type,
                                A_Type::COLS> {

  return LinalgSolverEigRealDense<typename A_Type::Value_Complex_Type,
                                  A_Type::COLS>();
}

template <typename A_Type, typename std::enable_if<
                               Is_Diag_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEigReal()
    -> LinalgSolverEigRealDiag<typename A_Type::Value_Complex_Type,
                               A_Type::COLS> {

  return LinalgSolverEigRealDiag<typename A_Type::Value_Complex_Type,
                                 A_Type::COLS>();
}

template <
    typename A_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEigReal()
    -> LinalgSolverEigRealSparse<typename A_Type::Value_Complex_Type,
                                 A_Type::COLS,
                                 typename A_Type::SparseAvailable_Type> {

  return LinalgSolverEigRealSparse<typename A_Type::Value_Complex_Type,
                                   A_Type::COLS,
                                   typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolverEig Real Type */
template <typename A_Type>
using LinalgSolverEigReal_Type = decltype(make_LinalgSolverEigReal<A_Type>());

namespace ForLinalgSolverEig {

template <typename T, std::size_t M>
using EigenValues_Type = Matrix<DefDense, Base::Matrix::Complex<T>, M, 1>;

template <typename T, std::size_t M>
using EigenVectors_Type = Matrix<DefDense, Base::Matrix::Complex<T>, M, M>;

} // namespace ForLinalgSolverEig

/* Linalg solver for Complex Eigen values and vectors of Dense and Sparse Matrix
 */
template <typename T, std::size_t M,
          std::size_t Default_Iteration_Max =
              PythonNumpy::DEFAULT_ITERATION_MAX_LINALG_EIG>
class LinalgSolverEigDense {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDense, T, M, M>;
  using EigenSolver_Type = Base::Matrix::EigenSolverComplex<T, M>;

public:
  /* Constructor */
  LinalgSolverEigDense() {}

  LinalgSolverEigDense(const A_Type &A) {
    this->_Eigen_solver = Base::Matrix::EigenSolverComplex<T, M>(
        A.matrix, Default_Iteration_Max, this->_division_min);
  }

  /* Copy Constructor */
  LinalgSolverEigDense(const LinalgSolverEigDense<T, M> &other)
      : _Eigen_solver(other._Eigen_solver), _division_min(other._division_min),
        _iteration_max(other._iteration_max) {}

  LinalgSolverEigDense<T, M> &
  operator=(const LinalgSolverEigDense<T, M> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
      this->_division_min = other._division_min;
      this->_iteration_max = other._iteration_max;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigDense(LinalgSolverEigDense<T, M> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)),
        _division_min(std::move(other._division_min)),
        _iteration_max(std::move(other._iteration_max)) {}

  LinalgSolverEigDense<T, M> &
  operator=(LinalgSolverEigDense<T, M> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
      this->_division_min = std::move(other._division_min);
      this->_iteration_max = std::move(other._iteration_max);
    }
    return *this;
  }

  /* Solve method */
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(A.matrix);
  }

  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix);
  }

  /* Get */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEig::EigenValues_Type<T, M> {
    return ForLinalgSolverEig::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<Base::Matrix::Complex<T>, M, 1>(
            this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEig::EigenVectors_Type<T, M> {
    return ForLinalgSolverEig::EigenVectors_Type<T, M>(
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

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  EigenSolver_Type _Eigen_solver;
  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_EIG);
  std::size_t _iteration_max = static_cast<std::size_t>(Default_Iteration_Max);
};

/* Linalg solver for Complex Eigen values and vectors of Diag Matrix */
template <typename T, std::size_t M,
          std::size_t Default_Iteration_Max =
              PythonNumpy::DEFAULT_ITERATION_MAX_LINALG_EIG>
class LinalgSolverEigDiag {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDiag, T, M>;
  using EigenValues_Type = Matrix<DefDense, Base::Matrix::Complex<T>, M, 1>;
  using EigenVectors_Type = Matrix<DefDiag, Base::Matrix::Complex<T>, M>;

public:
  /* Constructor */
  LinalgSolverEigDiag() {}

  LinalgSolverEigDiag(const A_Type &A) {
    this->_eigen_values =
        EigenValues_Type(Base::Matrix::convert_matrix_real_to_complex(
            Base::Matrix::Matrix<T, M, 1>(A.matrix.data)));
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
    this->_eigen_values =
        EigenValues_Type(Base::Matrix::convert_matrix_real_to_complex(
            Base::Matrix::Matrix<T, M, 1>(A.matrix.data)));
  }

  inline void solve_eigen_vectors(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values =
        EigenValues_Type(Base::Matrix::convert_matrix_real_to_complex(
            Base::Matrix::Matrix<T, M, 1>(A.matrix.data)));
  }

  /* Get */
  inline auto get_eigen_values(void) -> EigenValues_Type {
    return EigenValues_Type(this->_eigen_values);
  }

  inline auto get_eigen_vectors(void) -> EigenVectors_Type {
    return EigenVectors_Type(Base::Matrix::convert_matrix_real_to_complex(
        Base::Matrix::DiagMatrix<T, M>::identity()));
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  EigenValues_Type _eigen_values;
};

/* Linalg solver for Complex Eigen values and vectors of Sparse Matrix */
template <typename T, std::size_t M, typename SparseAvailable,
          std::size_t Default_Iteration_Max =
              PythonNumpy::DEFAULT_ITERATION_MAX_LINALG_EIG>
class LinalgSolverEigSparse {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefSparse, T, M, M, SparseAvailable>;
  using EigenSolver_Type = Base::Matrix::EigenSolverComplex<T, M>;

public:
  /* Constructor */
  LinalgSolverEigSparse() {}

  LinalgSolverEigSparse(const A_Type &A) {
    this->_Eigen_solver =
        EigenSolver_Type(Base::Matrix::output_dense_matrix(A.matrix),
                         Default_Iteration_Max, this->_division_min);
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
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  /* Get */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEig::EigenValues_Type<T, M> {
    return ForLinalgSolverEig::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<Base::Matrix::Complex<T>, M, 1>(
            this->_Eigen_solver.get_eigen_values()));
  }

  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEig::EigenVectors_Type<T, M> {
    return ForLinalgSolverEig::EigenVectors_Type<T, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

private:
  /* Variable */
  EigenSolver_Type _Eigen_solver;
  T _division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_EIG);
  std::size_t _iteration_max = Default_Iteration_Max;
};

/* make LinalgSolverEig Complex */
template <typename T, std::size_t M,
          std::size_t Default_Iteration_Max =
              PythonNumpy::DEFAULT_ITERATION_MAX_LINALG_EIG>
inline auto make_LinalgSolverEig(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolverEigDense<T, M, Default_Iteration_Max> {

  return LinalgSolverEigDense<T, M, Default_Iteration_Max>(A);
}

template <typename T, std::size_t M,
          std::size_t Default_Iteration_Max =
              PythonNumpy::DEFAULT_ITERATION_MAX_LINALG_EIG>
inline auto make_LinalgSolverEig(const Matrix<DefDiag, T, M> &A)
    -> LinalgSolverEigDiag<T, M, Default_Iteration_Max> {

  return LinalgSolverEigDiag<T, M, Default_Iteration_Max>(A);
}

template <typename T, std::size_t M, typename SparseAvailable,
          std::size_t Default_Iteration_Max =
              PythonNumpy::DEFAULT_ITERATION_MAX_LINALG_EIG>
inline auto
make_LinalgSolverEig(const Matrix<DefSparse, T, M, M, SparseAvailable> &A)
    -> LinalgSolverEigSparse<T, M, SparseAvailable, Default_Iteration_Max> {

  return LinalgSolverEigSparse<T, M, SparseAvailable, Default_Iteration_Max>(A);
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_EIG_HPP__
