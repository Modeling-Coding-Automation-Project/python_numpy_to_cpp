/**
 * @file base_matrix_eigen_solver.hpp
 * @brief Eigenvalue and Eigenvector Solver for Real and Complex Matrices
 *
 * This header provides template classes for computing eigenvalues and
 * eigenvectors of square matrices, supporting both real and complex types. The
 * solvers use the QR algorithm (with optional Hessenberg reduction and
 * Wilkinson shift) for eigenvalue computation, and the inverse iteration method
 * for eigenvector computation. The implementation is compatible with both
 * std::vector and std::array storage, controlled by the
 * __BASE_MATRIX_USE_STD_VECTOR__ macro.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_EIGEN_SOLVER_HPP__
#define __BASE_MATRIX_EIGEN_SOLVER_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_complex.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_inverse.hpp"
#include "base_matrix_lu_decomposition.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_variable_sparse.hpp"
#include "base_matrix_vector.hpp"
#include "base_utility.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace Base {
namespace Matrix {

const std::size_t DEFAULT_ITERATION_MAX_EIGEN_SOLVER = 10;
const double DEFAULT_DIVISION_MIN_EIGEN_SOLVER = 1.0e-20;
const double EIGEN_SMALL_VALUE = 1.0e-6;

/**
 * @brief EigenSolverReal is a template class for computing eigenvalues and
 * eigenvectors of real square matrices.
 *
 * This class provides methods to solve eigenvalues and eigenvectors using the
 * QR algorithm, with optional Hessenberg reduction and Wilkinson shift. It
 * supports both std::vector and std::array storage for eigenvalues.
 *
 * @tparam T Type of the matrix elements (e.g., float, double).
 * @tparam M Size of the square matrix (number of rows and columns).
 */
template <typename T, std::size_t M> class EigenSolverReal {
public:
  /* Constructor */

#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  EigenSolverReal()
      : iteration_max(DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        small_value(static_cast<T>(EIGEN_SMALL_VALUE)),
        gmres_k_decay_rate(static_cast<T>(0)), _House(), _Hessen(),
        _eigen_values(M, static_cast<T>(0)),
        _eigen_vectors(Matrix<T, M, M>::ones()),
        _gmres_k_rho(static_cast<std::size_t>(0)),
        _gmres_k_rep_num(static_cast<std::size_t>(0)) {}

#else // __BASE_MATRIX_USE_STD_VECTOR__

  EigenSolverReal()
      : iteration_max(DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        small_value(static_cast<T>(EIGEN_SMALL_VALUE)),
        gmres_k_decay_rate(static_cast<T>(0)), _House(), _Hessen(),
        _eigen_values(), _eigen_vectors(Matrix<T, M, M>::ones()),
        _gmres_k_rho(static_cast<std::size_t>(0)),
        _gmres_k_rep_num(static_cast<std::size_t>(0)) {}

#endif // __BASE_MATRIX_USE_STD_VECTOR__

  /* Copy Constructor */
  EigenSolverReal(const EigenSolverReal<T, M> &other)
      : iteration_max(other.iteration_max), division_min(other.division_min),
        small_value(other.small_value),
        gmres_k_decay_rate(other.gmres_k_decay_rate), _House(other._House),
        _Hessen(other._Hessen), _eigen_values(other._eigen_values),
        _eigen_vectors(other._eigen_vectors), _gmres_k_rho(other._gmres_k_rho),
        _gmres_k_rep_num(other._gmres_k_rep_num) {}

  EigenSolverReal &operator=(const EigenSolverReal<T, M> &other) {
    if (this != &other) {
      this->iteration_max = other.iteration_max;
      this->division_min = other.division_min;
      this->small_value = other.small_value;
      this->gmres_k_decay_rate = other.gmres_k_decay_rate;
      this->_House = other._House;
      this->_Hessen = other._Hessen;
      this->_eigen_values = other._eigen_values;
      this->_eigen_vectors = other._eigen_vectors;
      this->_gmres_k_rho = other._gmres_k_rho;
      this->_gmres_k_rep_num = other._gmres_k_rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  EigenSolverReal(EigenSolverReal<T, M> &&other) noexcept
      : iteration_max(std::move(other.iteration_max)),
        division_min(std::move(other.division_min)),
        small_value(std::move(other.small_value)),
        gmres_k_decay_rate(std::move(other.gmres_k_decay_rate)),
        _House(std::move(other._House)), _Hessen(std::move(other._Hessen)),
        _eigen_values(std::move(other._eigen_values)),
        _eigen_vectors(std::move(other._eigen_vectors)),
        _gmres_k_rho(std::move(other._gmres_k_rho)),
        _gmres_k_rep_num(std::move(other._gmres_k_rep_num)) {}

  EigenSolverReal &operator=(EigenSolverReal<T, M> &&other) noexcept {
    if (this != &other) {
      this->iteration_max = std::move(other.iteration_max);
      this->division_min = std::move(other.division_min);
      this->small_value = std::move(other.small_value);
      this->gmres_k_decay_rate = std::move(other.gmres_k_decay_rate);
      this->_House = std::move(other._House);
      this->_Hessen = std::move(other._Hessen);
      this->_eigen_values = std::move(other._eigen_values);
      this->_eigen_vectors = std::move(other._eigen_vectors);
      this->_gmres_k_rho = std::move(other._gmres_k_rho);
      this->_gmres_k_rep_num = std::move(other._gmres_k_rep_num);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Computes the eigenvalues of a square matrix using the QR method.
   *
   * This function takes a square matrix as input and computes its eigenvalues
   * by internally invoking the QR algorithm. The results are stored within the
   * class instance.
   *
   * @tparam T The data type of the matrix elements.
   * @tparam M The dimension of the square matrix.
   * @param matrix The input square matrix whose eigenvalues are to be computed.
   */
  inline void solve_eigen_values(const Matrix<T, M, M> &matrix) {
    this->_solve_values_with_qr_method(matrix);
  }

  /**
   * @brief Continues solving eigenvalues using the QR method.
   *
   * This function continues the process of computing eigenvalues using the QR
   * method, which may have been started previously. It is useful for iterative
   * refinement of eigenvalue calculations.
   */
  inline void continue_solving_eigen_values(void) {
    this->_continue_solving_values_with_qr_method();
  }

  /**
   * @brief Computes the eigenvectors of a square matrix using the inverse
   * iteration method.
   *
   * This function takes a square matrix as input and computes its eigenvectors
   * by internally invoking the inverse iteration method. The results are stored
   * within the class instance.
   *
   * @tparam T The data type of the matrix elements.
   * @tparam M The dimension of the square matrix.
   * @param matrix The input square matrix whose eigenvectors are to be
   * computed.
   */
  inline void solve_eigen_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

  /**
   * @brief Computes both eigenvalues and eigenvectors of a square matrix.
   *
   * This function computes the eigenvalues using the QR method and the
   * eigenvectors using the inverse iteration method, storing the results
   * within the class instance.
   *
   * @tparam T The data type of the matrix elements.
   * @tparam M The dimension of the square matrix.
   * @param matrix The input square matrix whose eigenvalues and eigenvectors
   * are to be computed.
   */
  inline void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_values_with_qr_method(matrix);
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  /**
   * @brief Retrieves the computed eigenvalues.
   * This function returns the eigenvalues computed by the QR method as a
   * vector.
   * @return std::vector<T> A vector containing the eigenvalues.
   * */
  inline std::vector<T> get_eigen_values(void) { return this->_eigen_values; }
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  /**
   * @brief Retrieves the computed eigenvalues.
   * This function returns the eigenvalues computed by the QR method as an
   * array.
   * @return std::array<T, M> An array containing the eigenvalues.
   * */
  inline std::array<T, M> get_eigen_values(void) { return this->_eigen_values; }
#endif // __BASE_MATRIX_USE_STD_VECTOR__

  /**
   * @brief Retrieves the computed eigenvectors.
   * This function returns the eigenvectors computed by the inverse iteration
   * method as a matrix.
   * @return Matrix<T, M, M> A matrix containing the eigenvectors.
   */
  inline Matrix<T, M, M> get_eigen_vectors(void) {
    return this->_eigen_vectors;
  }

  /**
   * @brief Retrieves the Hessenberg form of the matrix.
   * This function returns the Hessenberg form of the matrix computed during the
   * QR method.
   * @return Matrix<T, M, M> A matrix in Hessenberg form.
   */
  inline void set_iteration_max(const std::size_t &iteration_max_in) {
    this->iteration_max = iteration_max_in;
  }

  /**
   * @brief Sets the minimum division value for numerical stability.
   * This function sets the minimum value used to avoid division by zero in
   * numerical computations.
   * @param division_min_in The minimum division value.
   */
  inline void set_division_min(const T &division_min_in) {
    this->division_min = division_min_in;
  }

  /**
   * @brief Sets a small value for numerical stability.
   * This function sets a small value used to determine convergence in the QR
   * method.
   * @param small_value_in The small value for numerical stability.
   */
  inline void set_small_value(const T &small_value_in) {
    this->small_value = small_value_in;
  }

  /**
   * @brief Sets the decay rate for GMRES k.
   * This function sets the decay rate for the GMRES k method, which is used in
   * iterative methods for solving linear systems.
   * @param gmres_k_decay_rate_in The decay rate for GMRES k.
   */
  inline void set_gmres_k_decay_rate(const T &gmres_k_decay_rate_in) {
    this->gmres_k_decay_rate = gmres_k_decay_rate_in;
  }

  /**
   * @brief Retrieves the Hessenberg form of the matrix.
   * This function returns the Hessenberg form of the matrix computed during the
   * QR method.
   * @return Matrix<T, M, M> A matrix in Hessenberg form.
   */
  inline Matrix<T, M, M> check_validity(const Matrix<T, M, M> &matrix) {
    Matrix<T, M, M> result =
        matrix * this->_eigen_vectors -
        this->_eigen_vectors *
            Base::Matrix::DiagMatrix<T, M>(this->_eigen_values);

    return result;
  }

public:
  /* Variable */
  std::size_t iteration_max;
  T division_min;
  T small_value;
  T gmres_k_decay_rate;

protected:
  /* Variable */
  VariableSparseMatrix<T, M, M> _House;
  Matrix<T, M, M> _Hessen;
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> _eigen_values;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, M> _eigen_values;
#endif // __BASE_MATRIX_USE_STD_VECTOR__

  Matrix<T, M, M> _eigen_vectors;

  T _gmres_k_rho = static_cast<T>(0);
  std::size_t _gmres_k_rep_num = static_cast<std::size_t>(0);

protected:
  /* Function */

  /**
   * @brief Solves eigenvalues using the QR method.
   * This function computes the eigenvalues of a square matrix using the QR
   * algorithm, with optional Hessenberg reduction and Wilkinson shift.
   * @param A The input square matrix whose eigenvalues are to be computed.
   */
  inline void _hessenberg(const Matrix<T, M, M> &A) {
    Matrix<T, M, M> R = A;
    std::array<T, M> u;

    for (std::size_t k = 0; k < M - 2; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k + 1; i < M; ++i) {
        x_abs += R(i, k) * R(i, k);
      }
      if (Base::Utility::near_zero(x_abs, this->division_min)) {
        continue;
      }
      x_abs = Base::Math::sqrt<T>(x_abs, this->division_min);

      u[k + 1] = R(k + 1, k) + Base::Utility::sign(R(k + 1, k)) * x_abs;
      T u_abs = u[k + 1] * u[k + 1];
      for (std::size_t i = k + 2; i < M; ++i) {
        u[i] = R(i, k);
        u_abs += u[i] * u[i];
      }

      std::fill(this->_House.values.begin(), this->_House.values.end(),
                static_cast<T>(0));
      std::fill(this->_House.row_indices.begin(),
                this->_House.row_indices.end(), static_cast<std::size_t>(0));
      std::fill(this->_House.row_pointers.begin(),
                this->_House.row_pointers.end(), static_cast<std::size_t>(0));

      std::size_t H_value_count = 0;
      for (std::size_t i = 0; i < k + 1; i++) {
        this->_House.values[H_value_count] = static_cast<T>(1);
        this->_House.row_indices[H_value_count] = i;
        H_value_count++;

        this->_House.row_pointers[i + 1] = H_value_count;
      }

      for (std::size_t i = k + 1; i < M; ++i) {
        for (std::size_t j = k + 1; j < M; ++j) {

          if (i == j) {
            this->_House.values[H_value_count] = static_cast<T>(1);
          }

          this->_House.values[H_value_count] -=
              static_cast<T>(2) * u[i] * u[j] /
              Base::Utility::avoid_zero_divide(u_abs, this->division_min);

          this->_House.row_indices[H_value_count] = j;
          H_value_count++;
        }
        this->_House.row_pointers[i + 1] = H_value_count;
      }

      R = this->_House * R * this->_House;
    }

    this->_Hessen = R;
  }

  /**
   * @brief Solves eigenvalues using the QR method.
   * This function computes the eigenvalues of a square matrix using the QR
   * algorithm, with optional Hessenberg reduction and Wilkinson shift.
   * @param A The input square matrix whose eigenvalues are to be computed.
   */
  inline void _qr_decomposition(Matrix<T, M, M> &Q, Matrix<T, M, M> &R,
                                const Matrix<T, M, M> &A) {
    R = A;
    std::array<T, M> u;

    for (std::size_t k = 0; k < M - 1; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k; i < k + 2; ++i) {
        x_abs += R(i, k) * R(i, k);
      }
      if (Base::Utility::near_zero(x_abs, this->division_min)) {
        continue;
      }
      x_abs = Base::Math::sqrt<T>(x_abs, this->division_min);

      u[k] = R(k, k) + Base::Utility::sign(R(k, k)) * x_abs;
      u[k + 1] = R(k + 1, k);
      T u_abs = u[k] * u[k] + u[k + 1] * u[k + 1];

      std::fill(this->_House.values.begin(), this->_House.values.end(),
                static_cast<T>(0));
      std::fill(this->_House.row_indices.begin(),
                this->_House.row_indices.end(), static_cast<std::size_t>(0));
      std::fill(this->_House.row_pointers.begin(),
                this->_House.row_pointers.end(), static_cast<std::size_t>(0));

      std::size_t H_value_count = 0;
      for (std::size_t i = 0; i < k; i++) {
        this->_House.values[H_value_count] = static_cast<T>(1);
        this->_House.row_indices[H_value_count] = i;
        H_value_count++;

        this->_House.row_pointers[i + 1] = H_value_count;
      }

      for (std::size_t i = k; i < k + 2; ++i) {
        for (std::size_t j = k; j < k + 2; ++j) {

          if (i == j) {
            this->_House.values[H_value_count] = static_cast<T>(1);
          }

          this->_House.values[H_value_count] -=
              static_cast<T>(2) * u[i] * u[j] /
              Base::Utility::avoid_zero_divide(u_abs, this->division_min);

          this->_House.row_indices[H_value_count] = j;
          H_value_count++;
        }
        this->_House.row_pointers[i + 1] = H_value_count;
      }

      for (std::size_t i = k + 2; i < M; i++) {
        this->_House.values[H_value_count] = static_cast<T>(1);
        this->_House.row_indices[H_value_count] = i;
        H_value_count++;

        this->_House.row_pointers[i + 1] = H_value_count;
      }

      R = this->_House * R;
      Q = Q * this->_House;
    }
  }

  /**
   * @brief Computes the Wilkinson shift for the QR method.
   * This function computes the Wilkinson shift, which is used to improve the
   * convergence of the QR algorithm for eigenvalue computation.
   * @param A The input square matrix from which the Wilkinson shift is
   * computed.
   * @return T The computed Wilkinson shift value.
   */
  inline T _wilkinson_shift(const Matrix<T, M, M> &A) {
    T a11 = A(M - 2, M - 2);
    T a12 = A(M - 2, M - 1);
    T a21 = A(M - 1, M - 2);
    T a22 = A(M - 1, M - 1);
    T c1 = a11 + a22;

    T c2_2 = (a11 - a22) * (a11 - a22) + static_cast<T>(4) * a12 * a21;
    T c2;
    if (c2_2 >= 0) {
      c2 = Base::Math::sqrt<T>(c2_2, this->division_min);
    } else {
      c2 = static_cast<T>(0);
    }

    T mu1 = static_cast<T>(0.5) * (c1 + c2);
    T mu2 = static_cast<T>(0.5) * (c1 - c2);
    T dmu1 = Base::Math::abs(a22 - mu1);
    T dmu2 = Base::Math::abs(a22 - mu2);
    return (dmu1 <= dmu2) ? mu1 : mu2;
  }

  /**
   * @brief Continues solving eigenvalues using the QR method.
   * This function continues the process of computing eigenvalues using the QR
   * method, which may have been started previously. It is useful for iterative
   * refinement of eigenvalue calculations.
   */
  inline void _continue_solving_values_with_qr_method(void) {
    for (std::size_t k = M; k > 1; --k) {
      Matrix<T, M, M> A = this->_Hessen;

      for (std::size_t iter = 0; iter < this->iteration_max; ++iter) {
        T mu = this->_wilkinson_shift(A);
        for (std::size_t i = 0; i < k; ++i) {
          A(i, i) -= mu;
        }

        Matrix<T, M, M> Q = Matrix<T, M, M>::identity();
        Matrix<T, M, M> R;
        this->_qr_decomposition(Q, R, A);
        A = Base::Matrix::matrix_multiply_Upper_triangular_A_mul_B(R, Q);

        for (std::size_t i = 0; i < k; ++i) {
          A(i, i) += mu;
        }

        if (Base::Math::abs(A(k - 1, k - 2)) < this->division_min) {
          break;
        }
      }

      this->_eigen_values[k - 1] = A(k - 1, k - 1);
      if (k == 2) {
        this->_eigen_values[0] = A(0, 0);
      }
      this->_Hessen = A;
    }
  }

  /**
   * @brief Solves eigenvalues using the QR method.
   * This function computes the eigenvalues of a square matrix using the QR
   * algorithm, with optional Hessenberg reduction and Wilkinson shift.
   * @param A0 The input square matrix whose eigenvalues are to be computed.
   */
  inline void _solve_values_with_qr_method(const Matrix<T, M, M> &A0) {
    this->_hessenberg(A0);

    this->_continue_solving_values_with_qr_method();
  }

  /**
   * @brief Solves eigenvectors using the inverse iteration method.
   * This function computes the eigenvectors of a square matrix using the
   * inverse iteration method, which is applied to each eigenvalue computed by
   * the QR method.
   * @param matrix The input square matrix whose eigenvectors are to be
   * computed.
   */
  inline void
  _solve_vectors_with_inverse_iteration_method(const Matrix<T, M, M> &matrix) {

    for (std::size_t k = 0; k < M; ++k) {
      // A - mu * I
      Matrix<T, M, M> A = matrix;
      T mu = this->_eigen_values[k] + this->small_value;
      for (std::size_t i = 0; i < M; ++i) {
        A(i, i) -= mu;
      }

      // initial eigen vector
      Vector<T, M> x(this->_eigen_vectors.data[k]);

      // inverse iteration method
      for (std::size_t iter = 0; iter < this->iteration_max; ++iter) {
        Vector<T, M> x_old = x;

        x = Base::Matrix::gmres_k(A, x_old, x, this->gmres_k_decay_rate,
                                  this->division_min, this->_gmres_k_rho,
                                  this->_gmres_k_rep_num);

        Base::Matrix::vector_normalize(x, this->division_min);

        // conversion check
        bool converged = true;
        for (std::size_t i = 0; i < M; ++i) {
          if (Base::Math::abs(Base::Math::abs(x[i]) -
                              Base::Math::abs(x_old[i])) > this->division_min) {
            converged = false;
            break;
          }
        }

        if (converged) {
          break;
        }
      }

      Base::Utility::copy<T, 0, M, 0, M, M>(x.data,
                                            this->_eigen_vectors.data[k]);
    }
  }
};

/**
 * @brief EigenSolverComplex is a template class for computing eigenvalues and
 * eigenvectors of complex square matrices.
 *
 * This class provides methods to solve eigenvalues and eigenvectors using the
 * QR algorithm, with optional Hessenberg reduction and Wilkinson shift. It
 * supports both std::vector and std::array storage for eigenvalues.
 *
 * @tparam T Type of the matrix elements (e.g., float, double).
 * @tparam M Size of the square matrix (number of rows and columns).
 */
template <typename T, std::size_t M> class EigenSolverComplex {
public:
/* Constructor */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  EigenSolverComplex()
      : iteration_max(DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        iteration_max_for_eigen_vector(3 * DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        small_value(static_cast<T>(EIGEN_SMALL_VALUE)),
        gmres_k_decay_rate(static_cast<T>(0)), _House(), _House_comp(),
        _Hessen(), _eigen_values(M, static_cast<Complex<T>>(0)),
        _eigen_vectors(Matrix<Complex<T>, M, M>::ones()),
        _gmres_k_rho(static_cast<T>(0)), _gmres_k_rep_num(0) {}

#else // __BASE_MATRIX_USE_STD_VECTOR__

  EigenSolverComplex()
      : iteration_max(DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        iteration_max_for_eigen_vector(3 * DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        small_value(static_cast<T>(EIGEN_SMALL_VALUE)),
        gmres_k_decay_rate(static_cast<T>(0)), _House(), _House_comp(),
        _Hessen(), _eigen_values(),
        _eigen_vectors(Matrix<Complex<T>, M, M>::ones()),
        _gmres_k_rho(static_cast<T>(0)), _gmres_k_rep_num(0) {}

#endif // __BASE_MATRIX_USE_STD_VECTOR__

  /* Copy Constructor */
  EigenSolverComplex(const EigenSolverComplex<T, M> &other)
      : iteration_max(other.iteration_max),
        iteration_max_for_eigen_vector(other.iteration_max_for_eigen_vector),
        division_min(other.division_min), small_value(other.small_value),
        gmres_k_decay_rate(other.gmres_k_decay_rate), _House(other._House),
        _House_comp(other._House_comp), _Hessen(other._Hessen),
        _eigen_values(other._eigen_values),
        _eigen_vectors(other._eigen_vectors), _gmres_k_rho(other._gmres_k_rho),
        _gmres_k_rep_num(other._gmres_k_rep_num) {}

  EigenSolverComplex &operator=(const EigenSolverComplex<T, M> &other) {
    if (this != &other) {
      this->iteration_max = other.iteration_max;
      this->iteration_max_for_eigen_vector =
          other.iteration_max_for_eigen_vector;
      this->division_min = other.division_min;
      this->small_value = other.small_value;
      this->gmres_k_decay_rate = other.gmres_k_decay_rate;
      this->_House = other._House;
      this->_House_comp = other._House_comp;
      this->_Hessen = other._Hessen;
      this->_eigen_values = other._eigen_values;
      this->_eigen_vectors = other._eigen_vectors;
      this->_gmres_k_rho = other._gmres_k_rho;
      this->_gmres_k_rep_num = other._gmres_k_rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  EigenSolverComplex(EigenSolverComplex<T, M> &&other) noexcept
      : iteration_max(std::move(other.iteration_max)),
        iteration_max_for_eigen_vector(
            std::move(other.iteration_max_for_eigen_vector)),
        division_min(std::move(other.division_min)),
        small_value(std::move(other.small_value)),
        gmres_k_decay_rate(std::move(other.gmres_k_decay_rate)),
        _House(std::move(other._House)),
        _House_comp(std::move(other._House_comp)),
        _Hessen(std::move(other._Hessen)),
        _eigen_values(std::move(other._eigen_values)),
        _eigen_vectors(std::move(other._eigen_vectors)),
        _gmres_k_rho(std::move(other._gmres_k_rho)),
        _gmres_k_rep_num(std::move(other._gmres_k_rep_num)) {}

  EigenSolverComplex &operator=(EigenSolverComplex<T, M> &&other) noexcept {
    if (this != &other) {
      this->iteration_max = std::move(other.iteration_max);
      this->iteration_max_for_eigen_vector =
          std::move(other.iteration_max_for_eigen_vector);
      this->division_min = std::move(other.division_min);
      this->small_value = std::move(other.small_value);
      this->gmres_k_decay_rate = std::move(other.gmres_k_decay_rate);
      this->_House = std::move(other._House);
      this->_House_comp = std::move(other._House_comp);
      this->_Hessen = std::move(other._Hessen);
      this->_eigen_values = std::move(other._eigen_values);
      this->_eigen_vectors = std::move(other._eigen_vectors);
      this->_gmres_k_rho = std::move(other._gmres_k_rho);
      this->_gmres_k_rep_num = std::move(other._gmres_k_rep_num);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Computes the eigenvalues of a complex square matrix using the QR
   * method.
   *
   * This function takes a complex square matrix as input and computes its
   * eigenvalues by internally invoking the QR algorithm. The results are stored
   * within the class instance.
   *
   * @tparam T The data type of the matrix elements.
   * @tparam M The dimension of the square matrix.
   * @param matrix The input complex square matrix whose eigenvalues are to be
   * computed.
   */
  inline void solve_eigen_values(const Matrix<T, M, M> &matrix) {
    this->_solve_with_qr_method(matrix);
  }

  /**
   * @brief Continues solving eigenvalues using the QR method.
   *
   * This function continues the process of computing eigenvalues using the QR
   * method, which may have been started previously. It is useful for iterative
   * refinement of eigenvalue calculations.
   */
  inline void continue_solving_eigen_values(void) {
    this->_continue_solving_values_with_qr_method();
  }

  /**
   * @brief Computes the eigenvectors of a complex square matrix using the
   * inverse iteration method.
   *
   * This function takes a complex square matrix as input and computes its
   * eigenvectors by internally invoking the inverse iteration method. The
   * results are stored within the class instance.
   *
   * @tparam T The data type of the matrix elements.
   * @tparam M The dimension of the square matrix.
   * @param matrix The input complex square matrix whose eigenvectors are to be
   * computed.
   */
  inline void solve_eigen_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

  /**
   * @brief Computes both eigenvalues and eigenvectors of a complex square
   * matrix.
   *
   * This function computes the eigenvalues using the QR method and the
   * eigenvectors using the inverse iteration method, storing the results
   * within the class instance.
   *
   * @tparam T The data type of the matrix elements.
   * @tparam M The dimension of the square matrix.
   * @param matrix The input complex square matrix whose eigenvalues and
   * eigenvectors are to be computed.
   */
  inline void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_with_qr_method(matrix);
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  /**
   * @brief Retrieves the computed eigenvalues.
   * This function returns the eigenvalues computed by the QR method as a
   * vector.
   * @return std::vector<Complex<T>> A vector containing the eigenvalues.
   * */
  inline std::vector<Complex<T>> get_eigen_values(void) {
    return this->_eigen_values;
  }
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  /**
   * @brief Retrieves the computed eigenvalues.
   * This function returns the eigenvalues computed by the QR method as an
   * array.
   * @return std::array<Complex<T>, M> An array containing the eigenvalues.
   * */
  inline std::array<Complex<T>, M> get_eigen_values(void) {
    return this->_eigen_values;
  }
#endif // __BASE_MATRIX_USE_STD_VECTOR__

  /**
   * @brief Retrieves the computed eigenvectors.
   * This function returns the eigenvectors computed by the inverse iteration
   * method as a matrix.
   * @return Matrix<Complex<T>, M, M> A matrix containing the eigenvectors.
   */
  inline Matrix<Complex<T>, M, M> get_eigen_vectors(void) {
    return this->_eigen_vectors;
  }

  /**
   * @brief Retrieves the Hessenberg form of the matrix.
   * This function returns the Hessenberg form of the matrix computed during the
   * QR method.
   * @return Matrix<Complex<T>, M, M> A matrix in Hessenberg form.
   */
  inline void set_iteration_max(const std::size_t &iteration_max_in) {
    this->iteration_max = iteration_max_in;
  }

  /**
   * @brief Sets the maximum number of iterations for eigenvector computation.
   * This function sets the maximum number of iterations for computing
   * eigenvectors using the inverse iteration method.
   * @param iteration_max_for_eigen_vector_in The maximum number of iterations
   * for eigenvector computation.
   */
  inline void set_iteration_max_for_eigen_vector(
      const std::size_t &iteration_max_for_eigen_vector_in) {
    this->iteration_max_for_eigen_vector = iteration_max_for_eigen_vector_in;
  }

  /**
   * @brief Sets the minimum division value for numerical stability.
   * This function sets the minimum value used to avoid division by zero in
   * numerical computations.
   * @param division_min_in The minimum division value.
   */
  inline void set_division_min(const T &division_min_in) {
    this->division_min = division_min_in;
  }

  /**
   * @brief Sets a small value for numerical stability.
   * This function sets a small value used to determine convergence in the QR
   * method.
   * @param small_value_in The small value for numerical stability.
   */
  inline void set_small_value(const T &small_value_in) {
    this->small_value = small_value_in;
  }

  /**
   * @brief Sets the decay rate for GMRES k.
   * This function sets the decay rate for the GMRES k method, which is used in
   * iterative methods for solving linear systems.
   * @param gmres_k_decay_rate_in The decay rate for GMRES k.
   */
  inline void set_gmres_k_decay_rate(const T &gmres_k_decay_rate_in) {
    this->gmres_k_decay_rate = gmres_k_decay_rate_in;
  }

  /**
   * @brief Retrieves the Hessenberg form of the matrix.
   * This function returns the Hessenberg form of the matrix computed during the
   * QR method.
   * @return Matrix<Complex<T>, M, M> A matrix in Hessenberg form.
   */
  inline Matrix<Complex<T>, M, M>
  check_validity(const Matrix<T, M, M> &matrix) {
    Matrix<Complex<T>, M, M> result =
        Base::Matrix::convert_matrix_real_to_complex(matrix) *
            this->_eigen_vectors -
        this->_eigen_vectors *
            Base::Matrix::DiagMatrix<Complex<T>, M>(this->_eigen_values);

    return result;
  }

public:
  /* Variable */
  std::size_t iteration_max;
  std::size_t iteration_max_for_eigen_vector;
  T division_min;
  T small_value;
  T gmres_k_decay_rate;

protected:
  /* Variable */
  VariableSparseMatrix<T, M, M> _House;
  VariableSparseMatrix<Complex<T>, M, M> _House_comp;
  Matrix<Complex<T>, M, M> _Hessen;
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<Complex<T>> _eigen_values;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<Complex<T>, M> _eigen_values;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
  Matrix<Complex<T>, M, M> _eigen_vectors;
  T _gmres_k_rho = static_cast<T>(0);
  std::size_t _gmres_k_rep_num = static_cast<std::size_t>(0);

protected:
  /* Function */

  /**
   * @brief Solves eigenvalues using the QR method.
   * This function computes the eigenvalues of a square matrix using the QR
   * algorithm, with optional Hessenberg reduction and Wilkinson shift.
   * @param A The input square matrix whose eigenvalues are to be computed.
   */
  inline void _hessenberg(const Matrix<T, M, M> &A) {
    Matrix<T, M, M> R = A;
    std::array<T, M> u;

    for (std::size_t k = 0; k < M - 2; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k + 1; i < M; ++i) {
        x_abs += R(i, k) * R(i, k);
      }
      if (Base::Utility::near_zero(x_abs, this->division_min)) {
        continue;
      }
      x_abs = Base::Math::sqrt<T>(x_abs, this->division_min);

      u[k + 1] = R(k + 1, k) + Base::Utility::sign(R(k + 1, k)) * x_abs;
      T u_abs = u[k + 1] * u[k + 1];
      for (std::size_t i = k + 2; i < M; ++i) {
        u[i] = R(i, k);
        u_abs += u[i] * u[i];
      }

      std::fill(this->_House.values.begin(), this->_House.values.end(),
                static_cast<T>(0));
      std::fill(this->_House.row_indices.begin(),
                this->_House.row_indices.end(), static_cast<std::size_t>(0));
      std::fill(this->_House.row_pointers.begin(),
                this->_House.row_pointers.end(), static_cast<std::size_t>(0));

      std::size_t H_value_count = 0;
      for (std::size_t i = 0; i < k + 1; i++) {
        this->_House.values[H_value_count] = static_cast<T>(1);
        this->_House.row_indices[H_value_count] = i;
        H_value_count++;

        this->_House.row_pointers[i + 1] = H_value_count;
      }

      for (std::size_t i = k + 1; i < M; ++i) {
        for (std::size_t j = k + 1; j < M; ++j) {

          if (i == j) {
            this->_House.values[H_value_count] = static_cast<T>(1);
          }

          this->_House.values[H_value_count] -=
              static_cast<T>(2) * u[i] * u[j] /
              Base::Utility::avoid_zero_divide(u_abs, this->division_min);

          this->_House.row_indices[H_value_count] = j;
          H_value_count++;
        }
        this->_House.row_pointers[i + 1] = H_value_count;
      }

      R = this->_House * R * this->_House;
    }

    this->_Hessen = Base::Matrix::convert_matrix_real_to_complex(R);
  }

  /**
   * @brief Solves eigenvalues using the QR method.
   * This function computes the eigenvalues of a square matrix using the QR
   * algorithm, with optional Hessenberg reduction and Wilkinson shift.
   * @param A The input square matrix whose eigenvalues are to be computed.
   */
  inline void _qr_decomposition(Matrix<Complex<T>, M, M> &Q,
                                Matrix<Complex<T>, M, M> &R,
                                const Matrix<Complex<T>, M, M> &A) {
    R = A;
    std::array<Complex<T>, M> u;

    for (std::size_t k = 0; k < M - 1; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k; i < k + 2; ++i) {
        x_abs += Base::Matrix::complex_abs_sq(R(i, k));
      }
      if (Base::Utility::near_zero(x_abs, this->division_min)) {
        continue;
      }
      x_abs = Base::Math::sqrt<T>(x_abs, this->division_min);

      u[k] = R(k, k) +
             Base::Matrix::complex_sign(R(k, k), this->division_min) * x_abs;
      u[k + 1] = R(k + 1, k);
      T u_abs = Base::Matrix::complex_abs_sq(u[k]) +
                Base::Matrix::complex_abs_sq(u[k + 1]);

      std::fill(this->_House_comp.values.begin(),
                this->_House_comp.values.end(), Complex<T>());
      std::fill(this->_House_comp.row_indices.begin(),
                this->_House_comp.row_indices.end(),
                static_cast<std::size_t>(0));
      std::fill(this->_House_comp.row_pointers.begin(),
                this->_House_comp.row_pointers.end(),
                static_cast<std::size_t>(0));

      std::size_t H_value_count = 0;
      for (std::size_t i = 0; i < k; i++) {
        this->_House_comp.values[H_value_count] = static_cast<T>(1);
        this->_House_comp.row_indices[H_value_count] = i;
        H_value_count++;

        this->_House_comp.row_pointers[i + 1] = H_value_count;
      }

      for (std::size_t i = k; i < k + 2; ++i) {
        for (std::size_t j = k; j < k + 2; ++j) {

          if (i == j) {
            this->_House_comp.values[H_value_count] = static_cast<T>(1);
          }

          this->_House_comp.values[H_value_count] -=
              static_cast<T>(2) *
              (u[i] * Base::Matrix::complex_conjugate(u[j])) /
              Base::Utility::avoid_zero_divide(u_abs, this->division_min);

          this->_House_comp.row_indices[H_value_count] = j;
          H_value_count++;
        }
        this->_House_comp.row_pointers[i + 1] = H_value_count;
      }

      for (std::size_t i = k + 2; i < M; i++) {
        this->_House_comp.values[H_value_count] = static_cast<T>(1);
        this->_House_comp.row_indices[H_value_count] = i;
        H_value_count++;

        this->_House_comp.row_pointers[i + 1] = H_value_count;
      }

      R = this->_House_comp * R;
      Q = Q * this->_House_comp;
    }
  }

  /**
   * @brief Computes the Wilkinson shift for the QR method.
   * This function computes the Wilkinson shift, which is used to improve the
   * convergence of the QR algorithm for eigenvalue computation.
   * @param A The input square matrix from which the Wilkinson shift is
   * computed.
   * @return Complex<T> The computed Wilkinson shift value.
   */
  inline Complex<T> _wilkinson_shift(const Matrix<Complex<T>, M, M> &A) {
    Complex<T> a11 = A(M - 2, M - 2);
    Complex<T> a12 = A(M - 2, M - 1);
    Complex<T> a21 = A(M - 1, M - 2);
    Complex<T> a22 = A(M - 1, M - 1);
    Complex<T> c1 = a11 + a22;

    Complex<T> c2_2 = (a11 - a22) * (a11 - a22) + static_cast<T>(4) * a12 * a21;
    Complex<T> c2 = Base::Matrix::complex_sqrt(c2_2, this->division_min);

    Complex<T> mu1 = static_cast<T>(0.5) * (c1 + c2);
    Complex<T> mu2 = static_cast<T>(0.5) * (c1 - c2);

    if (Base::Matrix::complex_abs(a22 - mu1) <=
        Base::Matrix::complex_abs(a22 - mu2)) {
      return mu1;
    } else {
      return mu2;
    }
  }

  /**
   * @brief Solves eigenvalues using the QR method.
   * This function computes the eigenvalues of a square matrix using the QR
   * algorithm, with optional Hessenberg reduction and Wilkinson shift.
   * @param A0 The input square matrix whose eigenvalues are to be computed.
   */
  inline void _solve_with_qr_method(const Matrix<T, M, M> &A0) {
    this->_hessenberg(A0);

    this->_continue_solving_values_with_qr_method();
  }

  /**
   * @brief Continues solving eigenvalues using the QR method.
   * This function continues the process of computing eigenvalues using the QR
   * method, which may have been started previously. It is useful for iterative
   * refinement of eigenvalue calculations.
   */
  inline void _continue_solving_values_with_qr_method(void) {
    for (std::size_t k = M; k > 1; --k) {
      Matrix<Complex<T>, M, M> A = this->_Hessen;

      for (std::size_t iter = 0; iter < this->iteration_max; ++iter) {
        Complex<T> mu = this->_wilkinson_shift(A);
        for (std::size_t i = 0; i < k; ++i) {
          A(i, i) -= mu;
        }

        Matrix<Complex<T>, M, M> Q = Matrix<Complex<T>, M, M>::identity();
        Matrix<Complex<T>, M, M> R;
        this->_qr_decomposition(Q, R, A);
        A = Base::Matrix::matrix_multiply_Upper_triangular_A_mul_B(R, Q);

        for (std::size_t i = 0; i < k; ++i) {
          A(i, i) += mu;
        }

        if (Base::Matrix::complex_abs(A(k - 1, k - 2)) < this->division_min) {
          break;
        }
      }

      this->_eigen_values[k - 1] = A(k - 1, k - 1);
      if (k == 2) {
        this->_eigen_values[0] = A(0, 0);
      }
      this->_Hessen = A;
    }
  }

  /**
   * @brief Solves eigenvectors using the inverse iteration method.
   * This function computes the eigenvectors of a square matrix using the
   * inverse iteration method, which is applied to each eigenvalue computed by
   * the QR method.
   * @param matrix The input square matrix whose eigenvectors are to be
   * computed.
   */
  inline void
  _solve_vectors_with_inverse_iteration_method(const Matrix<T, M, M> &matrix) {

    for (std::size_t k = 0; k < M; ++k) {
      // A - mu * I
      Matrix<Complex<T>, M, M> A =
          Base::Matrix::convert_matrix_real_to_complex(matrix);

      Complex<T> mu = this->_eigen_values[k] + this->small_value;
      for (std::size_t i = 0; i < M; ++i) {
        A(i, i) -= mu;
      }

      // initial eigen vector
      Vector<Complex<T>, M> x(this->_eigen_vectors.data[k]);

      // inverse iteration method
      for (std::size_t iter = 0; iter < this->iteration_max_for_eigen_vector;
           ++iter) {
        Vector<Complex<T>, M> x_old = x;

        x = Base::Matrix::complex_gmres_k(
            A, x_old, x, this->gmres_k_decay_rate, this->division_min,
            this->_gmres_k_rho, this->_gmres_k_rep_num);

        Base::Matrix::complex_vector_normalize(x, this->division_min);

        // conversion check
        bool converged = true;
        for (std::size_t i = 0; i < M; ++i) {
          if (Base::Math::abs(Base::Math::abs(x[i].real) -
                              Base::Math::abs(x_old[i].real)) +
                  Base::Math::abs(Base::Math::abs(x[i].imag) -
                                  Base::Math::abs(x_old[i].imag)) >
              this->division_min) {
            converged = false;
            break;
          }
        }

        if (converged) {
          break;
        }
      }

      for (std::size_t i = 0; i < M; ++i) {
        this->_eigen_vectors(i, k) = x[i];
      }
    }
  }
};

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_EIGEN_SOLVER_HPP__
