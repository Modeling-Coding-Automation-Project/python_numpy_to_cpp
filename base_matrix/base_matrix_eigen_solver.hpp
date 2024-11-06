#ifndef BASE_MATRIX_EIGEN_SOLVER_HPP
#define BASE_MATRIX_EIGEN_SOLVER_HPP

#include "base_matrix_complex.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_lu_decomposition.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_variable_sparse.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace Base {
namespace Matrix {

const double EIGEN_SMALL_VALUE = 1.0e-6;

template <typename T, std::size_t M> class EigenSolverReal {
public:
#ifdef USE_STD_VECTOR

  EigenSolverReal()
      : iteration_max(0), _division_min(static_cast<T>(0)),
        _eigen_values(M, static_cast<T>(0)),
        _eigen_vectors(Matrix<T, M, M>::ones()) {}

  EigenSolverReal(const Matrix<T, M, M> &matrix, std::size_t iteration_max,
                  T division_min)
      : iteration_max(iteration_max), _division_min(division_min),
        _eigen_values(M, static_cast<T>(0)),
        _eigen_vectors(Matrix<T, M, M>::ones()) {
    static_assert(M > 1, "Matrix must be larger than 2x2.");

    this->_solve_values_with_qr_method(matrix);
  }

#else

  EigenSolverReal()
      : iteration_max(0), _division_min(static_cast<T>(0)), _eigen_values{},
        _eigen_vectors(Matrix<T, M, M>::ones()) {}

  EigenSolverReal(const Matrix<T, M, M> &matrix, std::size_t iteration_max,
                  T division_min)
      : iteration_max(iteration_max), _division_min(division_min),
        _eigen_values{}, _eigen_vectors(Matrix<T, M, M>::ones()) {
    static_assert(M > 1, "Matrix must be larger than 2x2.");

    this->_solve_values_with_qr_method(matrix);
  }

#endif

  /* Copy Constructor */
  EigenSolverReal(EigenSolverReal<T, M> &other)
      : iteration_max(other.iteration_max), _House(other._House),
        _Hessen(other._Hessen), _eigen_values(other._eigen_values),
        _division_min(other._division_min),
        _eigen_vectors(other._eigen_vectors), _small_value(other._small_value),
        _gmres_k_decay_rate(other._gmres_k_decay_rate),
        _gmres_k_rho(other._gmres_k_rho),
        _gmres_k_rep_num(other._gmres_k_rep_num) {}

  EigenSolverReal &operator=(EigenSolverReal<T, M> &other) {
    if (this != &other) {
      this->iteration_max = other.iteration_max;
      this->_House = other._House;
      this->_Hessen = other._Hessen;
      this->_eigen_values = other._eigen_values;
      this->_division_min = other._division_min;
      this->_eigen_vectors = other._eigen_vectors;
      this->_small_value = other._small_value;
      this->_gmres_k_decay_rate = other._gmres_k_decay_rate;
      this->_gmres_k_rho = other._gmres_k_rho;
      this->_gmres_k_rep_num = other._gmres_k_rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  EigenSolverReal(EigenSolverReal<T, M> &&other) noexcept
      : iteration_max(std::move(other.iteration_max)),
        _House(std::move(other._House)), _Hessen(std::move(other._Hessen)),
        _eigen_values(std::move(other._eigen_values)),
        _division_min(std::move(other._division_min)),
        _eigen_vectors(std::move(other._eigen_vectors)),
        _small_value(std::move(other._small_value)),
        _gmres_k_decay_rate(std::move(other._gmres_k_decay_rate)),
        _gmres_k_rho(std::move(other._gmres_k_rho)),
        _gmres_k_rep_num(std::move(other._gmres_k_rep_num)) {}

  EigenSolverReal &operator=(EigenSolverReal<T, M> &&other) noexcept {
    if (this != &other) {
      this->iteration_max = std::move(other.iteration_max);

      this->_House = std::move(other._House);
      this->_Hessen = std::move(other._Hessen);
      this->_eigen_values = std::move(other._eigen_values);
      this->_division_min = std::move(other._division_min);
      this->_eigen_vectors = std::move(other._eigen_vectors);
      this->_small_value = std::move(other._small_value);
      this->_gmres_k_decay_rate = std::move(other._gmres_k_decay_rate);
      this->_gmres_k_rho = std::move(other._gmres_k_rho);
      this->_gmres_k_rep_num = std::move(other._gmres_k_rep_num);
    }
    return *this;
  }

  /* Function */
  void solve_eigen_values(const Matrix<T, M, M> &matrix) {
    this->_solve_values_with_qr_method(matrix);
  }

  void continue_solving_eigen_values(void) {
    this->_continue_solving_values_with_qr_method();
  }

  void solve_eigen_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

  void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_values_with_qr_method(matrix);
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

#ifdef USE_STD_VECTOR
  std::vector<T> get_eigen_values(void) { return this->_eigen_values; }
#else
  std::array<T, M> get_eigen_values(void) { return this->_eigen_values; }
#endif

  Matrix<T, M, M> get_eigen_vectors(void) { return this->_eigen_vectors; }

  /* Variable */
  std::size_t iteration_max;

private:
  VariableSparseMatrix<T, M, M> _House;
  Matrix<T, M, M> _Hessen;
#ifdef USE_STD_VECTOR
  std::vector<T> _eigen_values;
#else
  std::array<T, M> _eigen_values;
#endif
  T _division_min;
  Matrix<T, M, M> _eigen_vectors;
  T _small_value = static_cast<T>(EIGEN_SMALL_VALUE);
  T _gmres_k_decay_rate = static_cast<T>(0.2);
  T _gmres_k_rho = static_cast<T>(0);
  std::size_t _gmres_k_rep_num = static_cast<std::size_t>(0);

  /* Function */
  void _hessenberg(const Matrix<T, M, M> &A) {
    Matrix<T, M, M> R = A;
    std::array<T, M> u;

    for (std::size_t k = 0; k < M - 2; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k + 1; i < M; ++i) {
        x_abs += R(i, k) * R(i, k);
      }
      if (near_zero(x_abs, this->_division_min)) {
        continue;
      }
      x_abs = std::sqrt(x_abs);

      u[k + 1] = R(k + 1, k) + sign(R(k + 1, k)) * x_abs;
      T u_abs = u[k + 1] * u[k + 1];
      for (std::size_t i = k + 2; i < M; ++i) {
        u[i] = R(i, k);
        u_abs += u[i] * u[i];
      }

      std::memset(&this->_House.values[0], 0,
                  (M * M) * sizeof(this->_House.values[0]));
      std::memset(&this->_House.row_indices[0], 0,
                  (M * M) * sizeof(this->_House.row_indices[0]));
      std::memset(&this->_House.row_pointers[0], 0,
                  (M + 1) * sizeof(this->_House.row_pointers[0]));

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
              avoid_zero_divide(u_abs, this->_division_min);

          this->_House.row_indices[H_value_count] = j;
          H_value_count++;
        }
        this->_House.row_pointers[i + 1] = H_value_count;
      }

      R = this->_House * R * this->_House;
    }

    this->_Hessen = R;
  }

  void _qr_decomposition(Matrix<T, M, M> &Q, Matrix<T, M, M> &R,
                         const Matrix<T, M, M> &A) {
    R = A;
    std::array<T, M> u;

    for (std::size_t k = 0; k < M - 1; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k; i < k + 2; ++i) {
        x_abs += R(i, k) * R(i, k);
      }
      if (near_zero(x_abs, this->_division_min)) {
        continue;
      }
      x_abs = std::sqrt(x_abs);

      u[k] = R(k, k) + sign(R(k, k)) * x_abs;
      u[k + 1] = R(k + 1, k);
      T u_abs = u[k] * u[k] + u[k + 1] * u[k + 1];

      std::memset(&this->_House.values[0], 0,
                  (M * M) * sizeof(this->_House.values[0]));
      std::memset(&this->_House.row_indices[0], 0,
                  (M * M) * sizeof(this->_House.row_indices[0]));
      std::memset(&this->_House.row_pointers[0], 0,
                  (M + 1) * sizeof(this->_House.row_pointers[0]));

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
              avoid_zero_divide(u_abs, this->_division_min);

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

  T _wilkinson_shift(const Matrix<T, M, M> &A) {
    T a11 = A(M - 2, M - 2);
    T a12 = A(M - 2, M - 1);
    T a21 = A(M - 1, M - 2);
    T a22 = A(M - 1, M - 1);
    T c1 = a11 + a22;

    T c2_2 = (a11 - a22) * (a11 - a22) + static_cast<T>(4) * a12 * a21;
    T c2;
    if (c2_2 >= 0) {
      c2 = std::sqrt(c2_2);
    } else {
      c2 = static_cast<T>(0);
    }

    T mu1 = static_cast<T>(0.5) * (c1 + c2);
    T mu2 = static_cast<T>(0.5) * (c1 - c2);
    T dmu1 = std::abs(a22 - mu1);
    T dmu2 = std::abs(a22 - mu2);
    return (dmu1 <= dmu2) ? mu1 : mu2;
  }

  void _continue_solving_values_with_qr_method(void) {
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
        A = matrix_multiply_Upper_triangular_A_mul_B(R, Q);

        for (std::size_t i = 0; i < k; ++i) {
          A(i, i) += mu;
        }

        if (std::abs(A(k - 1, k - 2)) < this->_division_min) {
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

  void _solve_values_with_qr_method(const Matrix<T, M, M> &A0) {
    this->_hessenberg(A0);

    this->_continue_solving_values_with_qr_method();
  }

  void
  _solve_vectors_with_inverse_iteration_method(const Matrix<T, M, M> &matrix) {

    for (std::size_t k = 0; k < M; ++k) {
      // A - mu * I
      Matrix<T, M, M> A = matrix;
      T mu = this->_eigen_values[k] + this->_small_value;
      for (std::size_t i = 0; i < M; ++i) {
        A(i, i) -= mu;
      }

      // initial eigen vector
      Vector<T, M> x(this->_eigen_vectors.data[k]);

      // inverse iteration method
      for (std::size_t iter = 0; iter < this->iteration_max; ++iter) {
        Vector<T, M> x_old = x;

        x = gmres_k(A, x_old, x, this->_gmres_k_decay_rate, this->_division_min,
                    this->_gmres_k_rho, this->_gmres_k_rep_num);

        // normalization
        T norm_inv = static_cast<T>(1) /
                     avoid_zero_divide(x.norm(), this->_division_min);
        for (std::size_t i = 0; i < M; ++i) {
          x[i] *= norm_inv;
        }

        // conversion check
        bool converged = true;
        for (std::size_t i = 0; i < M; ++i) {
          if (std::abs(std::abs(x[i]) - std::abs(x_old[i])) >
              this->_division_min) {
            converged = false;
            break;
          }
        }

        if (converged) {
          break;
        }
      }

      std::copy(x.data.begin(), x.data.end(),
                this->_eigen_vectors.data[k].begin());
    }
  }
};

template <typename T, std::size_t M> class EigenSolverComplex {
public:
#ifdef USE_STD_VECTOR

  EigenSolverComplex()
      : iteration_max(0), iteration_max_for_eigen_vector(0),
        _division_min(static_cast<T>(0)), _eigen_values(M, static_cast<T>(0)),
        _eigen_vectors(Matrix<Complex<T>, M, M>::ones()) {}

  EigenSolverComplex(const Matrix<T, M, M> &matrix, std::size_t iteration_max,
                     T division_min)
      : iteration_max(iteration_max),
        iteration_max_for_eigen_vector(iteration_max * 3),
        _division_min(division_min), _eigen_values(M, static_cast<T>(0)),
        _eigen_vectors(Matrix<Complex<T>, M, M>::ones()) {
    static_assert(M > 1, "Matrix must be larger than 2x2.");

    this->_solve_with_qr_method(matrix);
  }

#else

  EigenSolverComplex()
      : iteration_max(0), iteration_max_for_eigen_vector(0),
        _division_min(static_cast<T>(0)), _eigen_values{},
        _eigen_vectors(Matrix<Complex<T>, M, M>::ones()) {}

  EigenSolverComplex(const Matrix<T, M, M> &matrix, std::size_t iteration_max,
                     T division_min)
      : iteration_max(iteration_max),
        iteration_max_for_eigen_vector(iteration_max * 3),
        _division_min(division_min), _eigen_values{},
        _eigen_vectors(Matrix<Complex<T>, M, M>::ones()) {
    static_assert(M > 1, "Matrix must be larger than 2x2.");

    this->_solve_with_qr_method(matrix);
  }

#endif

  /* Copy Constructor */
  EigenSolverComplex(EigenSolverComplex<T, M> &other)
      : iteration_max(other.iteration_max),
        iteration_max_for_eigen_vector(other.iteration_max_for_eigen_vector),
        _House(other._House), _Hessen(other._Hessen),
        _eigen_values(other._eigen_values), _division_min(other._division_min),
        _eigen_vectors(other._eigen_vectors), _small_value(other._small_value),
        _gmres_k_decay_rate(other._gmres_k_decay_rate),
        _gmres_k_rho(other._gmres_k_rho),
        _gmres_k_rep_num(other._gmres_k_rep_num) {}

  EigenSolverComplex &operator=(EigenSolverComplex<T, M> &other) {
    if (this != &other) {
      this->iteration_max = other.iteration_max;
      this->iteration_max_for_eigen_vector =
          other.iteration_max_for_eigen_vector;
      this->_House = other._House;
      this->_Hessen = other._Hessen;
      this->_eigen_values = other._eigen_values;
      this->_division_min = other._division_min;
      this->_eigen_vectors = other._eigen_vectors;
      this->_small_value = other._small_value;
      this->_gmres_k_decay_rate = other._gmres_k_decay_rate;
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
        _House(std::move(other._House)), _Hessen(std::move(other._Hessen)),
        _eigen_values(std::move(other._eigen_values)),
        _division_min(std::move(other._division_min)),
        _eigen_vectors(std::move(other._eigen_vectors)),
        _small_value(std::move(other._small_value)),
        _gmres_k_decay_rate(std::move(other._gmres_k_decay_rate)),
        _gmres_k_rho(std::move(other._gmres_k_rho)),
        _gmres_k_rep_num(std::move(other._gmres_k_rep_num)) {}

  EigenSolverComplex &operator=(EigenSolverComplex<T, M> &&other) noexcept {
    if (this != &other) {
      this->iteration_max = std::move(other.iteration_max);
      this->iteration_max_for_eigen_vector =
          std::move(other.iteration_max_for_eigen_vector);
      this->_House = std::move(other._House);
      this->_Hessen = std::move(other._Hessen);
      this->_eigen_values = std::move(other._eigen_values);
      this->_division_min = std::move(other._division_min);
      this->_eigen_vectors = std::move(other._eigen_vectors);
      this->_small_value = std::move(other._small_value);
      this->_gmres_k_decay_rate = std::move(other._gmres_k_decay_rate);
      this->_gmres_k_rho = std::move(other._gmres_k_rho);
      this->_gmres_k_rep_num = std::move(other._gmres_k_rep_num);
    }
    return *this;
  }

  /* Function */
  void solve_eigen_values(const Matrix<T, M, M> &matrix) {
    this->_solve_with_qr_method(matrix);
  }

  void continue_solving_eigen_values(void) {
    this->_continue_solving_values_with_qr_method();
  }

  void solve_eigen_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

  void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->_solve_with_qr_method(matrix);
    this->_solve_vectors_with_inverse_iteration_method(matrix);
  }

#ifdef USE_STD_VECTOR
  std::vector<Complex<T>> get_eigen_values(void) { return this->_eigen_values; }
#else
  std::array<Complex<T>, M> get_eigen_values(void) {
    return this->_eigen_values;
  }
#endif

  Matrix<Complex<T>, M, M> get_eigen_vectors(void) {
    return this->_eigen_vectors;
  }

  /* Variable */
  std::size_t iteration_max;
  std::size_t iteration_max_for_eigen_vector;

private:
  VariableSparseMatrix<T, M, M> _House;
  VariableSparseMatrix<Complex<T>, M, M> _House_comp;
  Matrix<Complex<T>, M, M> _Hessen;
#ifdef USE_STD_VECTOR
  std::vector<Complex<T>> _eigen_values;
#else
  std::array<Complex<T>, M> _eigen_values;
#endif
  T _division_min;
  Matrix<Complex<T>, M, M> _eigen_vectors;
  T _small_value = static_cast<T>(EIGEN_SMALL_VALUE);
  T _gmres_k_decay_rate = static_cast<T>(0.2);
  T _gmres_k_rho = static_cast<T>(0);
  std::size_t _gmres_k_rep_num = static_cast<std::size_t>(0);

  /* Function */
  void _hessenberg(const Matrix<T, M, M> &A) {
    Matrix<T, M, M> R = A;
    std::array<T, M> u;

    for (std::size_t k = 0; k < M - 2; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k + 1; i < M; ++i) {
        x_abs += R(i, k) * R(i, k);
      }
      if (near_zero(x_abs, this->_division_min)) {
        continue;
      }
      x_abs = std::sqrt(x_abs);

      u[k + 1] = R(k + 1, k) + sign(R(k + 1, k)) * x_abs;
      T u_abs = u[k + 1] * u[k + 1];
      for (std::size_t i = k + 2; i < M; ++i) {
        u[i] = R(i, k);
        u_abs += u[i] * u[i];
      }

      std::memset(&this->_House.values[0], 0,
                  (M * M) * sizeof(this->_House.values[0]));
      std::memset(&this->_House.row_indices[0], 0,
                  (M * M) * sizeof(this->_House.row_indices[0]));
      std::memset(&this->_House.row_pointers[0], 0,
                  (M + 1) * sizeof(this->_House.row_pointers[0]));

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
              avoid_zero_divide(u_abs, this->_division_min);

          this->_House.row_indices[H_value_count] = j;
          H_value_count++;
        }
        this->_House.row_pointers[i + 1] = H_value_count;
      }

      R = this->_House * R * this->_House;
    }

    this->_Hessen = convert_matrix_real_to_complex(R);
  }

  void _qr_decomposition(Matrix<Complex<T>, M, M> &Q,
                         Matrix<Complex<T>, M, M> &R,
                         const Matrix<Complex<T>, M, M> &A) {
    R = A;
    std::array<Complex<T>, M> u;

    for (std::size_t k = 0; k < M - 1; ++k) {
      T x_abs = static_cast<T>(0);
      for (std::size_t i = k; i < k + 2; ++i) {
        x_abs += complex_abs_sq(R(i, k));
      }
      if (near_zero(x_abs, this->_division_min)) {
        continue;
      }
      x_abs = std::sqrt(x_abs);

      u[k] = R(k, k) + complex_sign(R(k, k), this->_division_min) * x_abs;
      u[k + 1] = R(k + 1, k);
      T u_abs = complex_abs_sq(u[k]) + complex_abs_sq(u[k + 1]);

      std::memset(&this->_House_comp.values[0], 0,
                  (M * M) * sizeof(this->_House_comp.values[0]));
      std::memset(&this->_House_comp.row_indices[0], 0,
                  (M * M) * sizeof(this->_House_comp.row_indices[0]));
      std::memset(&this->_House_comp.row_pointers[0], 0,
                  (M + 1) * sizeof(this->_House_comp.row_pointers[0]));

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
              static_cast<T>(2) * (u[i] * complex_conjugate(u[j])) /
              avoid_zero_divide(u_abs, this->_division_min);

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

  Complex<T> _wilkinson_shift(const Matrix<Complex<T>, M, M> &A) {
    Complex<T> a11 = A(M - 2, M - 2);
    Complex<T> a12 = A(M - 2, M - 1);
    Complex<T> a21 = A(M - 1, M - 2);
    Complex<T> a22 = A(M - 1, M - 1);
    Complex<T> c1 = a11 + a22;

    Complex<T> c2_2 = (a11 - a22) * (a11 - a22) + static_cast<T>(4) * a12 * a21;
    Complex<T> c2 = complex_sqrt(c2_2);

    Complex<T> mu1 = static_cast<T>(0.5) * (c1 + c2);
    Complex<T> mu2 = static_cast<T>(0.5) * (c1 - c2);

    if (complex_abs(a22 - mu1) <= complex_abs(a22 - mu2)) {
      return mu1;
    } else {
      return mu2;
    }
  }

  void _solve_with_qr_method(const Matrix<T, M, M> &A0) {
    this->_hessenberg(A0);

    this->_continue_solving_values_with_qr_method();
  }

  void _continue_solving_values_with_qr_method(void) {
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
        A = matrix_multiply_Upper_triangular_A_mul_B(R, Q);

        for (std::size_t i = 0; i < k; ++i) {
          A(i, i) += mu;
        }

        if (complex_abs(A(k - 1, k - 2)) < this->_division_min) {
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

  void
  _solve_vectors_with_inverse_iteration_method(const Matrix<T, M, M> &matrix) {

    for (std::size_t k = 0; k < M; ++k) {
      // A - mu * I
      Matrix<Complex<T>, M, M> A = convert_matrix_real_to_complex(matrix);

      Complex<T> mu = this->_eigen_values[k] + this->_small_value;
      for (std::size_t i = 0; i < M; ++i) {
        A(i, i) -= mu;
      }

      // initial eigen vector
      Vector<Complex<T>, M> x(this->_eigen_vectors.data[k]);

      // inverse iteration method
      for (std::size_t iter = 0; iter < this->iteration_max_for_eigen_vector;
           ++iter) {
        Vector<Complex<T>, M> x_old = x;

        x = complex_gmres_k(A, x_old, x, this->_gmres_k_decay_rate,
                            this->_division_min, this->_gmres_k_rho,
                            this->_gmres_k_rep_num);

        // normalization
        T norm_inv =
            static_cast<T>(1) /
            avoid_zero_divide(complex_vector_norm(x), this->_division_min);
        for (std::size_t i = 0; i < M; ++i) {
          x[i] *= norm_inv;
        }

        // conversion check
        bool converged = true;
        for (std::size_t i = 0; i < M; ++i) {
          if (std::abs(std::abs(x[i].real) - std::abs(x_old[i].real)) +
                  std::abs(std::abs(x[i].imag) - std::abs(x_old[i].imag)) >
              this->_division_min) {
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

#endif // BASE_MATRIX_EIGEN_SOLVER_HPP
