#ifndef PYTHON_NUMPY_LINALG_SOLVER_HPP
#define PYTHON_NUMPY_LINALG_SOLVER_HPP

#include "python_numpy_base.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_SOLVER = 1.0e-10;

template <typename T, std::size_t M, std::size_t K, std::size_t V,
          std::size_t W>
class LinalgSolver {
public:
  /* Constructor */
  LinalgSolver() {}

  LinalgSolver(const Matrix<DefDense, T, M, M> &A) { this->inv(A); }

  LinalgSolver(const Matrix<DefDiag, T, M> &A) { this->inv(A); }

  LinalgSolver(const Matrix<DefSparse, T, M, K, V> &A) { this->inv(A); }

  LinalgSolver(const Matrix<DefDense, T, M, M> &A,
               const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefDense, T, M, M> &A,
               const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefDense, T, M, M> &A,
               const Matrix<DefSparse, T, M, K, V> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefSparse, T, M, K, V> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefSparse, T, M, M, V> &A,
               const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefSparse, T, M, M, V> &A,
               const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefSparse, T, M, M, V> &A,
               const Matrix<DefSparse, T, M, K, W> &B) {

    this->solve(A, B);
  }

  /* Copy Constructor */
  LinalgSolver(const LinalgSolver<T, M, K, V, W> &other)
      : X_1(other.X_1), decay_rate(other.decay_rate),
        division_min(other.division_min), rho(other.rho),
        rep_num(other.rep_num) {}

  LinalgSolver<T, M, K, V, W> &
  operator=(const LinalgSolver<T, M, K, V, W> &other) {
    if (this != &other) {
      this->X_1 = other.X_1;
      this->decay_rate = other.decay_rate;
      this->division_min = other.division_min;
      this->rho = other.rho;
      this->rep_num = other.rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolver(LinalgSolver<T, M, K, V, W> &&other) noexcept
      : X_1(std::move(other.X_1)), decay_rate(std::move(other.decay_rate)),
        division_min(std::move(other.division_min)), rho(std::move(other.rho)),
        rep_num(std::move(other.rep_num)) {}

  LinalgSolver<T, M, K, V, W> &operator=(LinalgSolver<T, M, K, V, W> &&other) {
    if (this != &other) {
      this->X_1 = std::move(other.X_1);
      this->decay_rate = std::move(other.decay_rate);
      this->division_min = std::move(other.division_min);
      this->rho = std::move(other.rho);
      this->rep_num = std::move(other.rep_num);
    }
    return *this;
  }

  /* Solve function */
  auto solve(const Matrix<DefDense, T, M, M> &A,
             const Matrix<DefDense, T, M, K> &B) -> Matrix<DefDense, T, M, K> {

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, M> x = Base::Matrix::gmres_k(
          A.matrix, B.matrix.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, M, K>(X_1);
  }

  auto solve(const Matrix<DefDense, T, M, M> &A, const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, M, M> {

    for (std::size_t i = 0; i < M; i++) {
      Base::Matrix::Vector<T, M> x = Base::Matrix::gmres_k(
          A.matrix, B.matrix.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, M, M>(X_1);
  }

  auto solve(const Matrix<DefDense, T, M, M> &A,
             const Matrix<DefSparse, T, M, K, V> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense = B.matrix.create_dense();

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, M> x = Base::Matrix::gmres_k(
          A.matrix, B_dense.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, M, K>(X_1);
  }

  auto solve(const Matrix<DefDiag, T, M> &A, const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, M, K> {

    X_1 = Base::Matrix::diag_inv_multiply_dense(A.matrix, B.matrix);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  auto solve(const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDiag, T, M> {

    Base::Matrix::DiagMatrix<T, M> result =
        Base::Matrix::diag_divide_diag(B.matrix, A.matrix, this->division_min);

    X_1 = result.create_dense();

    return Matrix<DefDiag, T, M>(std::move(result));
  }

  auto solve(const Matrix<DefDiag, T, M> &A,
             const Matrix<DefSparse, T, M, K, V> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense = B.matrix.create_dense();

    X_1 = Base::Matrix::diag_inv_multiply_dense(A.matrix, B_dense);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  auto solve(const Matrix<DefSparse, T, M, M, V> &A,
             const Matrix<DefDense, T, M, K> &B) -> Matrix<DefDense, T, M, K> {

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, M> x = Base::Matrix::sparse_gmres_k(
          A.matrix, B.matrix.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, M, K>(X_1);
  }

  auto solve(const Matrix<DefSparse, T, M, M, V> &A,
             const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, M, M> {

    Base::Matrix::Matrix<T, M, M> B_dense = B.matrix.create_dense();

    for (std::size_t i = 0; i < M; i++) {
      Base::Matrix::Vector<T, M> x = Base::Matrix::sparse_gmres_k(
          A.matrix, B_dense.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, M, M>(X_1);
  }

  auto solve(const Matrix<DefSparse, T, M, M, V> &A,
             const Matrix<DefSparse, T, M, K, W> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense = B.matrix.create_dense();

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, M> x = Base::Matrix::sparse_gmres_k(
          A.matrix, B_dense.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, M, K>(X_1);
  }

  /* Inv function */
  auto inv(const Matrix<DefDense, T, M, M> &A) -> Matrix<DefDense, T, M, M> {

    X_1 = Base::Matrix::gmres_k_matrix_inv(A.matrix, this->decay_rate,
                                           this->division_min, X_1);

    return Matrix<DefDense, T, M, M>(X_1);
  }

  auto inv(const Matrix<DefDiag, T, M> &A) -> Matrix<DefDiag, T, M> {

    Base::Matrix::DiagMatrix<T, M> result = A.matrix.inv(this->division_min);

    X_1 = result.create_dense();

    return Matrix<DefDiag, T, M>(std::move(result));
  }

  auto inv(const Matrix<DefSparse, T, M, M, V> &A)
      -> Matrix<DefDense, T, M, M> {

    X_1 = Base::Matrix::sparse_gmres_k_matrix_inv(A.matrix, this->decay_rate,
                                                  this->division_min, X_1);

    return Matrix<DefDense, T, M, M>(X_1);
  }

  auto get_answer(void) -> Matrix<DefDense, T, M, K> {
    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /* Variable */
  Base::Matrix::Matrix<T, M, K> X_1;

  T decay_rate = static_cast<T>(0);
  T division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
  std::array<T, K> rho;
  std::array<std::size_t, K> rep_num;
};

/* make LinalgSolver for inv */
template <typename T, std::size_t M, std::size_t K = 1, std::size_t V = 1,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolver<T, M, M, V, W> {

  return LinalgSolver<T, M, M, V, W>(A);
}

template <typename T, std::size_t M, std::size_t K = 1, std::size_t V = 1,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDiag, T, M> &A)
    -> LinalgSolver<T, M, M, V, W> {

  return LinalgSolver<T, M, M, V, W>(A);
}

template <typename T, std::size_t M, std::size_t K = 1, std::size_t V,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefSparse, T, M, M, V> &A)
    -> LinalgSolver<T, M, M, V, W> {

  return LinalgSolver<T, M, M, V, W>(A);
}

/* make LinalgSolver */
template <typename T, std::size_t M, std::size_t K, std::size_t V = 1,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDense, T, M, M> &A,
                       const Matrix<DefDense, T, M, K> &B)
    -> LinalgSolver<T, M, K, V, W> {

  return LinalgSolver<T, M, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K = 1, std::size_t V = 1,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDense, T, M, M> &A,
                       const Matrix<DefDiag, T, M> &B)
    -> LinalgSolver<T, M, M, V, W> {

  return LinalgSolver<T, M, M, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K, std::size_t V,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDense, T, M, M> &A,
                       const Matrix<DefSparse, T, M, K, V> &B)
    -> LinalgSolver<T, M, K, V, W> {

  return LinalgSolver<T, M, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K, std::size_t V = 1,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDiag, T, M> &A,
                       const Matrix<DefDense, T, M, K> &B)
    -> LinalgSolver<T, M, K, V, W> {

  return LinalgSolver<T, M, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K = 1, std::size_t V = 1,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDiag, T, M> &A,
                       const Matrix<DefDiag, T, M> &B)
    -> LinalgSolver<T, M, M, V, W> {

  return LinalgSolver<T, M, M, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K, std::size_t V,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefDiag, T, M> &A,
                       const Matrix<DefSparse, T, M, K, V> &B)
    -> LinalgSolver<T, M, K, V, W> {

  return LinalgSolver<T, M, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K, std::size_t V,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefSparse, T, M, M, V> &A,
                       const Matrix<DefDense, T, M, K> &B)
    -> LinalgSolver<T, M, K, V, W> {

  return LinalgSolver<T, M, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K = 1, std::size_t V,
          std::size_t W = 1>
auto make_LinalgSolver(const Matrix<DefSparse, T, M, M, V> &A,
                       const Matrix<DefDiag, T, M> &B)
    -> LinalgSolver<T, M, M, V, W> {

  return LinalgSolver<T, M, M, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t K, std::size_t V,
          std::size_t W>
auto make_LinalgSolver(const Matrix<DefSparse, T, M, M, V> &A,
                       const Matrix<DefSparse, T, M, K, W> &B)
    -> LinalgSolver<T, M, K, V, W> {

  return LinalgSolver<T, M, K, V, W>(A, B);
}

/* least-squares solution to a linear matrix equation */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
class LinalgLstsqSolver {
public:
  /* Constructor */
  LinalgLstsqSolver() {}

  LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefSparse, T, M, K, V> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, V> &A,
                    const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, V> &A,
                    const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, V> &A,
                    const Matrix<DefSparse, T, M, K, W> &B) {

    this->solve(A, B);
  }

  /* Copy Constructor */
  LinalgLstsqSolver(const LinalgLstsqSolver<T, M, N, K, V, W> &other)
      : X_1(other.X_1), decay_rate(other.decay_rate),
        division_min(other.division_min), rho(other.rho),
        rep_num(other.rep_num) {}

  LinalgLstsqSolver<T, M, N, K, V, W> &
  operator=(const LinalgLstsqSolver<T, M, N, K, V, W> &other) {
    if (this != &other) {
      this->X_1 = other.X_1;
      this->decay_rate = other.decay_rate;
      this->division_min = other.division_min;
      this->rho = other.rho;
      this->rep_num = other.rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgLstsqSolver(LinalgLstsqSolver<T, M, N, K, V, W> &&other) noexcept
      : X_1(std::move(other.X_1)), decay_rate(std::move(other.decay_rate)),
        division_min(std::move(other.division_min)), rho(std::move(other.rho)),
        rep_num(std::move(other.rep_num)) {}

  LinalgLstsqSolver<T, M, N, K, V, W> &
  operator=(LinalgLstsqSolver<T, M, N, K, V, W> &&other) {
    if (this != &other) {
      this->X_1 = std::move(other.X_1);
      this->decay_rate = std::move(other.decay_rate);
      this->division_min = std::move(other.division_min);
      this->rho = std::move(other.rho);
      this->rep_num = std::move(other.rep_num);
    }
    return *this;
  }

  /* Solve method */
  auto solve(const Matrix<DefDense, T, M, N> &A,
             const Matrix<DefDense, T, M, K> &B) -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, N> x = Base::Matrix::gmres_k_rect(
          A.matrix, B.matrix.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, N, K>(X_1);
  }

  auto solve(const Matrix<DefDense, T, M, N> &A, const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, N, M> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, N> x = Base::Matrix::gmres_k_rect(
          A.matrix, B.matrix.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, N, M>(X_1);
  }

  auto solve(const Matrix<DefDense, T, M, N> &A,
             const Matrix<DefSparse, T, M, K, V> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::Matrix<T, M, K> B_dense = B.matrix.create_dense();

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, N> x = Base::Matrix::gmres_k_rect(
          A.matrix, B_dense.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, N, K>(X_1);
  }

  auto solve(const Matrix<DefSparse, T, M, N, V> &A,
             const Matrix<DefDense, T, M, K> &B) -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, N> x = Base::Matrix::sparse_gmres_k_rect(
          A.matrix, B.matrix.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, N, K>(X_1);
  }

  auto solve(const Matrix<DefSparse, T, M, N, V> &A,
             const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, N, M> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, N> x = Base::Matrix::sparse_gmres_k_rect(
          A.matrix, B.matrix.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, N, M>(X_1);
  }

  auto solve(const Matrix<DefSparse, T, M, N, V> &A,
             const Matrix<DefSparse, T, M, K, W> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::Matrix<T, M, K> B_dense = B.matrix.create_dense();

    for (std::size_t i = 0; i < K; i++) {
      Base::Matrix::Vector<T, N> x = Base::Matrix::sparse_gmres_k_rect(
          A.matrix, B_dense.get_row(i), this->X_1.get_row(i), this->decay_rate,
          this->division_min, this->rho[i], this->rep_num[i]);
      this->X_1.set_row(i, x);
    }

    return Matrix<DefDense, T, N, K>(X_1);
  }

  auto get_answer(void) -> Matrix<DefDense, T, N, K> {
    return Matrix<DefDense, T, N, K>(this->X_1);
  }

  /* Properties */
  Base::Matrix::Matrix<T, N, K> X_1;

  T decay_rate = static_cast<T>(0);
  T division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
  std::array<T, K> rho;
  std::array<std::size_t, K> rep_num;
};

/* make LinalgLstsqSolver */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V = 1, std::size_t W = 1>
auto make_LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                            const Matrix<DefDense, T, M, K> &B)
    -> LinalgLstsqSolver<T, M, N, K, V, W> {

  return LinalgLstsqSolver<T, M, N, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K = 1,
          std::size_t V = 1, std::size_t W = 1>
auto make_LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                            const Matrix<DefDiag, T, M> &B)
    -> LinalgLstsqSolver<T, M, N, M, V, W> {

  return LinalgLstsqSolver<T, M, N, M, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W = 1>
auto make_LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                            const Matrix<DefSparse, T, M, K, V> &B)
    -> LinalgLstsqSolver<T, M, N, K, V, W> {

  return LinalgLstsqSolver<T, M, N, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W = 1>
auto make_LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, V> &A,
                            const Matrix<DefDense, T, M, K> &B)
    -> LinalgLstsqSolver<T, M, N, K, V, W> {

  return LinalgLstsqSolver<T, M, N, K, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K = 1,
          std::size_t V, std::size_t W = 1>
auto make_LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, V> &A,
                            const Matrix<DefDiag, T, M> &B)
    -> LinalgLstsqSolver<T, M, N, M, V, W> {

  return LinalgLstsqSolver<T, M, N, M, V, W>(A, B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
auto make_LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, V> &A,
                            const Matrix<DefSparse, T, M, K, W> &B)
    -> LinalgLstsqSolver<T, M, N, K, V, W> {

  return LinalgLstsqSolver<T, M, N, K, V, W>(A, B);
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_LINALG_SOLVER_HPP
