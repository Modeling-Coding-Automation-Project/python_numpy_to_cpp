#ifndef __PYTHON_NUMPY_LINALG_SOLVER_HPP__
#define __PYTHON_NUMPY_LINALG_SOLVER_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_SOLVER = 1.0e-23;

/* Inverse */
template <typename T, std::size_t M, std::size_t K, bool IsComplex>
struct inverse {};

template <typename T, std::size_t M, std::size_t K>
struct inverse<T, M, K, true> {
  static auto compute(const Matrix<DefDense, T, M, M> &A, T decay_rate,
                      T division_min, Base::Matrix::Matrix<T, M, K> &X_1)
      -> Matrix<DefDense, T, M, M> {

    // X_1 = Base::Matrix::gmres_k_matrix_inv(A.matrix, decay_rate,
    //                                        division_min, X_1);

    return Matrix<DefDense, T, M, M>(X_1);
  }
};

template <typename T, std::size_t M, std::size_t K>
struct inverse<T, M, K, false> {
  static auto compute(const Matrix<DefDense, T, M, M> &A, T decay_rate,
                      T division_min, Base::Matrix::Matrix<T, M, K> &X_1)
      -> Matrix<DefDense, T, M, M> {

    X_1 = Base::Matrix::gmres_k_matrix_inv(A.matrix, decay_rate, division_min,
                                           X_1);

    return Matrix<DefDense, T, M, M>(X_1);
  }
};

/* Linalg Solver */
template <typename T, std::size_t M, std::size_t K, typename SparseAvailable_A,
          typename SparseAvailable_B>
class LinalgSolver {
public:
  /* Type */
  using Value_Type = T;

public:
  /* Constructor */
  LinalgSolver() {}

  LinalgSolver(const Matrix<DefDense, T, M, M> &A) { this->inv(A); }

  LinalgSolver(const Matrix<DefDiag, T, M> &A) { this->inv(A); }

  LinalgSolver(const Matrix<DefSparse, T, M, K, SparseAvailable_A> &A) {
    this->inv(A);
  }

  LinalgSolver(const Matrix<DefDense, T, M, M> &A,
               const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefDense, T, M, M> &A,
               const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefDense, T, M, M> &A,
               const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B) {

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
               const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
               const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
               const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgSolver(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
               const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B) {

    this->solve(A, B);
  }

  /* Copy Constructor */
  LinalgSolver(
      const LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &other)
      : X_1(other.X_1), decay_rate(other.decay_rate),
        division_min(other.division_min), rho(other.rho),
        rep_num(other.rep_num) {}

  LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(const LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>
                &other) {
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
  LinalgSolver(LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>
                   &&other) noexcept
      : X_1(std::move(other.X_1)), decay_rate(std::move(other.decay_rate)),
        division_min(std::move(other.division_min)), rho(std::move(other.rho)),
        rep_num(std::move(other.rep_num)) {}

  LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &operator=(
      LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &&other) {
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
  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::gmres_k_matrix(A.matrix, B.matrix, this->X_1,
                                 this->decay_rate, this->division_min,
                                 this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, M, M> {

    Base::Matrix::gmres_k_matrix(A.matrix, B.matrix, this->X_1,
                                 this->decay_rate, this->division_min,
                                 this->rho, this->rep_num);

    return Matrix<DefDense, T, M, M>(X_1);
  }

  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::gmres_k_matrix(A.matrix, B_dense_matrix, this->X_1,
                                 this->decay_rate, this->division_min,
                                 this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, M, K> {

    X_1 = Base::Matrix::diag_inv_multiply_dense(A.matrix, B.matrix,
                                                this->division_min);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

    Base::Matrix::DiagMatrix<T, M> result =
        Base::Matrix::diag_divide_diag(B.matrix, A.matrix, this->division_min);

    X_1 = Base::Matrix::output_dense_matrix(result);

    return Matrix<DefDiag, T, M>(std::move(result));
  }

  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense =
        Base::Matrix::output_dense_matrix(B.matrix);

    X_1 = Base::Matrix::diag_inv_multiply_dense(A.matrix, B_dense,
                                                this->division_min);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::sparse_gmres_k_matrix(A.matrix, B.matrix, this->X_1,
                                        this->decay_rate, this->division_min,
                                        this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, M, M> {

    Base::Matrix::Matrix<T, M, M> B_dense_matrix = B.matrix.create_dense();

    Base::Matrix::sparse_gmres_k_matrix(A.matrix, B_dense_matrix, this->X_1,
                                        this->decay_rate, this->division_min,
                                        this->rho, this->rep_num);

    return Matrix<DefDense, T, M, M>(X_1);
  }

  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_matrix(A.matrix, B_dense_matrix, this->X_1,
                                        this->decay_rate, this->division_min,
                                        this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(X_1);
  }

  /* Inv function */
  inline auto inv(const Matrix<DefDense, T, M, M> &A)
      -> Matrix<DefDense, T, M, M> {

    return inverse<T, M, K, IS_COMPLEX>::compute(A, this->decay_rate,
                                                 this->division_min, X_1);
  }

  inline auto inv(const Matrix<DefDiag, T, M> &A) -> Matrix<DefDiag, T, M> {

    Base::Matrix::DiagMatrix<T, M> result = A.matrix.inv(this->division_min);

    X_1 = Base::Matrix::output_dense_matrix(result);

    return Matrix<DefDiag, T, M>(std::move(result));
  }

  inline auto inv(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A)
      -> Matrix<DefDense, T, M, M> {

    X_1 = Base::Matrix::sparse_gmres_k_matrix_inv(A.matrix, this->decay_rate,
                                                  this->division_min, X_1);

    return Matrix<DefDense, T, M, M>(X_1);
  }

  inline auto get_answer(void) -> Matrix<DefDense, T, M, K> {
    return Matrix<DefDense, T, M, K>(this->X_1);
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = K;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::Matrix<T, M, K> X_1;

  T decay_rate = static_cast<T>(0);
  T division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
  std::array<T, K> rho;
  std::array<std::size_t, K> rep_num;
};

template <typename T, std::size_t M> class LinalgInvDiag {
public:
  /* Type */
  using Value_Type = T;

public:
  /* Constructor */
  LinalgInvDiag() {}

  LinalgInvDiag(const Matrix<DefDiag, T, M> &A) { this->inv(A); }

  /* Copy Constructor */
  LinalgInvDiag(const LinalgInvDiag<T, M> &other)
      : X_1(other.X_1), division_min(other.division_min) {}

  LinalgInvDiag<T, M> &operator=(const LinalgInvDiag<T, M> &other) {
    if (this != &other) {
      this->X_1 = other.X_1;
      this->division_min = other.division_min;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgInvDiag(LinalgInvDiag<T, M> &&other) noexcept
      : X_1(std::move(other.X_1)), division_min(std::move(other.division_min)) {
  }

  LinalgInvDiag<T, M> &operator=(LinalgInvDiag<T, M> &&other) {
    if (this != &other) {
      this->X_1 = std::move(other.X_1);
      this->division_min = std::move(other.division_min);
    }
    return *this;
  }

public:
  /* Function */
  inline auto inv(const Matrix<DefDiag, T, M> &A) -> Matrix<DefDiag, T, M> {

    X_1 = A.matrix.inv(this->division_min);

    return Matrix<DefDiag, T, M>(std::move(X_1));
  }

  inline auto get_answer(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(this->X_1);
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

public:
  /* Variable */
  Base::Matrix::DiagMatrix<T, M> X_1;

  T division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
};

/* make LinalgSolver for inv */
template <typename T, std::size_t M, std::size_t K = 1,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, M>,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgSolver(const Matrix<DefDense, T, M, M> &A)
    -> LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B>(A);
}

template <typename T, std::size_t M>
inline auto make_LinalgSolver(const Matrix<DefDiag, T, M> &A)
    -> LinalgInvDiag<T, M> {

  return LinalgInvDiag<T, M>(A);
}

template <typename T, std::size_t M, std::size_t K = 1,
          typename SparseAvailable_A,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, M>>
inline auto
make_LinalgSolver(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A)
    -> LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B>(A);
}

/* make LinalgSolver */
template <typename T, std::size_t M, std::size_t K,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, K>,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, K>>
inline auto make_LinalgSolver(const Matrix<DefDense, T, M, M> &A,
                              const Matrix<DefDense, T, M, K> &B)
    -> LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K = 1,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, M>,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgSolver(const Matrix<DefDense, T, M, M> &A,
                              const Matrix<DefDiag, T, M> &B)
    -> LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, K>,
          typename SparseAvailable_B>
inline auto
make_LinalgSolver(const Matrix<DefDense, T, M, M> &A,
                  const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
    -> LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, K>,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, K>>
inline auto make_LinalgSolver(const Matrix<DefDiag, T, M> &A,
                              const Matrix<DefDense, T, M, K> &B)
    -> LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K = 1,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, M>,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgSolver(const Matrix<DefDiag, T, M> &A,
                              const Matrix<DefDiag, T, M> &B)
    -> LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, K>,
          typename SparseAvailable_B>
inline auto
make_LinalgSolver(const Matrix<DefDiag, T, M> &A,
                  const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
    -> LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K, typename SparseAvailable_A,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, K>>
inline auto
make_LinalgSolver(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                  const Matrix<DefDense, T, M, K> &B)
    -> LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K = 1,
          typename SparseAvailable_A,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, M>>
inline auto
make_LinalgSolver(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                  const Matrix<DefDiag, T, M> &B)
    -> LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, M, SparseAvailable_A, SparseAvailable_B>(A, B);
}

template <typename T, std::size_t M, std::size_t K, typename SparseAvailable_A,
          typename SparseAvailable_B>
inline auto
make_LinalgSolver(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                  const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
    -> LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>(A, B);
}

/* least-squares solution to a linear matrix equation */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A, typename SparseAvailable_B>
class LinalgLstsqSolver {
public:
  /* Type */
  using Value_Type = T;
  using SparseAvailable_A_Type = SparseAvailable_A;
  using SparseAvailable_B_Type = SparseAvailable_B;

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
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefDense, T, M, K> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefDiag, T, M> &B) {

    this->solve(A, B);
  }

  LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B) {

    this->solve(A, B);
  }

  /* Copy Constructor */
  LinalgLstsqSolver(const LinalgLstsqSolver<T, M, N, K, SparseAvailable_A,
                                            SparseAvailable_B> &other)
      : X_1(other.X_1), decay_rate(other.decay_rate),
        division_min(other.division_min), rho(other.rho),
        rep_num(other.rep_num) {}

  LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(const LinalgLstsqSolver<T, M, N, K, SparseAvailable_A,
                                    SparseAvailable_B> &other) {
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
  LinalgLstsqSolver(LinalgLstsqSolver<T, M, N, K, SparseAvailable_A,
                                      SparseAvailable_B> &&other) noexcept
      : X_1(std::move(other.X_1)), decay_rate(std::move(other.decay_rate)),
        division_min(std::move(other.division_min)), rho(std::move(other.rho)),
        rep_num(std::move(other.rep_num)) {}

  LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B>
                &&other) {
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
  inline auto solve(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::gmres_k_rect_matrix(A.matrix, B.matrix, this->X_1,
                                      this->decay_rate, this->division_min,
                                      this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  inline auto solve(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, N, M> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::gmres_k_rect_matrix(A.matrix, B.matrix, this->X_1,
                                      this->decay_rate, this->division_min,
                                      this->rho, this->rep_num);

    return Matrix<DefDense, T, N, M>(X_1);
  }

  inline auto solve(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::gmres_k_rect_matrix(A.matrix, B_dense_matrix, this->X_1,
                                      this->decay_rate, this->division_min,
                                      this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  inline auto solve(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::sparse_gmres_k_rect_matrix(
        A.matrix, B.matrix, this->X_1, this->decay_rate, this->division_min,
        this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  inline auto solve(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, N, M> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::sparse_gmres_k_rect_matrix(
        A.matrix, B.matrix, this->X_1, this->decay_rate, this->division_min,
        this->rho, this->rep_num);

    return Matrix<DefDense, T, N, M>(X_1);
  }

  inline auto solve(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_rect_matrix(
        A.matrix, B_dense_matrix, this->X_1, this->decay_rate,
        this->division_min, this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  inline auto get_answer(void) -> Matrix<DefDense, T, N, K> {
    return Matrix<DefDense, T, N, K>(this->X_1);
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = N;
  static constexpr std::size_t ROWS = K;

public:
  /* Properties */
  Base::Matrix::Matrix<T, N, K> X_1;

  T decay_rate = static_cast<T>(0);
  T division_min = static_cast<T>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
  std::array<T, K> rho;
  std::array<std::size_t, K> rep_num;
};

/* make LinalgLstsqSolver */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, N>,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, N>>
inline auto make_LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                                   const Matrix<DefDense, T, M, K> &B)
    -> LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B>(A,
                                                                             B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K = 1,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, N>,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, M>>
inline auto make_LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                                   const Matrix<DefDiag, T, M> &B)
    -> LinalgLstsqSolver<T, M, N, M, SparseAvailable_A, SparseAvailable_B> {

  return LinalgLstsqSolver<T, M, N, M, SparseAvailable_A, SparseAvailable_B>(A,
                                                                             B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A = SparseAvailable_NoUse<M, N>,
          typename SparseAvailable_B>
inline auto
make_LinalgLstsqSolver(const Matrix<DefDense, T, M, N> &A,
                       const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
    -> LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B>(A,
                                                                             B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, N>>
inline auto
make_LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefDense, T, M, K> &B)
    -> LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B>(A,
                                                                             B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K = 1,
          typename SparseAvailable_A,
          typename SparseAvailable_B = SparseAvailable_NoUse<M, N>>
inline auto
make_LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefDiag, T, M> &B)
    -> LinalgLstsqSolver<T, M, N, M, SparseAvailable_A, SparseAvailable_B> {

  return LinalgLstsqSolver<T, M, N, M, SparseAvailable_A, SparseAvailable_B>(A,
                                                                             B);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
make_LinalgLstsqSolver(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
    -> LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> {

  return LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B>(A,
                                                                             B);
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_SOLVER_HPP__
