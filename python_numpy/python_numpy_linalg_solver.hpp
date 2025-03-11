#ifndef __PYTHON_NUMPY_LINALG_SOLVER_HPP__
#define __PYTHON_NUMPY_LINALG_SOLVER_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_SOLVER = 1.0e-23;

namespace InverseOperation {

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          bool IsComplex>
struct InverseDense {};

template <typename T, typename Complex_T, std::size_t M, std::size_t K>
struct InverseDense<T, Complex_T, M, K, true> {
  static auto compute(const Matrix<DefDense, Complex_T, M, M> &A,
                      const T &decay_rate, const T &division_min,
                      std::array<T, K> &rho,
                      std::array<std::size_t, K> &rep_num,
                      Base::Matrix::Matrix<Complex_T, M, K> &X_1)
      -> Matrix<DefDense, Complex_T, M, M> {

    X_1 = Base::Matrix::complex_gmres_k_matrix_inv(
        A.matrix, decay_rate, division_min, rho, rep_num, X_1);

    return Matrix<DefDense, Complex_T, M, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t K>
struct InverseDense<T, Complex_T, M, K, false> {
  static auto compute(const Matrix<DefDense, T, M, M> &A, const T &decay_rate,
                      const T &division_min, std::array<T, K> &rho,
                      std::array<std::size_t, K> &rep_num,
                      Base::Matrix::Matrix<T, M, K> &X_1)
      -> Matrix<DefDense, T, M, M> {

    X_1 = Base::Matrix::gmres_k_matrix_inv(A.matrix, decay_rate, division_min,
                                           rho, rep_num, X_1);

    return Matrix<DefDense, T, M, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, bool IsComplex>
struct InverseDiag {};

template <typename T, typename Complex_T, std::size_t M>
struct InverseDiag<T, Complex_T, M, true> {
  static auto compute(const Matrix<DefDiag, Complex_T, M> &A,
                      const T &division_min,
                      Base::Matrix::DiagMatrix<Complex_T, M> &X_1)
      -> Matrix<DefDiag, Complex_T, M> {

    X_1 = Base::Matrix::inverse_complex_diag_matrix(A.matrix, division_min);

    return Matrix<DefDiag, Complex_T, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M>
struct InverseDiag<T, Complex_T, M, false> {
  static auto compute(const Matrix<DefDiag, T, M> &A, const T &division_min,
                      Base::Matrix::DiagMatrix<T, M> &X_1)
      -> Matrix<DefDiag, T, M> {

    X_1 = Base::Matrix::inverse_diag_matrix(A.matrix, division_min);

    return Matrix<DefDiag, T, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          typename SparseAvailable, bool IsComplex>
struct InverseSparse {};

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          typename SparseAvailable>
struct InverseSparse<T, Complex_T, M, K, SparseAvailable, true> {
  static auto
  compute(const Matrix<DefSparse, Complex_T, M, M, SparseAvailable> &A,
          const T &decay_rate, const T &division_min, std::array<T, K> &rho,
          std::array<std::size_t, K> &rep_num,
          Base::Matrix::Matrix<Complex_T, M, K> &X_1)
      -> Matrix<DefDense, Complex_T, M, M> {

    X_1 = Base::Matrix::complex_sparse_gmres_k_matrix_inv(
        A.matrix, decay_rate, division_min, rho, rep_num, X_1);

    return Matrix<DefDense, Complex_T, M, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          typename SparseAvailable>
struct InverseSparse<T, Complex_T, M, K, SparseAvailable, false> {
  static auto compute(const Matrix<DefSparse, T, M, M, SparseAvailable> &A,
                      const T &decay_rate, const T &division_min,
                      std::array<T, K> &rho,
                      std::array<std::size_t, K> &rep_num,
                      Base::Matrix::Matrix<T, M, K> &X_1)
      -> Matrix<DefDense, Complex_T, M, M> {

    X_1 = Base::Matrix::sparse_gmres_k_matrix_inv(
        A.matrix, decay_rate, division_min, rho, rep_num, X_1);

    return Matrix<DefDense, Complex_T, M, M>(X_1);
  }
};

} // namespace InverseOperation

/* Linalg Solver */
template <typename T, std::size_t M, std::size_t K, typename SparseAvailable_A,
          typename SparseAvailable_B>
class LinalgSolver {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

public:
  /* Constructor */
  LinalgSolver()
      : X_1(), decay_rate(static_cast<Value_Type>(0)),
        division_min(
            static_cast<Value_Type>(DEFAULT_DIVISION_MIN_LINALG_SOLVER)),
        rho({}), rep_num({}) {}

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

    return InverseOperation::InverseDense<
        Value_Type, T, M, K, IS_COMPLEX>::compute(A, this->decay_rate,
                                                  this->division_min, this->rho,
                                                  this->rep_num, X_1);
  }

  inline auto inv(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A)
      -> Matrix<DefDense, T, M, M> {

    return InverseOperation::InverseSparse<
        Value_Type, T, M, K, SparseAvailable_A,
        IS_COMPLEX>::compute(A, this->decay_rate, this->division_min, this->rho,
                             this->rep_num, X_1);
  }

  inline auto get_answer(void) -> Matrix<DefDense, T, M, K> {
    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  inline void set_decay_rate(const Value_Type &decay_rate_in) {
    this->decay_rate = decay_rate_in;
  }

  inline void set_division_min(const Value_Type &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = K;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::Matrix<T, M, K> X_1;

  Value_Type decay_rate = static_cast<Value_Type>(0);
  Value_Type division_min =
      static_cast<Value_Type>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
  std::array<Value_Type, K> rho;
  std::array<std::size_t, K> rep_num;
};

template <typename T, std::size_t M> class LinalgInvDiag {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

public:
  /* Constructor */
  LinalgInvDiag() {}

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

    return InverseOperation::InverseDiag<Value_Type, T, M, IS_COMPLEX>::compute(
        A, this->division_min, X_1);
  }

  inline auto get_answer(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(this->X_1);
  }

  inline void set_division_min(const T &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::DiagMatrix<T, M> X_1;

  Value_Type division_min =
      static_cast<Value_Type>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
};

/* make LinalgSolver for inv */
template <
    typename A_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverInv(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename A_Type::SparseAvailable_Type> {

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename A_Type::SparseAvailable_Type>();
}

template <typename A_Type, typename std::enable_if<
                               Is_Diag_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverInv(void)
    -> LinalgInvDiag<typename A_Type::Value_Complex_Type, A_Type::COLS> {

  return LinalgInvDiag<typename A_Type::Value_Complex_Type, A_Type::COLS>();
}

template <
    typename A_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverInv(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename A_Type::SparseAvailable_Type> {

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolver for inv Type */
template <typename A_Type>
using LinalgSolverInv_Type = decltype(make_LinalgSolverInv<A_Type>());

/* make LinalgSolver */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/* LinalgSolver Type */
template <typename A_Type, typename B_Type>
using LinalgSolver_Type = decltype(make_LinalgSolver<A_Type, B_Type>());

/* least-squares solution to a linear matrix equation */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A, typename SparseAvailable_B>
class LinalgLstsqSolver {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

  using SparseAvailable_A_Type = SparseAvailable_A;
  using SparseAvailable_B_Type = SparseAvailable_B;

public:
  /* Constructor */
  LinalgLstsqSolver() {}

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

  inline void set_decay_rate(const Value_Type &decay_rate_in) {
    this->decay_rate = decay_rate_in;
  }

  inline void set_division_min(const Value_Type &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = N;
  static constexpr std::size_t ROWS = K;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

public:
  /* Properties */
  Base::Matrix::Matrix<T, N, K> X_1;

  Value_Type decay_rate = static_cast<Value_Type>(0);
  Value_Type division_min =
      static_cast<Value_Type>(DEFAULT_DIVISION_MIN_LINALG_SOLVER);
  std::array<Value_Type, K> rho;
  std::array<std::size_t, K> rep_num;
};

/* make LinalgLstsqSolver */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, A_Type::COLS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, A_Type::COLS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, A_Type::COLS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, A_Type::COLS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

/* LinalgLstsqSolver Type */
template <typename A_Type, typename B_Type>
using LinalgLstsqSolver_Type =
    decltype(make_LinalgLstsqSolver<A_Type, B_Type>());

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_SOLVER_HPP__
