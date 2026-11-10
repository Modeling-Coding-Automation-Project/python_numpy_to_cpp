/**
 * @file base_matrix_gmres.hpp
 * @brief Fixed-capacity restarted GMRES with real or complex arithmetic.
 */
#ifndef BASE_MATRIX_GMRES_HPP_
#define BASE_MATRIX_GMRES_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_numerical.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace Base {
namespace Matrix {
namespace GmresOperation {

template <typename T, std::size_t N>
inline void multiply(const Matrix<T, N, N> &matrix, const Vector<T, N> &input,
                     Vector<T, N> &output, std::size_t active_size) {
  for (std::size_t i = 0; i < N; ++i) {
    output[i] = T();
  }
  if (active_size == N) {
    // Matrix storage is column-major; visit each contiguous column once.
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < N; ++i) {
        output[i] += matrix(i, j) * input[j];
      }
    }
  } else {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < N; ++i) {
        if (i < active_size && j < active_size) {
          output[i] += matrix(i, j) * input[j];
        }
      }
    }
  }
}

template <typename T, std::size_t N, typename Indices, typename Pointers>
inline void
multiply(const CompiledSparseMatrix<T, N, N, Indices, Pointers> &matrix,
         const Vector<T, N> &input, Vector<T, N> &output,
         std::size_t active_size) {
  for (std::size_t i = 0; i < N; ++i) {
    output[i] = T();
    for (std::size_t j = Pointers::list[i]; j < Pointers::list[i + 1]; ++j) {
      if (i < active_size && Indices::list[j] < active_size) {
        output[i] += matrix.values[j] * input[Indices::list[j]];
      }
    }
  }
}

template <typename MatrixType, typename T, std::size_t N>
struct MatrixOperator {
  const MatrixType &matrix;
  std::size_t active_size;
  void operator()(const Vector<T, N> &input, Vector<T, N> &output) const {
    multiply(matrix, input, output, active_size);
  }
};

/** Keep a public callback inside the requested leading subspace. */
template <typename Operator, typename T, std::size_t N>
inline void apply_operator(const Operator &operation, const Vector<T, N> &input,
                           Vector<T, N> &output, std::size_t active_size) {
  if (active_size == N) {
    operation(input, output);
  } else {
    Vector<T, N> active_input;
    for (std::size_t i = 0; i < N; ++i) {
      if (i < active_size) {
        active_input[i] = input[i];
      }
    }
    operation(active_input, output);
    for (std::size_t i = 0; i < N; ++i) {
      if (i >= active_size) {
        output[i] = T();
      }
    }
  }
}

template <typename T> struct IdentityPreconditioner {
  T operator[](std::size_t) const { return static_cast<T>(1); }
};

/** Explicit normal equations are useful for tall inputs or repeated RHS. */
template <typename T, std::size_t M, std::size_t N>
inline void normal_equations(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &right_hand_side,
                             Matrix<T, N, N> &gram, Vector<T, N> &normal_rhs) {
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      gram(i, j) = T();
      for (std::size_t row = 0; row < M; ++row) {
        gram(i, j) += matrix(row, i) * matrix(row, j);
      }
    }
    normal_rhs[j] = T();
    for (std::size_t row = 0; row < M; ++row) {
      normal_rhs[j] += matrix(row, j) * right_hand_side[row];
    }
  }
}

template <typename T, std::size_t M, std::size_t N, typename I, typename P>
inline void normal_equations(const CompiledSparseMatrix<T, M, N, I, P> &matrix,
                             const Vector<T, M> &right_hand_side,
                             Matrix<T, N, N> &gram, Vector<T, N> &normal_rhs) {
  for (std::size_t j = 0; j < N; ++j) {
    normal_rhs[j] = T();
    for (std::size_t i = 0; i < N; ++i) {
      gram(i, j) = T();
    }
  }
  for (std::size_t row = 0; row < M; ++row) {
    for (std::size_t first = P::list[row]; first < P::list[row + 1]; ++first) {
      normal_rhs[I::list[first]] += matrix.values[first] * right_hand_side[row];
      for (std::size_t second = P::list[row]; second < P::list[row + 1];
           ++second) {
        gram(I::list[first], I::list[second]) +=
            matrix.values[first] * matrix.values[second];
      }
    }
  }
}

/** Evaluate A^T(b-Ax) directly, including for an explicitly formed Gram. */
template <typename T, std::size_t M, std::size_t N>
inline T normal_residual(const Matrix<T, M, N> &matrix,
                         const Vector<T, M> &right_hand_side,
                         const Vector<T, N> &solution) {
  Vector<T, N> stationarity;
  for (std::size_t row = 0; row < M; ++row) {
    T residual = right_hand_side[row];
    for (std::size_t j = 0; j < N; ++j) {
      residual -= matrix(row, j) * solution[j];
    }
    for (std::size_t j = 0; j < N; ++j) {
      stationarity[j] += matrix(row, j) * residual;
    }
  }
  return Numerical::norm(stationarity);
}

template <typename T, std::size_t M, std::size_t N, typename I, typename P>
inline T normal_residual(const CompiledSparseMatrix<T, M, N, I, P> &matrix,
                         const Vector<T, M> &right_hand_side,
                         const Vector<T, N> &solution) {
  Vector<T, N> stationarity;
  for (std::size_t row = 0; row < M; ++row) {
    T residual = right_hand_side[row];
    for (std::size_t entry = P::list[row]; entry < P::list[row + 1]; ++entry) {
      residual -= matrix.values[entry] * solution[I::list[entry]];
    }
    for (std::size_t entry = P::list[row]; entry < P::list[row + 1]; ++entry) {
      stationarity[I::list[entry]] += matrix.values[entry] * residual;
    }
  }
  return Numerical::norm(stationarity);
}

/** The normal operator avoids storing A^T A; it is still GMRES on that system.
 */
template <typename MatrixType, typename T, std::size_t M, std::size_t N>
struct NormalOperator;

template <typename T, std::size_t M, std::size_t N>
struct NormalOperator<Matrix<T, M, N>, T, M, N> {
  const Matrix<T, M, N> &matrix;
  void transpose_multiply(const Vector<T, M> &input,
                          Vector<T, N> &output) const {
    for (std::size_t j = 0; j < N; ++j) {
      output[j] = T();
      for (std::size_t i = 0; i < M; ++i) {
        output[j] += matrix(i, j) * input[i];
      }
    }
  }
  void operator()(const Vector<T, N> &input, Vector<T, N> &output) const {
    Vector<T, M> intermediate;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        intermediate[i] += matrix(i, j) * input[j];
      }
    }
    transpose_multiply(intermediate, output);
  }
};

template <typename T, std::size_t M, std::size_t N, typename I, typename P>
struct NormalOperator<CompiledSparseMatrix<T, M, N, I, P>, T, M, N> {
  const CompiledSparseMatrix<T, M, N, I, P> &matrix;
  void transpose_multiply(const Vector<T, M> &input,
                          Vector<T, N> &output) const {
    for (std::size_t j = 0; j < N; ++j) {
      output[j] = T();
    }
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = P::list[i]; j < P::list[i + 1]; ++j) {
        output[I::list[j]] += matrix.values[j] * input[i];
      }
    }
  }
  void operator()(const Vector<T, N> &input, Vector<T, N> &output) const {
    Vector<T, M> intermediate;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = P::list[i]; j < P::list[i + 1]; ++j) {
        intermediate[i] += matrix.values[j] * input[I::list[j]];
      }
    }
    transpose_multiply(intermediate, output);
  }
};

} // namespace GmresOperation

/**
 * @brief Reusable GMRES workspace; K controls storage, cycles controls
 * restarts.
 *
 * All cycle/Arnoldi slots are visited; completed slots skip numerical updates.
 * Iteration counts report actual Arnoldi updates, not the configured slots.
 * rho is always the norm of the true final residual (also for complex inputs).
 * No non-finite checks are made here. Products and final norms must fit Real.
 */
template <typename T, std::size_t N, std::size_t K = N> class GmresSolver {
public:
  using Real = typename Numerical::Scalar<T>::Real;
  static_assert(K > 0 && K <= N, "GMRES requires 1 <= K <= N.");
  static_assert(std::is_floating_point<Real>::value,
                "GMRES requires floating-point real components.");

  std::size_t cycles = 1;
  std::size_t krylov_dimension = K;
  std::size_t orthogonalization_passes = 1;
  Real division_min = static_cast<Real>(1.0e-10);
  Real relative_tolerance = static_cast<Real>(0);
  Real absolute_tolerance = static_cast<Real>(0);

  template <typename MatrixType>
  Vector<T, N> solve(const MatrixType &matrix,
                     const Vector<T, N> &right_hand_side,
                     const Vector<T, N> &initial, std::size_t active_size = N) {
    active_size = active_size > N ? N : active_size;
    GmresOperation::MatrixOperator<MatrixType, T, N> operation = {matrix,
                                                                  active_size};
    return solve_operator(operation, right_hand_side, initial, active_size);
  }

  template <typename MatrixType>
  Vector<T, N> solve_preconditioned(const MatrixType &matrix,
                                    const Vector<T, N> &right_hand_side,
                                    const Vector<T, N> &initial,
                                    const Vector<T, N> &inverse_diagonal) {
    GmresOperation::MatrixOperator<MatrixType, T, N> operation = {matrix, N};
    return _solve(operation, right_hand_side, initial, inverse_diagonal, N);
  }

  template <typename Operator>
  Vector<T, N>
  solve_operator(const Operator &operation, const Vector<T, N> &right_hand_side,
                 const Vector<T, N> &initial, std::size_t active_size = N) {
    GmresOperation::IdentityPreconditioner<T> identity;
    return _solve(operation, right_hand_side, initial, identity,
                  active_size > N ? N : active_size);
  }

  Real get_residual_norm() const { return this->residual_norm_; }
  std::size_t get_iterations() const { return this->iterations_; }
  bool is_converged() const { return this->converged_; }
  bool has_breakdown() const { return this->breakdown_; }
  bool threshold_was_adjusted() const { return this->threshold_adjusted_; }

private:
  std::array<T, N * K> basis_ = {};
  std::array<T, K *(K + 1) / 2> triangular_ = {};
  std::array<T, K + 1> column_ = {};
  std::array<T, K + 1> transformed_rhs_ = {};
  std::array<T, K> sine_ = {};
  std::array<Real, K> cosine_ = {};
  std::array<T, K> coefficients_ = {};
  Real residual_norm_ = static_cast<Real>(0);
  std::size_t iterations_ = 0;
  bool converged_ = false;
  bool breakdown_ = false;
  bool threshold_adjusted_ = false;

  T &_r(std::size_t row, std::size_t column) {
    return this->triangular_[column * (column + 1) / 2 + row];
  }

  template <typename Operator, typename Preconditioner>
  Vector<T, N>
  _solve(const Operator &operation, const Vector<T, N> &right_hand_side,
         const Vector<T, N> &initial, const Preconditioner &inverse_diagonal,
         std::size_t active_size) {
    using Scalar = Numerical::Scalar<T>;
    const Real zero = static_cast<Real>(0);
    const Real floor = Numerical::positive_threshold(this->division_min);
    this->threshold_adjusted_ = this->division_min <= zero;
    this->iterations_ = 0;
    this->breakdown_ = false;
    this->converged_ = false;
    std::size_t dimension = this->krylov_dimension;
    dimension = dimension == 0 ? 1 : (dimension > K ? K : dimension);
    if (dimension > active_size) {
      dimension = active_size;
    }
    const std::size_t passes = this->orthogonalization_passes > 1 ? 2 : 1;
    const Real relative =
        this->relative_tolerance > zero ? this->relative_tolerance : zero;
    const Real absolute =
        this->absolute_tolerance > zero ? this->absolute_tolerance : zero;
    const Real tolerance =
        absolute + relative * Numerical::norm(right_hand_side, active_size);

    Vector<T, N> solution = initial;
    Vector<T, N> residual;
    GmresOperation::apply_operator(operation, solution, residual, active_size);
    for (std::size_t i = 0; i < N; ++i) {
      residual[i] = i < active_size ? right_hand_side[i] - residual[i] : T();
    }
    this->residual_norm_ = Numerical::norm(residual, active_size);
    this->converged_ = this->residual_norm_ <= tolerance;

    for (std::size_t cycle = 0; cycle < this->cycles; ++cycle) {
      if (!this->converged_ && !this->breakdown_ && dimension > 0) {
        const Real initial_norm = this->residual_norm_;
        for (std::size_t j = 0; j <= K; ++j) {
          this->transformed_rhs_[j] = T();
        }
        this->transformed_rhs_[0] = initial_norm;
        // initial_norm > tolerance >= 0, so this denominator is nonzero.
        // Use a shared reciprocal only when it is representable.
        const bool initial_reciprocal_safe =
            initial_norm >= std::numeric_limits<Real>::min();
        const Real initial_inverse = initial_reciprocal_safe
                                         ? static_cast<Real>(1) / initial_norm
                                         : zero;
        for (std::size_t i = 0; i < N; ++i) {
          this->basis_[i] = initial_reciprocal_safe
                                ? residual[i] * initial_inverse
                                : residual[i] / initial_norm;
        }

        std::size_t used = 0;
        bool active = true;
        bool arnoldi_breakdown = false;
        Vector<T, N> input;
        Vector<T, N> product;
        for (std::size_t column = 0; column < K; ++column) {
          if (active && column < dimension) {
            for (std::size_t i = 0; i < N; ++i) {
              input[i] = this->basis_[column * N + i] * inverse_diagonal[i];
            }
            GmresOperation::apply_operator(operation, input, product,
                                           active_size);
            Real product_norm = Numerical::norm(product, active_size);
            for (std::size_t j = 0; j <= K; ++j) {
              this->column_[j] = T();
            }

            // Modified Gram-Schmidt; the second pass adds its corrections.
            for (std::size_t pass = 0; pass < passes; ++pass) {
              for (std::size_t j = 0; j <= column; ++j) {
                T projection = T();
                for (std::size_t i = 0; i < N; ++i) {
                  projection +=
                      Scalar::conjugate(this->basis_[j * N + i]) * product[i];
                }
                this->column_[j] += projection;
                for (std::size_t i = 0; i < N; ++i) {
                  product[i] -= this->basis_[j * N + i] * projection;
                }
              }
            }
            Real next_norm = Numerical::norm(product, active_size);
            this->column_[column + 1] = next_norm;
            Real breakdown_limit = static_cast<Real>(8) *
                                   std::numeric_limits<Real>::epsilon() *
                                   product_norm;
            arnoldi_breakdown = next_norm <= breakdown_limit;

            // The last column still needs its subdiagonal and rotation.
            if (!arnoldi_breakdown && column + 1 < dimension) {
              const bool reciprocal_safe =
                  next_norm >= std::numeric_limits<Real>::min();
              const Real inverse_norm =
                  reciprocal_safe ? static_cast<Real>(1) / next_norm : zero;
              for (std::size_t i = 0; i < N; ++i) {
                this->basis_[(column + 1) * N + i] =
                    reciprocal_safe ? product[i] * inverse_norm
                                    : product[i] / next_norm;
              }
            }
            for (std::size_t j = 0; j < column; ++j) {
              T upper = this->column_[j];
              T lower = this->column_[j + 1];
              this->column_[j] =
                  upper * this->cosine_[j] + this->sine_[j] * lower;
              this->column_[j + 1] =
                  -Scalar::conjugate(this->sine_[j]) * upper +
                  lower * this->cosine_[j];
            }
            Numerical::givens(this->column_[column], this->column_[column + 1],
                              this->cosine_[column], this->sine_[column]);
            this->column_[column] =
                this->column_[column] * this->cosine_[column] +
                this->sine_[column] * this->column_[column + 1];
            for (std::size_t j = 0; j <= column; ++j) {
              this->_r(j, column) = this->column_[j];
            }
            this->transformed_rhs_[column + 1] =
                -Scalar::conjugate(this->sine_[column]) *
                this->transformed_rhs_[column];
            this->transformed_rhs_[column] *= this->cosine_[column];
            used = column + 1;
            ++this->iterations_;
            active = !arnoldi_breakdown &&
                     Scalar::magnitude(this->transformed_rhs_[column + 1]) >
                         tolerance;
          }
        }

        bool triangular_valid = true;
        for (std::size_t offset = 0; offset < K; ++offset) {
          const std::size_t row = K - 1 - offset;
          this->coefficients_[row] = T();
          if (row < used && triangular_valid) {
            T value = this->transformed_rhs_[row];
            for (std::size_t j = row + 1; j < K; ++j) {
              if (j < used) {
                value -= this->_r(row, j) * this->coefficients_[j];
              }
            }
            if (Scalar::magnitude(this->_r(row, row)) < floor) {
              triangular_valid = false;
            } else {
              this->coefficients_[row] =
                  Scalar::divide(value, this->_r(row, row), floor);
            }
          }
        }

        if (triangular_valid) {
          Vector<T, N> candidate = solution;
          for (std::size_t i = 0; i < N; ++i) {
            T correction = T();
            for (std::size_t j = 0; j < K; ++j) {
              if (j < used) {
                correction += this->basis_[j * N + i] * this->coefficients_[j];
              }
            }
            candidate[i] += inverse_diagonal[i] * correction;
          }
          GmresOperation::apply_operator(operation, candidate, residual,
                                         active_size);
          for (std::size_t i = 0; i < N; ++i) {
            residual[i] =
                i < active_size ? right_hand_side[i] - residual[i] : T();
          }
          Real candidate_norm = Numerical::norm(residual, active_size);
          if (candidate_norm <= initial_norm) {
            solution = candidate;
            this->residual_norm_ = candidate_norm;
          } else {
            GmresOperation::apply_operator(operation, solution, residual,
                                           active_size);
            for (std::size_t i = 0; i < N; ++i) {
              residual[i] =
                  i < active_size ? right_hand_side[i] - residual[i] : T();
            }
          }
          this->converged_ = this->residual_norm_ <= tolerance;
        }
        // A happy breakdown is successful only when the true residual agrees.
        this->breakdown_ =
            !triangular_valid || (arnoldi_breakdown && !this->converged_);
      }
    }
    return solution;
  }
};

/** Build an optional fixed right preconditioner; tiny diagonals use identity.
 */
template <typename T, std::size_t N>
inline Vector<T, N>
gmres_inverse_diagonal(const Matrix<T, N, N> &matrix,
                       typename Numerical::Scalar<T>::Real division_min) {
  using Scalar = Numerical::Scalar<T>;
  using Real = typename Scalar::Real;
  const Real floor = Numerical::positive_threshold(division_min);
  Vector<T, N> inverse;
  for (std::size_t i = 0; i < N; ++i) {
    const Real magnitude = Scalar::magnitude(matrix(i, i));
    inverse[i] =
        magnitude < floor || magnitude < std::numeric_limits<Real>::min()
            ? static_cast<T>(1)
            : Scalar::divide(static_cast<T>(1), matrix(i, i), floor);
  }
  return inverse;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_GMRES_HPP_
