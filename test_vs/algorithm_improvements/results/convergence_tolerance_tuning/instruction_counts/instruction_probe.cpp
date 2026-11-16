// Host-only dynamic instruction counting. No hooks are added to the library.
#include "base_matrix_gmres.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sys/ptrace.h>

using namespace Base::Matrix;

// Keep the measured interval visible to the debugger and prevent code motion
// across its boundaries. These attributes affect only the host wrappers.
extern "C" inline __attribute__((noinline, noipa, used))
void instruction_count_begin() { asm volatile("" ::: "memory"); }
extern "C" inline __attribute__((noinline, noipa, used))
void instruction_count_end() { asm volatile("" ::: "memory"); }

template <typename S> struct Components {
  static inline S make(double real, double imaginary) {
    static_cast<void>(imaginary);
    return static_cast<S>(real);
  }
  static inline void print(const S &value) {
    std::printf(",%a", static_cast<double>(value));
  }
};

template <typename T> struct Components<Complex<T>> {
  static inline Complex<T> make(double real, double imaginary) {
    return Complex<T>(static_cast<T>(real), static_cast<T>(imaginary));
  }
  static inline void print(const Complex<T> &value) {
    std::printf(",%a,%a", static_cast<double>(value.real),
                static_cast<double>(value.imag));
  }
};

template <typename S, std::size_t N> struct MultiplyCase {
  Matrix<S, N, N> matrix;
  Vector<S, N> input, output;
  std::size_t active_size;

  inline MultiplyCase() : active_size(N) {
    for (std::size_t row = 0; row < N; ++row) {
      this->input[row] = Components<S>::make(
          double(row + 1) / double(N + 1), -0.03 * double(row % 3 + 1));
      for (std::size_t column = 0; column < N; ++column) {
        this->matrix(row, column) = Components<S>::make(
            row == column ? 4.0 + double(row) / double(N)
                          : double((row + column) % 5) * 0.03,
            double((row + 2 * column) % 7) * 0.01);
      }
    }
  }

  static inline __attribute__((noinline, noipa)) void apply(void *opaque) {
    MultiplyCase &context = *static_cast<MultiplyCase *>(opaque);
    GmresOperation::multiply(context.matrix, context.input, context.output,
                             context.active_size);
  }

  inline void print() const {
    std::printf("input");
    for (std::size_t row = 0; row < N; ++row) {
      for (std::size_t column = 0; column < N; ++column) {
        Components<S>::print(this->matrix(row, column));
      }
    }
    for (std::size_t row = 0; row < N; ++row) {
      Components<S>::print(this->input[row]);
    }
    std::printf("\noutput");
    for (std::size_t row = 0; row < N; ++row) {
      Components<S>::print(this->output[row]);
    }
    std::printf("\n");
  }
};

template <typename S, std::size_t N> struct NormCase {
  using Real = typename Numerical::Scalar<S>::Real;
  Vector<S, N> input;
  Real output;
  std::size_t active_size;

  inline NormCase() : output(0), active_size(N) {
    for (std::size_t row = 0; row < N; ++row) {
      this->input[row] = Components<S>::make(
          double(row + 1) / double(N + 1), -0.03 * double(row % 3 + 1));
    }
  }

  static inline __attribute__((noinline, noipa)) void apply(void *opaque) {
    NormCase &context = *static_cast<NormCase *>(opaque);
    context.output = Numerical::norm(context.input, context.active_size);
  }

  inline void print() const {
    std::printf("input");
    for (std::size_t row = 0; row < N; ++row) {
      Components<S>::print(this->input[row]);
    }
    std::printf("\noutput,%a\n", static_cast<double>(this->output));
  }
};

// Same input, full dimension, and budget as the previous RAM double8 case.
template <typename T, std::size_t N> struct GmresCase {
  Matrix<T, N, N> matrix;
  Vector<T, N> rhs, initial, output;
  GmresSolver<T, N> solver;

  inline GmresCase() {
    for (std::size_t row = 0; row < N; ++row) {
      for (std::size_t column = 0; column < N; ++column) {
        this->matrix(row, column) = row == column
            ? T(4) + T(row) / T(N) : T((row + column) % 5) * T(0.03);
        this->rhs[row] += this->matrix(row, column);
      }
    }
    this->solver.division_min = T(1e-25);
    this->solver.cycles = 1;
    this->solver.orthogonalization_passes = 1;
    this->solver.relative_tolerance = T(0);
    this->solver.absolute_tolerance = T(0);
  }

  static inline __attribute__((noinline, noipa)) void apply(void *opaque) {
    GmresCase &context = *static_cast<GmresCase *>(opaque);
    context.output = context.solver.solve(context.matrix, context.rhs,
                                           context.initial);
  }

  inline void print() const {
    std::printf("input");
    for (std::size_t row = 0; row < N; ++row) {
      for (std::size_t column = 0; column < N; ++column) {
        Components<T>::print(this->matrix(row, column));
      }
    }
    for (std::size_t row = 0; row < N; ++row) {
      Components<T>::print(this->rhs[row]);
    }
    std::printf("\noutput");
    for (std::size_t row = 0; row < N; ++row) {
      Components<T>::print(this->output[row]);
    }
    std::printf("\nstatus,%zu,%d,%d,%a\n", this->solver.get_iterations(),
                int(this->solver.is_converged()),
                int(this->solver.has_breakdown()),
                static_cast<double>(this->solver.get_residual_norm()));
  }
};

template <typename Context> inline void run() {
  Context context;
  // Resolve external routines and warm the same solver before the interval.
  Context::apply(&context);
  instruction_count_begin();
  Context::apply(&context);
  instruction_count_end();
  context.print();
}

int main(int argc, char **argv) {
  if (argc != 3) { return EXIT_FAILURE; }
  if (std::strcmp(argv[1], "trace") == 0) {
    if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) != 0) {
      std::perror("PTRACE_TRACEME");
      return EXIT_FAILURE;
    }
    raise(SIGSTOP);
  }
  const char *requested = argv[2];
#define RUN_CASE(NAME, CONTEXT, TYPE, N)                                      \
  if (std::strcmp(requested, NAME) == 0) {                                    \
    run<CONTEXT<TYPE, N>>();                                                  \
    return EXIT_SUCCESS;                                                    \
  }
  RUN_CASE("multiply_real_float_8", MultiplyCase, float, 8)
  RUN_CASE("multiply_real_float_16", MultiplyCase, float, 16)
  RUN_CASE("multiply_real_double_8", MultiplyCase, double, 8)
  RUN_CASE("multiply_real_double_16", MultiplyCase, double, 16)
  RUN_CASE("multiply_complex_float_8", MultiplyCase, Complex<float>, 8)
  RUN_CASE("multiply_complex_float_16", MultiplyCase, Complex<float>, 16)
  RUN_CASE("multiply_complex_double_8", MultiplyCase, Complex<double>, 8)
  RUN_CASE("multiply_complex_double_16", MultiplyCase, Complex<double>, 16)
  RUN_CASE("norm_real_float_8", NormCase, float, 8)
  RUN_CASE("norm_real_float_16", NormCase, float, 16)
  RUN_CASE("norm_real_double_8", NormCase, double, 8)
  RUN_CASE("norm_real_double_16", NormCase, double, 16)
  RUN_CASE("norm_complex_float_8", NormCase, Complex<float>, 8)
  RUN_CASE("norm_complex_float_16", NormCase, Complex<float>, 16)
  RUN_CASE("norm_complex_double_8", NormCase, Complex<double>, 8)
  RUN_CASE("norm_complex_double_16", NormCase, Complex<double>, 16)
  RUN_CASE("gmres_real_float_8", GmresCase, float, 8)
  RUN_CASE("gmres_real_double_8", GmresCase, double, 8)
  RUN_CASE("gmres_real_float_16", GmresCase, float, 16)
  RUN_CASE("gmres_real_double_16", GmresCase, double, 16)
#undef RUN_CASE
  return EXIT_FAILURE;
}
