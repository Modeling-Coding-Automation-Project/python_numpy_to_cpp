// Host-only stack-watermark probe. No measurement code runs inside the kernels.
#include "base_matrix_cholesky_decomposition.hpp"
#include "base_matrix_eigen_solver.hpp"
#include "base_matrix_lu_decomposition.hpp"
#include "base_matrix_qr_decomposition.hpp"
#include "peak_stack_worker.hpp"

#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace Base::Matrix;

constexpr std::size_t PROBE_STACK_BYTES = 2 * 1024 * 1024;
constexpr std::size_t PROBE_REPEATS = 3;

template <typename T, std::size_t N>
inline Matrix<T, N, N> fixture() {
  Matrix<T, N, N> matrix;
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t column = 0; column < N; ++column) {
      matrix(row, column) = row == column
                                ? T(4) + T(row) / T(N)
                                : T((row + column) % 5) * T(0.03);
    }
  }
  return matrix;
}

template <typename T, std::size_t N, std::size_t K> struct GmresCase {
  Matrix<T, N, N> matrix;
  Vector<T, N> rhs, initial, result;
  GmresSolver<T, N, K> solver;
  inline GmresCase() : matrix(fixture<T, N>()) {
    for (std::size_t row = 0; row < N; ++row) {
      for (std::size_t column = 0; column < N; ++column) {
        this->rhs[row] += this->matrix(row, column);
      }
    }
    this->solver.division_min = T(1e-25);
    this->solver.cycles = K == N ? 1 : 4;
    this->solver.orthogonalization_passes = 1;
  }
  static inline void apply(void *opaque) {
    GmresCase &context = *static_cast<GmresCase *>(opaque);
    context.result = context.solver.solve(context.matrix, context.rhs,
                                         context.initial);
  }
};

template <typename T, std::size_t N, typename Solver, bool Vectors>
struct EigenCase {
  Matrix<T, N, N> matrix;
  Solver solver;
  inline EigenCase() : matrix(fixture<T, N>()) {
    this->solver.iteration_max = 64;
    this->solver.iteration_max_for_eigen_vector = 8;
    this->solver.division_min = T(1e-20);
  }
  static inline void apply(void *opaque) {
    EigenCase &context = *static_cast<EigenCase *>(opaque);
    context.solver.solve_eigen_values(context.matrix);
    if (Vectors) {
      context.solver.solve_eigen_vectors(context.matrix);
    }
  }
};

template <typename T, std::size_t N, bool Vectors> struct JacobiCase {
  Matrix<T, N, N> matrix;
  EigenSolverSymmetric<T, N, Vectors> solver;
  inline JacobiCase() : matrix(fixture<T, N>()) {
    this->solver.iteration_max = 8;
    this->solver.division_min = T(1e-20);
  }
  static inline void apply(void *opaque) {
    JacobiCase &context = *static_cast<JacobiCase *>(opaque);
    context.solver.solve_eigen_values(context.matrix);
  }
};

template <typename T, std::size_t N, typename Solver> struct QrCase {
  Matrix<T, N, N> matrix;
  Solver solver;
  inline QrCase() : matrix(fixture<T, N>()) {}
  static inline void apply(void *opaque) {
    QrCase &context = *static_cast<QrCase *>(opaque);
    context.solver.solve(context.matrix);
  }
};

template <typename T, std::size_t N> struct HouseholderQCase {
  Matrix<T, N, N> matrix, output;
  QRDecompositionHouseholder<T, N, N> solver;
  inline HouseholderQCase() : matrix(fixture<T, N>()) {}
  static inline void apply(void *opaque) {
    HouseholderQCase &context = *static_cast<HouseholderQCase *>(opaque);
    context.solver.solve(context.matrix);
    context.solver.get_Q(context.output);
  }
};

template <typename T, std::size_t N> struct LuCase {
  Matrix<T, N, N> matrix;
  Vector<T, N> rhs, result;
  LUDecomposition<T, N> solver;
  inline LuCase() : matrix(fixture<T, N>()) {
    for (std::size_t row = 0; row < N; ++row) {
      this->rhs[row] = T(row + 1);
    }
  }
  static inline void apply(void *opaque) {
    LuCase &context = *static_cast<LuCase *>(opaque);
    context.result = context.solver.solve(context.matrix, context.rhs);
  }
};

template <typename T, std::size_t N> struct CholeskyCase {
  Matrix<T, N, N> matrix, output;
  bool failed;
  inline CholeskyCase() : matrix(fixture<T, N>()), failed(false) {}
  static inline void apply(void *opaque) {
    CholeskyCase &context = *static_cast<CholeskyCase *>(opaque);
    context.failed = cholesky_decomposition_to(context.matrix, context.output,
                                             T(1e-20));
  }
};

template <typename T, std::size_t N> struct PackedCholeskyCase {
  Matrix<T, N, N> matrix;
  PackedCholesky<T, N> solver;
  inline PackedCholeskyCase() : matrix(fixture<T, N>()) {
    this->solver.division_min = T(1e-20);
  }
  static inline void apply(void *opaque) {
    PackedCholeskyCase &context = *static_cast<PackedCholeskyCase *>(opaque);
    context.solver.solve(context.matrix);
  }
};

// Each unmangled root is separately identifiable in GCC's call graph. The
// noinline boundary is identical for every revision. It does not inhibit any
// inlining among library functions beneath that boundary.
#define PEAK_DEFINE_ROOT(Name, ...)                                             \
  using Name##_Context = __VA_ARGS__;                                          \
  extern "C" inline __attribute__((noinline, used)) void Name(void *opaque) {   \
    Name##_Context::apply(opaque);                                             \
    asm volatile("" : : : "memory");                                         \
  }

#define PEAK_DEFINE_TYPE(T, N)                                                  \
  PEAK_DEFINE_ROOT(peak_gmres_full_##T##_##N, GmresCase<T, N, N>)                \
  PEAK_DEFINE_ROOT(peak_gmres_restart_##T##_##N, GmresCase<T, N, N / 2>)         \
  PEAK_DEFINE_ROOT(peak_eigen_real_##T##_##N,                                   \
                   EigenCase<T, N, EigenSolverReal<T, N>, true>)               \
  PEAK_DEFINE_ROOT(peak_eigen_complex_##T##_##N,                                \
                   EigenCase<T, N, EigenSolverComplex<T, N>, true>)            \
  PEAK_DEFINE_ROOT(peak_eigen_real_values_##T##_##N,                            \
                   EigenCase<T, N, EigenSolverReal<T, N>, false>)              \
  PEAK_DEFINE_ROOT(peak_eigen_complex_values_##T##_##N,                         \
                   EigenCase<T, N, EigenSolverComplex<T, N>, false>)           \
  PEAK_DEFINE_ROOT(peak_jacobi_values_##T##_##N, JacobiCase<T, N, false>)        \
  PEAK_DEFINE_ROOT(peak_jacobi_vectors_##T##_##N, JacobiCase<T, N, true>)        \
  PEAK_DEFINE_ROOT(peak_qr_givens_##T##_##N,                                    \
                   QrCase<T, N, QRDecomposition<T, N, N>>)                     \
  PEAK_DEFINE_ROOT(peak_qr_householder_##T##_##N,                               \
                   QrCase<T, N, QRDecompositionHouseholder<T, N, N>>)          \
  PEAK_DEFINE_ROOT(peak_qr_householder_q_##T##_##N, HouseholderQCase<T, N>)     \
  PEAK_DEFINE_ROOT(peak_lu_##T##_##N, LuCase<T, N>)                              \
  PEAK_DEFINE_ROOT(peak_cholesky_##T##_##N, CholeskyCase<T, N>)                  \
  PEAK_DEFINE_ROOT(peak_cholesky_packed_##T##_##N, PackedCholeskyCase<T, N>)

PEAK_DEFINE_TYPE(float, 8)
PEAK_DEFINE_TYPE(double, 8)
PEAK_DEFINE_TYPE(float, 16)
PEAK_DEFINE_TYPE(double, 16)

extern "C" inline __attribute__((noinline, used)) void
peak_calibration_empty(void *) {
  asm volatile("" : : : "memory");
}

template <std::size_t N>
inline __attribute__((noinline)) void calibration_buffer() {
  volatile unsigned char bytes[N];
  for (std::size_t index = 0; index < N; ++index) {
    bytes[index] = static_cast<unsigned char>(index);
  }
  asm volatile("" : : "m"(bytes) : "memory");
}

extern "C" inline __attribute__((noinline, used)) void
peak_calibration_nested(void *) {
  volatile unsigned char outer[256];
  for (std::size_t index = 0; index < sizeof(outer); ++index) {
    outer[index] = static_cast<unsigned char>(index);
  }
  calibration_buffer<4096>();
  asm volatile("" : : "m"(outer) : "memory");
}

inline std::size_t measure(void (*operation)(void *), void *context,
                           std::size_t &minimum, std::size_t &maximum) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) { std::exit(EXIT_FAILURE); }
  const std::size_t guard = static_cast<std::size_t>(page_size);
  void *mapping = mmap(nullptr, PROBE_STACK_BYTES + 2 * guard, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) { std::exit(EXIT_FAILURE); }
  unsigned char *stack = static_cast<unsigned char *>(mapping) + guard;
  if (mprotect(stack, PROBE_STACK_BYTES, PROT_READ | PROT_WRITE) != 0) {
    std::exit(EXIT_FAILURE);
  }
  pthread_attr_t attributes;
  if (pthread_attr_init(&attributes) != 0 ||
      pthread_attr_setstack(&attributes, stack, PROBE_STACK_BYTES) != 0) {
    std::exit(EXIT_FAILURE);
  }
  minimum = PROBE_STACK_BYTES;
  maximum = 0;
  for (std::size_t repeat = 0; repeat < PROBE_REPEATS; ++repeat) {
    for (std::size_t pattern = 0; pattern < 2; ++pattern) {
      PeakStackSample sample = {
          operation, context, reinterpret_cast<std::uintptr_t>(stack),
          reinterpret_cast<std::uintptr_t>(stack + PROBE_STACK_BYTES),
          static_cast<unsigned char>(pattern == 0 ? 0xa5 : 0x5a), 0, false};
      pthread_t thread;
      if (pthread_create(&thread, &attributes, peak_stack_worker_entry,
                         &sample) != 0 ||
          pthread_join(thread, nullptr) != 0 || !sample.valid) {
        std::fprintf(stderr, "Stack probe failed or overflowed.\n");
        std::exit(EXIT_FAILURE);
      }
      if (sample.written_bytes < minimum) { minimum = sample.written_bytes; }
      if (sample.written_bytes > maximum) { maximum = sample.written_bytes; }
    }
  }
  pthread_attr_destroy(&attributes);
  munmap(mapping, PROBE_STACK_BYTES + 2 * guard);
  return maximum;
}

template <typename Context>
inline void report(const char *name, const char *type, std::size_t dimension,
                   void (*operation)(void *), std::size_t solver_bytes) {
  Context context;
  // Warm external code paths before painting; dynamic-loader lazy binding
  // should not become part of the algorithm's steady-state stack estimate.
  operation(&context);
  std::size_t minimum, maximum;
  measure(operation, &context, minimum, maximum);
  std::printf("%s,%s,%zu,%zu,%zu,%zu,%zu,%zu\n", name, type, dimension,
              sizeof(context), solver_bytes, minimum, maximum,
              sizeof(context) + maximum);
}

#define PEAK_REPORT(Name, T, N, SolverBytes)                                    \
  report<Name##_##T##_##N##_Context>(#Name "_" #T "_" #N, #T, N,               \
                                     Name##_##T##_##N, SolverBytes)

#define PEAK_REPORT_TYPE(T, N)                                                 \
  PEAK_REPORT(peak_gmres_full, T, N, (sizeof(GmresSolver<T, N, N>)));            \
  PEAK_REPORT(peak_gmres_restart, T, N, (sizeof(GmresSolver<T, N, N / 2>)));     \
  PEAK_REPORT(peak_eigen_real, T, N, (sizeof(EigenSolverReal<T, N>)));          \
  PEAK_REPORT(peak_eigen_complex, T, N, (sizeof(EigenSolverComplex<T, N>)));    \
  PEAK_REPORT(peak_eigen_real_values, T, N, (sizeof(EigenSolverReal<T, N>)));   \
  PEAK_REPORT(peak_eigen_complex_values, T, N,                                 \
              (sizeof(EigenSolverComplex<T, N>)));                             \
  PEAK_REPORT(peak_jacobi_values, T, N,                                       \
              (sizeof(EigenSolverSymmetric<T, N, false>)));                    \
  PEAK_REPORT(peak_jacobi_vectors, T, N,                                      \
              (sizeof(EigenSolverSymmetric<T, N, true>)));                     \
  PEAK_REPORT(peak_qr_givens, T, N, (sizeof(QRDecomposition<T, N, N>)));        \
  PEAK_REPORT(peak_qr_householder, T, N,                                      \
              (sizeof(QRDecompositionHouseholder<T, N, N>)));                  \
  PEAK_REPORT(peak_qr_householder_q, T, N,                                    \
              (sizeof(QRDecompositionHouseholder<T, N, N>)));                  \
  PEAK_REPORT(peak_lu, T, N, (sizeof(LUDecomposition<T, N>)));                 \
  PEAK_REPORT(peak_cholesky, T, N, 0);                                        \
  PEAK_REPORT(peak_cholesky_packed, T, N, (sizeof(PackedCholesky<T, N>)))

int main() {
  // The child inherits this mask. This excludes unrelated asynchronous signal
  // handler frames from the watermark; synchronous faults still fail the run.
  sigset_t signals;
  sigfillset(&signals);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  std::puts("root,type,n,context_bytes,solver_bytes,stack_written_min,"
            "stack_written_max,context_plus_written_max");
  std::size_t minimum, maximum;
  measure(peak_calibration_empty, nullptr, minimum, maximum);
  std::printf("peak_calibration_empty,none,0,0,0,%zu,%zu,%zu\n", minimum,
              maximum, maximum);
  if (maximum != sizeof(void *)) { return EXIT_FAILURE; }
  measure(peak_calibration_nested, nullptr, minimum, maximum);
  std::printf("peak_calibration_nested,none,0,0,0,%zu,%zu,%zu\n", minimum,
              maximum, maximum);
  if (maximum < 4096 + 256) { return EXIT_FAILURE; }
  PEAK_REPORT_TYPE(float, 8);
  PEAK_REPORT_TYPE(double, 8);
  PEAK_REPORT_TYPE(float, 16);
  PEAK_REPORT_TYPE(double, 16);
  return EXIT_SUCCESS;
}
