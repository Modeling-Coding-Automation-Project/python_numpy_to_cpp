// Host-only benchmark. Compile the same source against baseline and new headers.
#include "base_matrix_eigen_solver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace Base::Matrix;
volatile double eigen_benchmark_sink = 0;

template <typename T, std::size_t N>
Matrix<T, N, N> input_matrix() {
  Matrix<T, N, N> q = Matrix<T, N, N>::identity();
  for (std::size_t p = 0; p < N; ++p) {
    for (std::size_t r = p + 1; r < N; ++r) {
      for (std::size_t i = 0; i < N; ++i) {
        T first = q(i, p);
        T second = q(i, r);
        q(i, p) = T(0.8) * first - T(0.6) * second;
        q(i, r) = T(0.6) * first + T(0.8) * second;
      }
    }
  }
  Matrix<T, N, N> result;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t k = 0; k < N; ++k) {
        result(i, j) += q(i, k) * T(k + 1) * q(j, k);
      }
    }
  }
  return result;
}

template <typename Function> double time_call(Function function, int repeats) {
  double best = 1e100;
  for (int trial = 0; trial < 5; ++trial) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
      function();
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::micro>(end - start).count() /
                     double(repeats);
    if (elapsed < best) {
      best = elapsed;
    }
  }
  return best;
}

template <typename T, std::size_t N, typename Values>
double spectrum_error(const Values &values) {
  bool used[N] = {};
  double maximum = 0;
  for (std::size_t k = 0; k < N; ++k) {
    double best = 1e100;
    std::size_t selected = 0;
    for (std::size_t i = 0; i < N; ++i) {
      double error = std::abs(double(values[i]) - double(k + 1));
      if (!used[i] && error < best) {
        best = error;
        selected = i;
      }
    }
    used[selected] = true;
    if (best > maximum) {
      maximum = best;
    }
  }
  return maximum / double(N);
}

template <typename T, std::size_t N, typename Solver>
double eigenvector_error(const Matrix<T, N, N> &matrix, const Solver &solver) {
  Solver &nonconst = const_cast<Solver &>(solver);
  auto values = nonconst.get_eigen_values();
  auto vectors = nonconst.get_eigen_vectors();
  double maximum = 0;
  for (std::size_t j = 0; j < N; ++j) {
    double length = 0;
    double residual = 0;
    for (std::size_t i = 0; i < N; ++i) {
      double r = -double(values[j]) * double(vectors(i, j));
      for (std::size_t k = 0; k < N; ++k) {
        r += double(matrix(i, k)) * double(vectors(k, j));
      }
      length += double(vectors(i, j)) * double(vectors(i, j));
      residual += r * r;
    }
    double error = length > 0 ? std::sqrt(residual / length) / double(N) : 1e100;
    if (error > maximum) {
      maximum = error;
    }
  }
  return maximum;
}

template <typename T, std::size_t N>
struct HessenbergProbe : EigenSolverReal<T, N> {
  void reduce(const Matrix<T, N, N> &matrix) {
#ifdef ALGORITHM_IMPROVEMENTS
    EigenOperation::hessenberg(matrix, this->Hessen_, T(1e-20));
#else
    this->_hessenberg(matrix);
#endif
  }
  T diagonal() { return this->Hessen_(0, 0); }
  Matrix<T, N, N> reduced() { return this->Hessen_; }
};

template <typename T, std::size_t N>
struct InverseIterationProbe : EigenSolverReal<T, N> {
  void use_known_values() {
    for (std::size_t i = 0; i < N; ++i) {
      this->_eigen_values[i] = T(i + 1);
    }
#ifdef ALGORITHM_IMPROVEMENTS
    this->_values_converged = true;
#endif
  }
};

#ifdef ALGORITHM_IMPROVEMENTS
template <typename T, std::size_t N>
struct RestartedInverseProbe : EigenSolverReal<T, N, N / 2> {
  void use_known_values() {
    for (std::size_t i = 0; i < N; ++i) {
      this->_eigen_values[i] = T(i + 1);
    }
    this->_values_converged = true;
  }
};
#endif

template <typename T, std::size_t N> void run() {
  const char *type = std::is_same<T, float>::value ? "float" : "double";
  auto matrix = input_matrix<T, N>();
  int repeats = N <= 8 ? 40 : 5;
  HessenbergProbe<T, N> reduction;
  double reduce_time = time_call([&]() {
    reduction.reduce(matrix);
    eigen_benchmark_sink += double(reduction.diagonal());
  }, repeats);
  auto reduced = reduction.reduced();
  double original_square = 0;
  double reduced_square = 0;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      original_square += double(matrix(i, j)) * double(matrix(i, j));
      reduced_square += double(reduced(i, j)) * double(reduced(i, j));
    }
  }
  double reduction_error = std::abs(reduced_square - original_square) /
                           original_square;
  std::printf("hessenberg,%s,%zu,0,%.9g,%.9g,%zu\n", type, N,
              reduce_time, reduction_error, sizeof(reduction));

  InverseIterationProbe<T, N> real;
#ifdef ALGORITHM_IMPROVEMENTS
  real.iteration_max = 64;
#else
  real.iteration_max = 10;
#endif
  double qr_time = time_call([&]() {
    real.solve_eigen_values(matrix);
    eigen_benchmark_sink += double(real.get_eigen_values()[0]);
  }, repeats);
  std::printf("qr_values,%s,%zu,%zu,%.9g,%.9g,%zu\n", type, N,
              real.iteration_max, qr_time,
              spectrum_error<T, N>(real.get_eigen_values()), sizeof(real));

  EigenSolverComplex<T, N> complex;
#ifdef ALGORITHM_IMPROVEMENTS
  complex.iteration_max = 64;
#else
  complex.iteration_max = 10;
#endif
  double complex_time = time_call([&]() {
    complex.solve_eigen_values(matrix);
    eigen_benchmark_sink += double(complex.get_eigen_values()[0].real);
  }, repeats);
  auto complex_values = complex.get_eigen_values();
  double complex_error = 0;
  bool matched[N] = {};
  for (std::size_t k = 0; k < N; ++k) {
    double best = 1e100;
    std::size_t selected = 0;
    for (std::size_t i = 0; i < N; ++i) {
      double real_error = double(complex_values[i].real) - double(k + 1);
      double imag_error = double(complex_values[i].imag);
      double error = std::sqrt(real_error * real_error + imag_error * imag_error);
      if (!matched[i] && error < best) {
        best = error;
        selected = i;
      }
    }
    matched[selected] = true;
    if (best > complex_error) {
      complex_error = best;
    }
  }
  std::printf("qr_complex_values,%s,%zu,%zu,%.9g,%.9g,%zu\n", type, N,
              complex.iteration_max, complex_time, complex_error / double(N),
              sizeof(complex));

  // Isolate inverse iteration from eigenvalue QR errors: these manufactured
  // matrices have the exact spectrum 1..N before input rounding.
  real.use_known_values();
#ifdef ALGORITHM_IMPROVEMENTS
  const std::size_t budgets[] = {1, 3, 8};
  for (std::size_t budget : budgets) {
    real.iteration_max_for_eigen_vector = budget;
#else
  const std::size_t budgets[] = {1, 3, 8};
  for (std::size_t budget : budgets) {
    real.iteration_max = budget;
#endif
    double inverse_time = time_call([&]() {
      InverseIterationProbe<T, N> candidate = real;
      candidate.solve_eigen_vectors(matrix);
      eigen_benchmark_sink += double(candidate.get_eigen_vectors()(0, 0));
    }, repeats);
    InverseIterationProbe<T, N> candidate = real;
    candidate.solve_eigen_vectors(matrix);
    std::printf("inverse_vectors,%s,%zu,%zu,%.9g,%.9g,%zu\n", type, N,
                budget, inverse_time, eigenvector_error(matrix, candidate), sizeof(real));
  }

#ifdef ALGORITHM_IMPROVEMENTS
  // Separate inner budgets from the outer V budget; capacity remains N.
  for (std::size_t passes = 1; passes <= 2; ++passes) {
    InverseIterationProbe<T, N> configuration = real;
    configuration.iteration_max_for_eigen_vector = 8;
    configuration.gmres_orthogonalization_passes = passes;
    configuration.eigenvector_tolerance = std::numeric_limits<T>::epsilon();
    configuration.gmres_k_decay_rate = EigenOperation::square_root(std::numeric_limits<T>::epsilon());
    double elapsed = time_call([&]() {
      InverseIterationProbe<T, N> candidate = configuration;
      candidate.solve_eigen_vectors(matrix);
      eigen_benchmark_sink += double(candidate.get_eigen_vectors()(0, 0));
    }, repeats);
    configuration.solve_eigen_vectors(matrix);
    std::printf("inverse_strict_p%zu,%s,%zu,8,%.9g,%.9g,%zu\n", passes, type, N,
                elapsed, eigenvector_error(matrix, configuration), sizeof(configuration));
  }
  RestartedInverseProbe<T, N> restarted;
  restarted.iteration_max = 64;
  restarted.solve_eigen_values(matrix);
  restarted.use_known_values();
  restarted.iteration_max_for_eigen_vector = 8;
  restarted.gmres_k_dimension = N / 2;
  restarted.gmres_cycles = 4;
  double restarted_time = time_call([&]() {
    RestartedInverseProbe<T, N> candidate = restarted;
    candidate.solve_eigen_vectors(matrix);
    eigen_benchmark_sink += double(candidate.get_eigen_vectors()(0, 0));
  }, repeats);
  restarted.solve_eigen_vectors(matrix);
  std::printf("inverse_restart_khalf_c4,%s,%zu,8,%.9g,%.9g,%zu\n", type, N,
              restarted_time, eigenvector_error(matrix, restarted), sizeof(restarted));

  const std::size_t jacobi_budgets[] = {2, 4, 8};
  for (std::size_t budget : jacobi_budgets) {
    EigenSolverSymmetric<T, N, false> packed;
    packed.iteration_max = budget;
    double value_time = time_call([&]() {
      packed.solve_eigen_values(matrix);
      eigen_benchmark_sink += double(packed.get_eigen_values()[0]);
    }, repeats);
    std::printf("jacobi_values,%s,%zu,%zu,%.9g,%.9g,%zu\n", type, N,
                budget, value_time, spectrum_error<T, N>(packed.get_eigen_values()),
                sizeof(packed));
    EigenSolverSymmetric<T, N> jacobi;
    jacobi.iteration_max = budget;
    double vector_time = time_call([&]() {
      jacobi.solve_eigen_values_and_vectors(matrix);
      eigen_benchmark_sink += double(jacobi.get_eigen_vectors()(0, 0));
    }, repeats);
    std::printf("jacobi_vectors,%s,%zu,%zu,%.9g,%.9g,%zu\n", type, N,
                budget, vector_time, eigenvector_error(matrix, jacobi), sizeof(jacobi));
  }
#endif
}

int main() {
  std::puts("algorithm,type,n,budget,microseconds,error,object_bytes");
  run<float, 4>();
  run<double, 4>();
  run<float, 8>();
  run<double, 8>();
  run<double, 16>();
}
