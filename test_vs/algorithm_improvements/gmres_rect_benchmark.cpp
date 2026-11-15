#include "base_matrix_inverse.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
using namespace Base::Matrix;
static volatile double observed = 0;

template <typename T, std::size_t M, std::size_t N, int Mode>
void rectangular_case() {
  Matrix<T, M, N> matrix;
  Vector<T, N> target;
  for (std::size_t i = 0; i < N; ++i) {
    target[i] = T(1);
  }
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      matrix(i, j) = i == j ? T(2) : T((i * 7 + j * 3) % 13) * T(0.01);
    }
  }
  auto rhs = matrix * target;
  Vector<T, N> solution, zero;
  const std::size_t repeats = 1000;
  auto start = std::chrono::steady_clock::now();
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
    if (Mode == 1) {
      Matrix<T, N, N> gram;
      Vector<T, N> normal_rhs;
      for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t i = 0; i < N; ++i) {
          for (std::size_t k = 0; k < M; ++k) {
            gram(i, j) += matrix(k, i) * matrix(k, j);
          }
        }
        for (std::size_t k = 0; k < M; ++k) {
          normal_rhs[j] += matrix(k, j) * rhs[k];
        }
      }
      solution = std::get<0>(gmres_k(gram, normal_rhs, zero, T(0), T(1e-25)));
    } else {
#ifdef ALGORITHM_IMPROVEMENTS
      if (Mode == 2) {
        solution = std::get<0>(
            gmres_k_rect_matrix_free(matrix, rhs, zero, T(0), T(1e-25)));
      } else
#endif
      {
        solution = std::get<0>(gmres_k_rect(matrix, rhs, zero, T(0), T(1e-25)));
      }
    }
    observed += double(solution[repeat % N]);
  }
  auto end = std::chrono::steady_clock::now();
  double square = 0;
  for (std::size_t i = 0; i < N; ++i) {
    double delta = double(solution[i]) - 1.0;
    square += delta * delta;
  }
  std::printf("%s,%s,%zu,%zu,%.3f,%.9g\n",
              Mode == 1 ? "explicit" : (Mode == 2 ? "matrix_free" : "rect_api"),
              sizeof(T) == 4 ? "float" : "double", M, N,
              std::chrono::duration<double, std::nano>(end - start).count() /
                  repeats,
              std::sqrt(square / N));
}
int main() {
  std::puts("mode,type,m,n,ns,rms_target_error");
  rectangular_case<float, 32, 8, false>();
  rectangular_case<float, 32, 8, true>();
  rectangular_case<double, 32, 8, false>();
  rectangular_case<double, 32, 8, true>();
  rectangular_case<float, 24, 16, false>();
  rectangular_case<float, 24, 16, true>();
  rectangular_case<double, 24, 16, false>();
  rectangular_case<double, 24, 16, true>();
  rectangular_case<double, 128, 4, false>();
#ifdef ALGORITHM_IMPROVEMENTS
  rectangular_case<float, 32, 8, 2>();
  rectangular_case<double, 32, 8, 2>();
  rectangular_case<float, 24, 16, 2>();
  rectangular_case<double, 24, 16, 2>();
  rectangular_case<double, 128, 4, 2>();
#endif
}
