#ifndef PYTHON_MATH_HPP_
#define PYTHON_MATH_HPP_

#include "base_matrix.hpp"

#include "python_numpy_augmented_matrix.hpp"
#include "python_numpy_base.hpp"

namespace PythonNumpy {

/* Constants */

constexpr double PI = Base::Matrix::PI;
constexpr double TWO_PI = Base::Matrix::TWO_PI;
constexpr double HALF_PI = Base::Matrix::HALF_PI;
constexpr double TAU = Base::Matrix::TAU;

constexpr double E = Base::Matrix::E;

constexpr double LN_2 = Base::Matrix::LN_2;
constexpr double LN_10 = Base::Matrix::LN_10;

/* abs */

template <typename T, std::size_t M, std::size_t N>
inline auto abs(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto abs(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto abs(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

namespace AbsAugmentedMatrixAction {

/**
 * @brief Helper struct to apply abs to each element of a std::tuple
 * using template metaprogramming.
 *
 * This struct recursively applies Base::Matrix::abs to each element of
 * the input tuple. Uses index-based recursion for C++11 compatibility.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @tparam Index The current index in the tuple (decremented recursively).
 */
template <typename Tuple_Type, std::size_t Index> struct ApplyAbsTupleCore {
  /**
   * @brief Recursively applies abs to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where abs results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::abs(std::get<Index>(input).matrix);
    ApplyAbsTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

/**
 * @brief Base case specialization for index 0.
 *
 * @tparam Tuple_Type The type of the tuple.
 */
template <typename Tuple_Type> struct ApplyAbsTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where abs results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::abs(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply abs to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where abs results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAbsTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace AbsAugmentedMatrixAction

template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
abs(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  AbsAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

} // namespace PythonNumpy

#endif // PYTHON_MATH_HPP_
