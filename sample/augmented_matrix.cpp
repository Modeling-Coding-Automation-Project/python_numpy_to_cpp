/**
 * @file augmented_matrix.cpp
 * @brief Demonstrates the usage of AugmentedMatrix in C++.
 *
 * This file contains a sample program that showcases the creation and
 * manipulation of an AugmentedMatrix using custom matrix types in C++. It
 * demonstrates the initialization of an AugmentedMatrix, multiplication of two
 * AugmentedMatrices, and accessing elements of the resulting matrix.
 */
#include "python_numpy.hpp"

#include <iostream>

using namespace PythonNumpy;

int main(void) {

  using SparseAvailable_A =
      SparseAvailable<ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>,
                      ColumnAvailable<true, true, true, true, true, true, true,
                                      true, true, true>>;

  using A_Type = SparseMatrix_Type<double, SparseAvailable_A>;

  auto A = make_SparseMatrixZeros<double, SparseAvailable_A>();

  using AA_Tuple_Type =
      AugmentedMatrix_Tuple_Type<A_Type, A_Type, A_Type, A_Type>;

  auto AA = make_AugmentedMatrixZeros<AA_Tuple_Type>();

  auto AB = make_AugmentedMatrix(A, A, A, A);

  auto C = AA * AB;

  std::cout << "C(19, 19) = " << C.template get<19, 19>() << std::endl;

  return 0;
}
