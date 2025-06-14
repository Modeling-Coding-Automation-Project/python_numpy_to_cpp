/**
 * @file python_numpy_templates.hpp
 * @brief Template aliases for Python-like Numpy matrix and sparse matrix
 * operations in C++.
 *
 * This header provides a collection of template aliases within the PythonNumpy
 * namespace, designed to facilitate matrix and sparse matrix operations similar
 * to those found in Python's Numpy library. The templates wrap and expose
 * various matrix utilities and structures from the Base::Matrix namespace,
 * enabling type-safe and efficient manipulation of dense, sparse, and
 * triangular matrices at compile time.
 *
 * The provided template aliases cover:
 * - Complex type detection.
 * - Construction and manipulation of sparse matrix structures (row indices, row
 * pointers, column availability).
 * - Dense and diagonal matrix availability.
 * - Creation and transformation of sparse matrix representations.
 * - Concatenation, addition, subtraction, and multiplication of sparse
 * matrices.
 * - Extraction of matrix rows and transposition.
 * - Validation and utility operations for matrix structures.
 *
 * All templates are intended to be used as building blocks for
 * high-performance, type-safe matrix computations in C++ projects that require
 * Numpy-like functionality.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_TEMPLATES_HPP__
#define __PYTHON_NUMPY_TEMPLATES_HPP__

#include "base_matrix.hpp"

namespace PythonNumpy {

/* Complex Templates */
template <typename T> using Is_Complex_Type = Base::Matrix::Is_Complex_Type<T>;

/* Compiled Sparse Matrix Templates */
template <std::size_t... Sizes>
using RowIndices = Base::Matrix::RowIndices<Sizes...>;

template <std::size_t... Sizes>
using RowPointers = Base::Matrix::RowPointers<Sizes...>;

template <bool... Flags>
using ColumnAvailable = Base::Matrix::ColumnAvailable<Flags...>;

template <typename... Columns>
using SparseAvailable = Base::Matrix::SparseAvailable<Columns...>;

template <std::size_t M, std::size_t N>
using DenseAvailable = Base::Matrix::DenseAvailable<M, N>;

template <std::size_t M, std::size_t N>
using DenseAvailableEmpty = Base::Matrix::DenseAvailableEmpty<M, N>;

template <std::size_t M, std::size_t N>
using SparseAvailableEmpty = Base::Matrix::DenseAvailableEmpty<M, N>;

template <std::size_t M> using DiagAvailable = Base::Matrix::DiagAvailable<M>;

template <typename SparseAvailable>
using RowIndicesFromSparseAvailable =
    Base::Matrix::RowIndicesFromSparseAvailable<SparseAvailable>;

template <typename SparseAvailable>
using RowPointersFromSparseAvailable =
    Base::Matrix::RowPointersFromSparseAvailable<SparseAvailable>;

template <std::size_t N, typename RowIndices, typename RowPointers>
using CreateSparseAvailableFromIndicesAndPointers =
    Base::Matrix::CreateSparseAvailableFromIndicesAndPointers<N, RowIndices,
                                                              RowPointers>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableVertically =
    Base::Matrix::ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                       SparseAvailable_B>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableHorizontally =
    Base::Matrix::ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                         SparseAvailable_B>;

template <std::size_t M, std::size_t N>
using LowerTriangularRowIndices = Base::Matrix::LowerTriangularRowIndices<M, N>;

template <std::size_t M, std::size_t N>
using LowerTriangularRowPointers =
    Base::Matrix::LowerTriangularRowPointers<M, N>;

template <std::size_t M, std::size_t N>
using UpperTriangularRowIndices = Base::Matrix::UpperTriangularRowIndices<M, N>;

template <std::size_t M, std::size_t N>
using UpperTriangularRowPointers =
    Base::Matrix::UpperTriangularRowPointers<M, N>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using MatrixAddSubSparseAvailable =
    Base::Matrix::MatrixAddSubSparseAvailable<SparseAvailable_A,
                                              SparseAvailable_B>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using SparseAvailableMatrixMultiply =
    Base::Matrix::SparseAvailableMatrixMultiply<SparseAvailable_A,
                                                SparseAvailable_B>;

template <typename SparseAvailable>
using SparseAvailableTranspose =
    Base::Matrix::SparseAvailableTranspose<SparseAvailable>;

template <typename SparseAvailable>
using ValidateSparseAvailable =
    Base::Matrix::ValidateSparseAvailable<SparseAvailable>;

template <std::size_t M, typename SparseAvailable, std::size_t Index>
using SparseAvailableGetRow =
    Base::Matrix::SparseAvailableGetRow<M, SparseAvailable, Index>;

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_TEMPLATES_HPP__
