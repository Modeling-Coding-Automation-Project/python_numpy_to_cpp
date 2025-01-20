#ifndef __PYTHON_NUMPY_TEMPLATES_HPP__
#define __PYTHON_NUMPY_TEMPLATES_HPP__

#include "base_matrix.hpp"

namespace PythonNumpy {

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

template <std::size_t M, std::size_t N>
using SparseAvailable_NoUse = DenseAvailableEmpty<M, N>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using MatrixAddSubSparseAvailable =
    Base::Matrix::MatrixAddSubSparseAvailable<SparseAvailable_A,
                                              SparseAvailable_B>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using SparseAvailableMatrixMultiply =
    Base::Matrix::SparseAvailableMatrixMultiply<SparseAvailable_A,
                                                SparseAvailable_B>;

template <typename SparseAvailable_A, typename SparseAvailable_BT>
using SparseAvailableMatrixMultiplyTranspose =
    Base::Matrix::SparseAvailableMatrixMultiplyTranspose<SparseAvailable_A,
                                                         SparseAvailable_BT>;

template <typename SparseAvailable>
using SparseAvailableTranspose =
    Base::Matrix::SparseAvailableTranspose<SparseAvailable>;

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_TEMPLATES_HPP__
