#include "check_base_matrix.hpp"
#include "check_python_numpy.hpp"


int main() {

    CheckBaseMatrix<double> check_base_matrix_double;
    check_base_matrix_double.calc();

    CheckBaseMatrix<float> check_base_matrix_float;
    check_base_matrix_float.calc();

    CheckPythonNumpy<double> check_python_numpy_double;
    check_python_numpy_double.calc();

    CheckPythonNumpy<float> check_python_numpy_float;
    check_python_numpy_float.calc();


    //using namespace Base::Matrix;

    //using Name_A = DiagAvailable<4>;

    //using Name_B = SparseAvailable<
    //    ColumnAvailable<true, false, true, true>,
    //    ColumnAvailable<true, false, false, true>,
    //    ColumnAvailable<false, true, false, false>>;

    //using Name_B = SparseAvailable<
    //    ColumnAvailable<true>,
    //    ColumnAvailable<true>,
    //    ColumnAvailable<false>,
    //    ColumnAvailable<false>>;

    //using Name_C = SparseAvailableMatrixMultiplyTranspose<Name_A, Name_B>;

    //for (int i = 0; i < Name_C::number_of_columns; i++) {
    //    for (int j = 0; j < Name_C::column_size; j++) {
    //        std::cout << Name_C::lists[i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;


    //DiagMatrix<double, 4> A;

    //using RowIndices = RowIndicesFromSparseAvailable<Name_B>;
    //using RowPointers = RowPointersFromSparseAvailable<Name_B>;

    //CompiledSparseMatrix<double, 4, 1, RowIndices, RowPointers> B({ 1, 2 });
    //CompiledSparseMatrix<double, 3, 4, RowIndices, RowPointers> B({ 1, 2, 3, 4, 5, 6 });

    //auto B_T = output_matrix_transpose_2(B);
    //auto B_T_dense = output_dense_matrix(B_T);

    //for (int i = 0; i < B_T_dense.cols(); i++) {
    //    for (int j = 0; j < B_T_dense.rows(); j++) {
    //        std::cout << B_T_dense(i, j) << " ";
    //    }
    //    std::cout << std::endl;
    //}



    return 0;
}
