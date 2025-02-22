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


    //using namespace PythonNumpy;

    //using SparseAvailable_C = SparseAvailable<
    //    ColumnAvailable<true, false, false, false>,
    //    ColumnAvailable<false, false, true, false>>;

    //auto A = make_DenseMatrixZeros<double, 4, 4>();
    //auto C = make_SparseMatrixZeros<double, SparseAvailable_C>();

    //auto A_mul_CT = A * C.transpose();

    //A_mul_CT = A_mul_BTranspose(A, C);


    return 0;
}
