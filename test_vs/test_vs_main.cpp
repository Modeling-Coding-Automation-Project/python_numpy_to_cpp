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

    //using A_S = DenseAvailable<3, 2>;
    //Matrix<DefSparse, double, 3, 2, A_S> A({ 4, 10, 5, 18, 6, 23 });

    //Matrix<DefDense, double, 3, 1> B({{1}, {2}, {3}});

    //auto C = ATranspose_mul_B(A, B);


    return 0;
}
