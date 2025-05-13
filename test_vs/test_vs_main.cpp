#include "check_base_matrix.hpp"
#include "check_python_numpy.hpp"


int main() {

    //CheckBaseMatrix<double> check_base_matrix_double;
    //check_base_matrix_double.calc();    

    //CheckBaseMatrix<float> check_base_matrix_float;
    //check_base_matrix_float.calc();

    //CheckPythonNumpy<double> check_python_numpy_double;
    //check_python_numpy_double.calc();

    //CheckPythonNumpy<float> check_python_numpy_float;
    //check_python_numpy_float.calc();


    using namespace PythonNumpy;

    auto A = make_DenseMatrix<3, 3>(1, 2, 3, 5, 4, 6, 9, 8, 7);
    auto B = make_DenseMatrix<3, 3>(0, 0, 0, 0, 0, 0, 0, 0, 0);

    PythonNumpy::reshaped_copy(B, A);


    return 0;
}
