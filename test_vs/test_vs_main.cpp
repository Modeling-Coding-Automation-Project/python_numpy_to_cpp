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

    auto A = make_DenseMatrix<12, 1>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    auto B = make_DenseMatrix<3, 4>(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    PythonNumpy::reshaped_copy(B, A);


    return 0;
}
