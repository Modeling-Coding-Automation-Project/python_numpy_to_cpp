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

    //Matrix<DefDense, Complex<double>, 3, 3> A;

    //auto solver = make_LinalgSolver(A);

    //std::cout << solver.IS_COMPLEX;


    return 0;
}
