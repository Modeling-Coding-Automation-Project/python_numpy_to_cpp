#include "check_base_matrix.hpp"
#include "check_python_numpy.hpp"
#include "check_static_sparse.hpp"


int main() {

    CheckBaseMatrix<double> check_base_matrix_double;
    check_base_matrix_double.calc();

    CheckBaseMatrix<float> check_base_matrix_float;
    check_base_matrix_float.calc();

    CheckPythonNumpy<double> check_python_numpy_double;
    check_python_numpy_double.calc();

    CheckPythonNumpy<float> check_python_numpy_float;
    check_python_numpy_float.calc();

    StaticSparseTests::check_access<double>();
    StaticSparseTests::check_access<float>();


    return 0;
}
