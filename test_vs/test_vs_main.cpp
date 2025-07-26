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

    using namespace Base::Matrix;

    //using Row_Pointers_Test =
        //    typename TemplatesOperation::RepeatConcatenateIndexSequence<3,
        //    TemplatesOperation::IndexSequence<3>>::type;

    using Row_Pointers_Test = UpperTriangularRowPointers<3, 3>;

    for (size_t i = 0; i < Row_Pointers_Test::size; ++i) {
        std::cout << Row_Pointers_Test::list[i] << " ";
    }
    std::cout << std::endl;


    return 0;
}
