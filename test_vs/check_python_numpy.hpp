#ifndef __CHECK_PYTHON_NUMPY_HPP__
#define __CHECK_PYTHON_NUMPY_HPP__

#include <type_traits>
#include <iostream>

#include "MCAP_tester.hpp"

#include "python_numpy.hpp"
#include "base_matrix.hpp"

using namespace Tester;

template<typename T>
class CheckPythonNumpy {
public:
    /* Constructor */
    CheckPythonNumpy() {}

public:
    /* Function */
    void check_python_numpy_base(void);
    void check_python_numpy_base_simplification(void);
    void check_python_numpy_left_divide_and_inv(void);
    void check_python_numpy_concatenate(void);
    void check_python_numpy_transpose(void);
    void check_python_numpy_lu(void);
    void check_python_numpy_cholesky(void);
    void check_python_numpy_transpose_operation(void);
    void check_python_numpy_qr(void);
    void check_python_numpy_eig(void);

    void calc(void);
};

template <typename T>
void CheckPythonNumpy<T>::calc(void) {

    check_python_numpy_base();

    check_python_numpy_base_simplification();

    check_python_numpy_left_divide_and_inv();

    check_python_numpy_concatenate();

    check_python_numpy_transpose();

    check_python_numpy_lu();

    check_python_numpy_cholesky();

    check_python_numpy_transpose_operation();

    check_python_numpy_qr();

    check_python_numpy_eig();
}

template <typename T>
void CheckPythonNumpy<T>::check_python_numpy_base(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 配列代入 */
    T in[2][3];
    in[0][0] = 1.0F;
    in[0][1] = 2.0F;
    in[0][2] = 3.0F;
    in[1][0] = 4.0F;
    in[1][1] = 5.0F;
    in[1][2] = 6.0F;

    Matrix<DefDense, T, 2, 3> IN(in);

    T in_diag[3];
    in_diag[0] = 1.0F;
    in_diag[1] = 2.0F;
    in_diag[2] = 3.0F;

    Matrix<DefDiag, T, 3> IN_DIAG(in_diag);

    /* 基本 */
    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDense, T, 4, 3> AA({ { 1, 3, 0 }, {0, 0, 2}, {0, 8, 4}, {0, 1, 0} });

    T a_value = A.access(1, 2);

    tester.expect_near(a_value, 6, NEAR_LIMIT_STRICT,
        "check Matrix get value.");

    T a_value_outlier = A(100, 100);

    tester.expect_near(a_value_outlier, 7, NEAR_LIMIT_STRICT,
        "check Matrix get value outlier.");

    A.access(1, 2) = 100;

    Matrix<DefDense, T, 3, 3> A_set_answer({
        {1, 2, 3},
        {5, 4, 100},
        {9, 8, 7}
        });

    tester.expect_near(A.matrix.data, A_set_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Matrix set value.");

    A(1, 2) = static_cast<T>(6);

    A.template set<1, 2>(static_cast<T>(100));

    tester.expect_near(A.matrix.data, A_set_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Matrix set value template.");

    T A_template_value = A.template get<1, 2>();

    tester.expect_near(A_template_value, 100, NEAR_LIMIT_STRICT,
        "check Matrix get value template.");

    A.template set<1, 2>(static_cast<T>(6));

    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefDense, T, 4, 2> BB({ { 1, 2 }, {3, 4}, {5, 6}, {7, 8} });

    T b_value = B.access(1);

    tester.expect_near(b_value, 2, NEAR_LIMIT_STRICT,
        "check DiagMatrix get value.");

    T b_value_outlier = B(100);

    tester.expect_near(b_value_outlier, 3, NEAR_LIMIT_STRICT,
        "check DiagMatrix get value outlier.");

    B.access(1) = 100;

    Matrix<DefDiag, T, 3> B_set_answer({ 1, 100, 3 });

    tester.expect_near(B.matrix.data, B_set_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix set value.");

    B(1) = static_cast<T>(2);

    T B_template_value = B.template get<1, 1>();

    tester.expect_near(B_template_value, 2, NEAR_LIMIT_STRICT,
        "check DiagMatrix get value template.");

    T B_template_value_2 = B.template get<1, 2>();

    tester.expect_near(B_template_value_2, 0, NEAR_LIMIT_STRICT,
        "check DiagMatrix get value template 0.");

    B.template set<1, 1>(static_cast<T>(100));

    Matrix<DefDiag, T, 3> B_set_answer_2({ 1, 100, 3 });

    tester.expect_near(B.matrix.data, B_set_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix set value template.");

    B.template set<1, 1>(static_cast<T>(2));

    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });

    Matrix<DefSparse, T, 3, 3,
        SparseAvailableEmpty<3, 3>> Empty;

    T c_value = C.access(2);

    tester.expect_near(c_value, 8, NEAR_LIMIT_STRICT,
        "check SparseMatrix get value.");

    T c_value_outlier = C(100);

    tester.expect_near(c_value_outlier, 4, NEAR_LIMIT_STRICT,
        "check SparseMatrix get value outlier.");

    C.access(2) = 100;

    Matrix<DefDense, T, 3, 3> C_set_dense_1 = C.create_dense();
    Matrix<DefDense, T, 3, 3> C_set_answer_1({
        {1, 0, 0},
        {3, 0, 100},
        {0, 2, 4}
        });

    tester.expect_near(C_set_dense_1.matrix.data, C_set_answer_1.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix set value.");

    C(2) = static_cast<T>(8);

    T C_val = C.template get<1, 2>();

    tester.expect_near(C_val, 8, NEAR_LIMIT_STRICT,
        "check SparseMatrix get value.");

    C.template set<2, 2>(static_cast<T>(100));
    auto C_set_dense = C.create_dense();

    Matrix<DefDense, T, 3, 3> C_set_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 100}
        });

    tester.expect_near(C_set_dense.matrix.data, C_set_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix set value.");

    C.template set<2, 2>(static_cast<T>(4));

    Empty.template set<1, 1>(static_cast<T>(100));

    T empty_value = Empty.template get<1, 1>();

    tester.expect_near(empty_value, static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check SparseMatrix get value empty.");

    static_assert(std::is_same<typename Matrix<DefDense, T, 3, 3>::Value_Type, T>::value,
        "check Dense Matrix get value type.");

    static_assert(std::is_same<typename Matrix<DefDiag, T, 3>::Value_Type, T>::value,
        "check Diag Matrix get value type.");

    static_assert(std::is_same<typename Matrix<DefSparse, T, 3, 3, DenseAvailable<3, 3>>::Value_Type, T>::value,
        "check Sparse Matrix get value type.");


    /* 演算 */
    Matrix<DefDense, T, 3, 3> A_minus = -A;

    Matrix<DefDense, T, 3, 3> A_minus_answer({
        { -1, -2, -3 },
        { -5, -4, -6 },
        { -9, -8, -7 }
        });

    tester.expect_near(A_minus.matrix.data, A_minus_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Dense Matrix minus.");

    Matrix<DefDiag, T, 3> B_minus = -B;

    Matrix<DefDiag, T, 3> B_minus_answer({ -1, -2, -3 });

    tester.expect_near(B_minus.matrix.data, B_minus_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Diag Matrix minus.");

    decltype(C) C_minus = -C;
    auto C_minus_dense = C_minus.create_dense();

    Matrix<DefDense, T, 3, 3> C_minus_answer({
        { -1, 0, 0 },
        { -3, 0, -8 },
        { 0, -2, -4 }
        });

    tester.expect_near(C_minus_dense.matrix.data, C_minus_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Sparse Matrix minus.");


    Matrix<DefDiag, T, 3> DiagJ({ 10, 20, 30 });

    auto Dense_add_Diag = A + DiagJ;

    Matrix<DefDense, T, 3, 3> Dense_add_Diag_answer({
        {11, 2, 3},
        {5, 24, 6},
        {9, 8, 37}
        });

    tester.expect_near(Dense_add_Diag.matrix.data, Dense_add_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DenseMatrix add DiagMatrix.");

    auto Diag_add_Dense = DiagJ + A;

    tester.expect_near(Diag_add_Dense.matrix.data, Dense_add_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix add DenseMatrix.");

    auto Dense_minus_Diag = A - DiagJ;

    Matrix<DefDense, T, 3, 3> Dense_minus_Diag_answer({
        { -9, 2, 3 },
        { 5, -16, 6 },
        { 9, 8, -23 }
        });

    tester.expect_near(Dense_minus_Diag.matrix.data, Dense_minus_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DenseMatrix minus DiagMatrix.");

    auto Diag_minus_Dense = DiagJ - A;

    Matrix<DefDense, T, 3, 3> Diag_minus_Dense_answer({
        { 9, -2, -3 },
        { -5, 16, -6 },
        { -9, -8, 23 }
        });

    tester.expect_near(Diag_minus_Dense.matrix.data, Diag_minus_Dense_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix minus DenseMatrix.");


    auto Diag_add_Diag = B + DiagJ;
    auto Diag_add_Diag_dense = Diag_add_Diag.create_dense();

    Matrix<DefDense, T, 3, 3> Diag_add_Diag_answer({
        {11, 0, 0},
        {0, 22, 0},
        {0, 0, 33}
        });

    tester.expect_near(Diag_add_Diag_dense.matrix.data, Diag_add_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix add DiagMatrix.");

    auto Diag_minus_Diag = B - DiagJ;
    auto Diag_minus_Diag_dense = Diag_minus_Diag.create_dense();

    Matrix<DefDense, T, 3, 3> Diag_minus_Diag_answer({
        { -9, 0, 0 },
        { 0, -18, 0 },
        { 0, 0, -27 }
        });

    tester.expect_near(Diag_minus_Diag_dense.matrix.data, Diag_minus_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix minus DiagMatrix.");


    auto Sparse_add_Diag = C + DiagJ;
    auto Sparse_add_Diag_dense = Sparse_add_Diag.create_dense();

    Matrix<DefDense, T, 3, 3> Sparse_add_Diag_answer({
        {11, 0, 0},
        {3, 20, 8},
        {0, 2, 34}
        });

    tester.expect_near(Sparse_add_Diag_dense.matrix.data, Sparse_add_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add DiagMatrix.");

    auto Diag_add_Sparse = DiagJ + C;
    auto Diag_add_Sparse_dense = Diag_add_Sparse.create_dense();

    tester.expect_near(Diag_add_Sparse_dense.matrix.data, Sparse_add_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix add SparseMatrix.");

    auto Sparse_sub_Diag = C - DiagJ;
    auto Sparse_sub_Diag_dense = Sparse_sub_Diag.create_dense();

    Matrix<DefDense, T, 3, 3> Sparse_sub_Diag_answer({
        {-9, 0, 0},
        {3, -20, 8},
        {0, 2, -26}
        });

    tester.expect_near(Sparse_sub_Diag_dense.matrix.data, Sparse_sub_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub DiagMatrix.");

    auto Diag_sub_Sparse = DiagJ - C;
    auto Diag_sub_Sparse_dense = Diag_sub_Sparse.create_dense();

    Matrix<DefDense, T, 3, 3> Diag_sub_Sparse_answer({
        {9, 0, 0},
        {-3, 20, -8},
        {0, -2, 26}
        });

    tester.expect_near(Diag_sub_Sparse_dense.matrix.data, Diag_sub_Sparse_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix sub SparseMatrix.");

    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, true, true>,
        ColumnAvailable<false, true, true>,
        ColumnAvailable<false, false, true>>
        > U({ 1, 2, 3, 4, 5, 6 });

    auto Sparse_add_Sparse = C + U;
    auto Sparse_add_Sparse_dense = Sparse_add_Sparse.create_dense();

    Matrix<DefDense, T, 3, 3> Sparse_add_Sparse_answer({
        {2, 2, 3},
        {3, 4, 13},
        {0, 2, 10}
        });

    tester.expect_near(Sparse_add_Sparse_dense.matrix.data, Sparse_add_Sparse_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add SparseMatrix.");

    auto Sparse_sub_Sparse = C - U;
    auto Sparse_sub_Sparse_dense = Sparse_sub_Sparse.create_dense();

    Matrix<DefDense, T, 3, 3> Sparse_sub_Sparse_answer({
        {0, -2, -3},
        {3, -4, 3},
        {0, 2, -2}
        });

    tester.expect_near(Sparse_sub_Sparse_dense.matrix.data, Sparse_sub_Sparse_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub SparseMatrix.");

    auto Dense_mul_Dense = AA * AA.transpose();

    Matrix<DefDense, T, 4, 4> Dense_mul_Dense_answer({
        { 10, 0, 24, 3 },
        { 0, 4, 8, 0 },
        { 24, 8, 80, 8 },
        { 3, 0, 8, 1 }
        });

    tester.expect_near(Dense_mul_Dense.matrix.data, Dense_mul_Dense_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DenseMatrix multiply DenseMatrix.");

    A_Multiply_B_Type<decltype(AA), decltype(B)>
        Dense_mul_Diag = AA * B;

    Matrix<DefDense, T, 4, 3> Dense_mul_Diag_answer({
        { 1, 6, 0 },
        { 0, 0, 6 },
        { 0, 16, 12 },
        { 0, 2, 0 }
        });

    tester.expect_near(Dense_mul_Diag.matrix.data, Dense_mul_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DenseMatrix multiply DiagMatrix.");

    auto Dense_mul_Sparse = AA * C;

    Matrix<DefDense, T, 4, 3> Dense_mul_Sparse_answer({
        { 10, 0, 24 },
        { 0, 4, 8 },
        { 24, 8, 80 },
        { 3, 0, 8 }
        });

    tester.expect_near(Dense_mul_Sparse.matrix.data, Dense_mul_Sparse_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DenseMatrix multiply SparseMatrix.");

    auto Diag_mul_Dense = B * AA.transpose();

    Matrix<DefDense, T, 3, 4> Diag_mul_Dense_answer({
        { 1, 0, 0, 0 },
        { 6, 0, 16, 2 },
        { 0, 6, 12, 0 }
        });

    tester.expect_near(Diag_mul_Dense.matrix.data, Diag_mul_Dense_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply DenseMatrix.");


    auto Diag_mul_Diag = B * B;
    auto Diag_mul_Diag_dense = Diag_mul_Diag.create_dense();

    Matrix<DefDense, T, 3, 3> Diag_mul_Diag_answer({
        {1, 0, 0},
        {0, 4, 0},
        {0, 0, 9}
        });

    tester.expect_near(Diag_mul_Diag_dense.matrix.data, Diag_mul_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply DiagMatrix.");

    Matrix<DefSparse, T, 3, 4,
        SparseAvailable<
        ColumnAvailable<true, false, false, false>,
        ColumnAvailable<true, false, true, true>,
        ColumnAvailable<false, true, true, false>
        >> CL({ 1, 3, 8, 1, 2, 4 });

    auto Diag_mul_Sparse = B * CL;
    auto Diag_mul_Sparse_dense = Diag_mul_Sparse.create_dense();

    Matrix<DefDense, T, 3, 4> Diag_mul_Sparse_answer({
        {1, 0, 0, 0},
        {6, 0, 16, 2},
        {0, 6, 12, 0}
        });

    tester.expect_near(Diag_mul_Sparse_dense.matrix.data, Diag_mul_Sparse_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply SparseMatrix.");

    auto Sparse_mul_Dense = C * AA.transpose();

    Matrix<DefDense, T, 3, 4> Sparse_mul_Dense_answer({
        { 1, 0, 0, 0 },
        { 3, 16, 32, 0 },
        { 6, 8, 32, 2 }
        });

    tester.expect_near(Sparse_mul_Dense.matrix.data, Sparse_mul_Dense_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply DenseMatrix.");

    Matrix<DefDiag, T, 4> BL({ 1, 2, 3, 4 });

    auto Sparse_mul_Diag = CL * BL;
    auto Sparse_mul_Diag_dense = Sparse_mul_Diag.create_dense();

    Matrix<DefDense, T, 3, 4> Sparse_mul_Diag_answer({
        { 1, 0, 0, 0 },
        { 3, 0, 24, 4 },
        { 0, 4, 12, 0 }
        });

    tester.expect_near(Sparse_mul_Diag_dense.matrix.data, Sparse_mul_Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply DiagMatrix.");

    auto Sparse_mul_Sparse = C * CL;
    auto Sparse_mul_Sparse_dense = Sparse_mul_Sparse.create_dense();

    Matrix<DefDense, T, 3, 4> Sparse_mul_Sparse_answer({
        { 1, 0, 0, 0 },
        { 3, 16, 32, 0 },
        { 6, 8, 32, 2 }
        });

    tester.expect_near(Sparse_mul_Sparse_dense.matrix.data, Sparse_mul_Sparse_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply SparseMatrix.");

    auto A_add_E = A + Empty;

    Matrix<DefDense, T, 3, 3> A_add_E_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    tester.expect_near(A_add_E.matrix.data, A_add_E_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Matrix add EmptyMatrix.");

    auto A_sub_E = A - Empty;

    tester.expect_near(A_sub_E.matrix.data, A_add_E_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Matrix sub EmptyMatrix.");

    auto A_mul_E = A * Empty;

    Matrix<DefDense, T, 3, 3> A_mul_E_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(A_mul_E.matrix.data, A_mul_E_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply EmptyMatrix.");


    /* Empty 演算 補足 */
    Matrix<DefSparse, T, 4, 1, SparseAvailableEmpty<4, 1>> SS_D;

    Matrix<DefSparse, T, 1, 1, DenseAvailable<1, 1>> SS_U;

    auto SS_D_mul_SS_U = SS_D * SS_U;
    auto SS_D_mul_SS_U_dense = SS_D_mul_SS_U.create_dense();

    Matrix<DefDense, T, 4, 1> SS_D_mul_SS_U_answer({
        {0},
        {0},
        {0},
        {0}
        });

    tester.expect_near(SS_D_mul_SS_U_dense.matrix.data, SS_D_mul_SS_U_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check EmptyMatrix multiply EmptyMatrix SS.");

    /* 複素数 */
    Matrix<DefDense, Complex<T>, 3, 3> A_complex = A.create_complex();

    Matrix<DefDense, Complex<T>, 3, 3> A_complex_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            tester.expect_near(A_complex.matrix.data[j][i].real, A_complex_answer.matrix.data[j][i].real, NEAR_LIMIT_STRICT,
                "check Dense Matrix create complex real.");
            tester.expect_near(A_complex.matrix.data[j][i].imag, A_complex_answer.matrix.data[j][i].imag, NEAR_LIMIT_STRICT,
                "check Dense Matrix create complex imag.");
        }
    }

    Matrix<DefDiag, Complex<T>, 3> B_complex = B.create_complex();

    Matrix<DefDiag, Complex<T>, 3> B_complex_answer({ 1, 2, 3 });

    for (int i = 0; i < 3; i++) {
        tester.expect_near(B_complex.matrix.data[i].real, B_complex_answer.matrix.data[i].real, NEAR_LIMIT_STRICT,
            "check Diag Matrix create complex real.");
        tester.expect_near(B_complex.matrix.data[i].imag, B_complex_answer.matrix.data[i].imag, NEAR_LIMIT_STRICT,
            "check Diag Matrix create complex imag.");
    }

    Matrix<DefSparse, Complex<T>, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C_complex = C.create_complex();

    Matrix<DefSparse, Complex<T>, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C_complex_answer({ 1, 3, 8, 2, 4 });

    for (int i = 0; i < 5; i++) {
        tester.expect_near(C_complex.matrix.values[i].real, C_complex_answer.matrix.values[i].real, NEAR_LIMIT_STRICT,
            "check Sparse Matrix create complex real.");
        tester.expect_near(C_complex.matrix.values[i].imag, C_complex_answer.matrix.values[i].imag, NEAR_LIMIT_STRICT,
            "check Sparse Matrix create complex imag.");
    }


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckPythonNumpy<T>::check_python_numpy_base_simplification(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    DenseMatrix_Type<T, 4, 3> Zeros = make_DenseMatrixZeros<T, 4, 3>();

    Matrix<DefDense, T, 4, 3> Zeros_answer;

    tester.expect_near(Zeros.matrix.data, Zeros_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_MatrixZeros.");

    auto Ones = make_DenseMatrixOnes<T, 4, 3>();

    Matrix<DefDense, T, 4, 3> Ones_answer({
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
        });

    tester.expect_near(Ones.matrix.data, Ones_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DenseMatrixOnes.");

    auto Full = make_DenseMatrixFull<3, 3>(static_cast<T>(2));

    Matrix<DefDense, T, 3, 3> Full_answer({
        {2, 2, 2},
        {2, 2, 2},
        {2, 2, 2}
        });

    tester.expect_near(Full.matrix.data, Full_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DenseMatrixFull.");


    auto Identity = make_DiagMatrixIdentity<T, 3>();
    auto Identity_dense = Identity.create_dense();

    Matrix<DefDense, T, 3, 3> Identity_answer({
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
        });

    tester.expect_near(Identity_dense.matrix.data, Identity_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DiagMatrixIdentity.");

    auto Diag_Full = make_DiagMatrixFull<3>(static_cast<T>(2));
    auto Diag_Full_dense = Diag_Full.create_dense();

    Matrix<DefDense, T, 3, 3> Diag_Full_answer({
        {2, 0, 0},
        {0, 2, 0},
        {0, 0, 2}
        });

    tester.expect_near(Diag_Full_dense.matrix.data, Diag_Full_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DiagMatrixFull.");


    SparseMatrixEmpty_Type<T, 3, 4> Empty = make_SparseMatrixEmpty<T, 3, 4>();
    auto Empty_dense = Empty.create_dense();

    Matrix<DefDense, T, 3, 4> Empty_answer;

    tester.expect_near(Empty_dense.matrix.data, Empty_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_SparseMatrixEmpty.");

    DiagMatrix_Type<T, 3> Diag = make_DiagMatrix<3>(
        static_cast<T>(1),
        static_cast<T>(2),
        static_cast<T>(3));

    Matrix<DefDiag, T, 3> Diag_answer({ 1, 2, 3 });

    tester.expect_near(Diag.matrix.data, Diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DiagMatrix.");

    DiagMatrix_Type<T, 3> Diag_zeros = make_DiagMatrixZeros<T, 3>();

    Matrix<DefDiag, T, 3> Diag_zeros_answer({ 0, 0, 0 });

    tester.expect_near(Diag_zeros.matrix.data, Diag_zeros_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DiagMatrixZeros.");

    auto Dense = make_DenseMatrix<3, 3>(
        static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
        static_cast<T>(5), static_cast<T>(4), static_cast<T>(6),
        static_cast<T>(9), static_cast<T>(8), static_cast<T>(7));

    Matrix<DefDense, T, 3, 3> Dense_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    tester.expect_near(Dense.matrix.data, Dense_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DenseMatrix.");

    auto Dense_2 = make_DenseMatrix<2, 4>(
        static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(5),
        static_cast<T>(4), static_cast<T>(6), static_cast<T>(9), static_cast<T>(8)
    );

    Matrix<DefDense, T, 2, 4> Dense_2_answer({
        {1, 2, 3, 5},
        {4, 6, 9, 8}
        });

    tester.expect_near(Dense_2.matrix.data, Dense_2_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_DenseMatrix 2x4.");


    using SparseAvailable_C = SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>;

    auto SparseZeros = make_SparseMatrixZeros<T, SparseAvailable_C>();
    auto SparseZeros_dense = SparseZeros.create_dense();

    Matrix<DefDense, T, 3, 3> SparseZeros_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(SparseZeros_dense.matrix.data, SparseZeros_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_SparseMatrixZeros.");

    auto SparseOnes = make_SparseMatrixOnes<T, SparseAvailable_C>();
    auto SparseOnes_dense = SparseOnes.create_dense();

    Matrix<DefDense, T, 3, 3> SparseOnes_answer({
        {1, 0, 0},
        {1, 0, 1},
        {0, 1, 1}
        });

    tester.expect_near(SparseOnes_dense.matrix.data, SparseOnes_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_SparseMatrixOnes.");

    auto SparseFull = make_SparseMatrixFull<SparseAvailable_C>(static_cast<T>(2));
    auto SparseFull_dense = SparseFull.create_dense();

    Matrix<DefDense, T, 3, 3> SparseFull_answer({
        {2, 0, 0},
        {2, 0, 2},
        {0, 2, 2}
        });

    tester.expect_near(SparseFull_dense.matrix.data, SparseFull_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_SparseMatrixFull.");

    SparseMatrix_Type<T, SparseAvailable_C> Sparse
        = make_SparseMatrix<SparseAvailable_C>(
            static_cast<T>(1),
            static_cast<T>(3),
            static_cast<T>(8),
            static_cast<T>(2),
            static_cast<T>(4));
    auto Sparse_dense = Sparse.create_dense();

    Matrix<DefDense, T, 3, 3> Sparse_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4}
        });

    tester.expect_near(Sparse_dense.matrix.data, Sparse_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_SparseMatrix.");

    T sparse_value = Sparse.template get<3>();

    tester.expect_near(sparse_value, static_cast<T>(2), NEAR_LIMIT_STRICT,
        "check SparseMatrix get value.");

    auto CD = make_SparseMatrixFromDenseMatrix(Dense);
    auto CD_dense = CD.create_dense();

    tester.expect_near(CD_dense.matrix.data, Dense_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check make_SparseMatrixFromDenseMatrix.");

    /* Get */
    auto Dense_row = get_row<1>(Dense);

    Matrix<DefDense, T, 3, 1> Dense_row_answer({
        {2},
        {4},
        {8}
        });

    tester.expect_near(Dense_row.matrix.data, Dense_row_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check get_row.");

    auto Diag_Full_row = get_row<1>(Diag_Full);
    auto Diag_Full_row_dense = Diag_Full_row.create_dense();

    Matrix<DefDense, T, 3, 1> Diag_Full_row_answer({
        { 0 },
        { 2 },
        { 0 }
    });

    tester.expect_near(Diag_Full_row_dense.matrix.data, Diag_Full_row_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check get_row DiagMatrix.");

    auto Sparse_row = get_row<1>(Sparse);
    auto Sparse_row_dense = Sparse_row.create_dense();

    Matrix<DefDense, T, 3, 1> Sparse_row_answer({
        { 0 },
        { 0 },
        { 2 }
    });

    tester.expect_near(Sparse_row_dense.matrix.data, Sparse_row_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check get_row SparseMatrix.");


    /* Set */
    auto row_vector = make_DenseMatrix<3, 1>(
        static_cast<T>(1), static_cast<T>(2), static_cast<T>(3));

    auto Dense_set_row = Dense;
    set_row<1>(Dense_set_row, row_vector);

    Matrix<DefDense, T, 3, 3> Dense_set_row_answer({
        {1, 1, 3},
        {5, 2, 6},
        {9, 3, 7}
    });

    tester.expect_near(Dense_set_row.matrix.data, Dense_set_row_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check set_row.");

    auto Diag_set_row = Diag_Full;

    set_row<2>(Diag_set_row, row_vector);
    auto Diag_set_row_dense = Diag_set_row.create_dense();

    Matrix<DefDense, T, 3, 3> Diag_set_row_answer({
        {2, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
    });

    tester.expect_near(Diag_set_row_dense.matrix.data, Diag_set_row_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check set_row DiagMatrix.");

    auto Sparse_set_row = Sparse;

    set_row<1>(Sparse_set_row, row_vector);
    auto Sparse_set_row_dense = Sparse_set_row.create_dense();

    Matrix<DefDense, T, 3, 3> Sparse_set_row_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 3, 4}
    });

    tester.expect_near(Sparse_set_row_dense.matrix.data, Sparse_set_row_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check set_row SparseMatrix.");

    /* reshape */
    Matrix<DefDense, T, 1, 12> A_reshape({
        {1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F}
    });
    Matrix<DefDense, T, 3, 4> B_reshape;

    update_reshaped_matrix(B_reshape, A_reshape);

    Matrix<DefDense, T, 3, 4> B_reshape_answer({
        {1.0F, 4.0F, 7.0F, 10.0F},
        {2.0F, 5.0F, 8.0F, 11.0F},
        {3.0F, 6.0F, 9.0F, 12.0F}
        });

    tester.expect_near(B_reshape.matrix.data, B_reshape_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check update_reshaped_matrix, 1, 12 to 3, 4");

    Matrix<DefDense, T, 12, 1> A_reshape_2({
        {1.0F}, {2.0F}, {3.0F}, {4.0F},
        {5.0F}, {6.0F}, {7.0F}, {8.0F},
        {9.0F}, {10.0F}, {11.0F}, {12.0F} 
    });
    Matrix<DefDense, T, 3, 4> B_reshape_2;

    Matrix<DefDense, T, 3, 4> B_reshape_answer_2({
        {1.0F, 4.0F, 7.0F, 10.0F},
        {2.0F, 5.0F, 8.0F, 11.0F},
        {3.0F, 6.0F, 9.0F, 12.0F}
    });

    update_reshaped_matrix(B_reshape_2, A_reshape_2);

    tester.expect_near(B_reshape_2.matrix.data, B_reshape_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update_reshaped_matrix, 12, 1 to 3, 4");

    Matrix<DefDense, T, 4, 3> A_reshape_3({
        {1.0F, 2.0F, 3.0F},
        {4.0F, 5.0F, 6.0F},
        {7.0F, 8.0F, 9.0F},
        {10.0F, 11.0F, 12.0F}
    });
    Matrix<DefDense, T, 3, 4> B_reshape_3;

    update_reshaped_matrix(B_reshape_3, A_reshape_3);

    Matrix<DefDense, T, 3, 4> B_reshape_answer_3({
        {1.0F, 10.0F, 8.0F, 6.0F},
        {4.0F, 2.0F, 11.0F, 9.0F},
        {7.0F, 5.0F, 3.0F, 12.0F}
        });

    tester.expect_near(B_reshape_3.matrix.data, B_reshape_answer_3.matrix.data, NEAR_LIMIT_STRICT,
        "check update_reshaped_matrix, 4, 3 to 3, 4");

    auto A_reshape_4 = reshape<6, 2>(A_reshape_3);

    Matrix<DefDense, T, 6, 2> A_reshape_answer_4({
        {1.0F, 8.0F},
        {4.0F, 11.0F},
        {7.0F, 3.0F},
        {10.0F, 6.0F},
        {2.0F, 9.0F},
        {5.0F, 12.0F}
        });

    tester.expect_near(A_reshape_4.matrix.data, A_reshape_answer_4.matrix.data, NEAR_LIMIT_STRICT,
        "check reshape, 4, 3 to 6, 2");


    /* 違うタイプ同士の行列を代入 */
    auto B = make_DiagMatrixIdentity<T, 3>();

    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });

    substitute_matrix(C, B);
    auto C_dense = C.create_dense();

    auto C_answer = make_DenseMatrix<3, 3>(
        static_cast<T>(1), static_cast<T>(0), static_cast<T>(0),
        static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
        static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));

    tester.expect_near(C_dense.matrix.data, C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check substitute_matrix.");

    /* 大きい行列に対して小さい行列を代入 */
    Matrix<DefDense, T, 4, 4> A_P({
        {1, 2, 3, 10},
        {5, 4, 6, 11},
        {9, 8, 7, 12},
        {13, 14, 15, 16}
        });

    Matrix<DefDense, T, 2, 2> B_P({
        {1, 2},
        {3, 4}
        });

    substitute_part_matrix<1, 1>(A_P, B_P);

    Matrix<DefDense, T, 4, 4> A_P_answer({
        {1, 2, 3, 10},
        {5, 1, 2, 11},
        {9, 3, 4, 12},
        {13, 14, 15, 16}
        });

    tester.expect_near(A_P.matrix.data, A_P_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check substitute_part_matrix, 4, 4 to 2, 2 at 1, 1");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckPythonNumpy<T>::check_python_numpy_left_divide_and_inv(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    const T NEAR_LIMIT_SOFT = 1.0e-3F;

    /* 定義 */
    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });


    /* 左除算 */
    Matrix<DefDense, T, 3, 2> b({ { 4, 10 }, { 5, 18 }, { 6, 23 } });

    LinalgSolver_Type<decltype(A), decltype(b)> A_b_linalg_solver
        = make_LinalgSolver<decltype(A), decltype(b)>();

    auto A_b_x = A_b_linalg_solver.solve(A, b);

    Matrix<DefDense, T, 3, 2> A_b_x_answer({
        { -1.0F, -0.66666667F },
        { 1.0F, 1.23333333F },
        { 1.0F, 2.73333333F }
        });

    tester.expect_near(A_b_x.matrix.data, A_b_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Dense small.");

    Matrix<DefDense, T, 3, 3> I({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
    });

    Matrix<DefDense, T, 3, 1> b_1({
        { 2.0F },
        { 3.0F },
        { 1.0F }
        });

    LinalgSolver_Type<decltype(I), decltype(b_1)> I_b_linalg_solver
        = make_LinalgSolver<decltype(I), decltype(b_1)>();

    I_b_linalg_solver.set_division_min(static_cast<T>(1.0e-5));

    auto I_b_x = I_b_linalg_solver.solve(I, b_1);

    Matrix<DefDense, T, 3, 1> I_b_x_answer({
        { 2.0F },
        { 3.0F },
        { 1.0F }
    });

    tester.expect_near(I_b_x.matrix.data, I_b_x_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolver solve Dense Identity and Vector.");

    LinalgSolver_Type<decltype(A), decltype(A)> A_A_linalg_solver
        = make_LinalgSolver<decltype(A), decltype(A)>();

    A_A_linalg_solver.set_division_min(static_cast<T>(1.0e-10));
    A_A_linalg_solver.set_decay_rate(static_cast<T>(0));

    LinalgSolver_Type<decltype(A), decltype(A)> A_A_linalg_solver_copy = A_A_linalg_solver;
    LinalgSolver_Type<decltype(A), decltype(A)> A_A_linalg_solver_move = std::move(A_A_linalg_solver_copy);
    A_A_linalg_solver = A_A_linalg_solver_move;

    auto A_A_x = A_A_linalg_solver.solve(A, A);

    Matrix<DefDense, T, 3, 3> A_A_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(A_A_x.matrix.data, A_A_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Dense.");

    static auto A_B_linalg_solver = make_LinalgSolver<decltype(A), decltype(B)>();

    auto A_B_x = A_B_linalg_solver.solve(A, B);

    Matrix<DefDense, T, 3, 3> A_B_x_answer({
        {-6.66666667e-01F, 6.66666667e-01F, 0.0F},
        {6.33333333e-01F, -1.33333333F, 0.9F},
        {1.33333333e-01F, 6.66666667e-01F, -0.6F}
        });

    tester.expect_near(A_B_x.matrix.data, A_B_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Diag.");

    static auto A_C_linalg_solver = make_LinalgSolver<decltype(A), decltype(C)>();

    auto A_C_x = A_C_linalg_solver.solve(A, C);

    Matrix<DefDense, T, 3, 3> A_C_x_answer({
        { 3.33333333e-01F, 0.0F, 2.66666667F},
        {-1.36666667F, 0.6F, -4.13333333F},
        {1.13333333F, -0.4F, 1.86666667F}
        });

    tester.expect_near(A_C_x.matrix.data, A_C_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Sparse.");

    static auto B_A_linalg_solver = make_LinalgSolver<decltype(B), decltype(A)>();

    auto B_A_x = B_A_linalg_solver.solve(B, A);

    Matrix<DefDense, T, 3, 3> B_A_x_answer({
        { 1.0F, 2.0F, 3.0F },
        { 2.5F, 2.0F, 3.0F },
        { 3.0F, 2.66666667F, 2.33333333F }
        });

    tester.expect_near(B_A_x.matrix.data, B_A_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Dense.");

    static auto B_B_linalg_solver = make_LinalgSolver<decltype(B), decltype(B)>();

    auto B_B_x = B_B_linalg_solver.solve(B, B);
    auto B_B_x_dense = B_B_x.create_dense();

    Matrix<DefDense, T, 3, 3> B_B_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(B_B_x_dense.matrix.data, B_B_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Diag.");

    static auto B_C_linalg_solver = make_LinalgSolver<decltype(B), decltype(C)>();

    auto B_C_x = B_C_linalg_solver.solve(B, C);

    Matrix<DefDense, T, 3, 3> B_C_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 1.5F, 0.0F, 4.0F },
        { 0.0F, 0.66666667F, 1.33333333F }
        });

    tester.expect_near(B_C_x.matrix.data, B_C_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Sparse.");

    static auto C_C_linalg_solver = make_LinalgSolver<decltype(C), decltype(C)>();

    auto C_C_x = C_C_linalg_solver.solve(C, C);

    Matrix<DefDense, T, 3, 3> C_C_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(C_C_x.matrix.data, C_C_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Sparse and Sparse.");

    /* 部分的 左除算 */
    Matrix<DefDense, T, 4, 4> A_P({
        {1, 2, 3, 10},
        {5, 4, 6, 11},
        {9, 8, 7, 12},
        {13, 14, 15, 16}
        });
    Matrix<DefDiag, T, 4> B_P({ 1, 2, 3, 4 });
    Matrix<DefSparse, T, 4, 4,
        SparseAvailable<
        ColumnAvailable<true, false, false, true>,
        ColumnAvailable<true, false, true, false>,
        ColumnAvailable<false, true, true, true>,
        ColumnAvailable<false, true, false, true>>
        > C_P({ 1, 10, 3, 8, 2, 4, 11, 12, 13 });

    LinalgPartitionSolver_Type<decltype(A_P), decltype(A_P)> A_A_P_linalg_solver
        = make_LinalgPartitionSolver<decltype(A_P), decltype(A_P)>();

    A_A_P_linalg_solver.set_division_min(static_cast<T>(1.0e-10));
    A_A_P_linalg_solver.set_decay_rate(static_cast<T>(0));

    LinalgPartitionSolver_Type<decltype(A_P), decltype(A_P)> A_A_P_linalg_solver_copy
        = A_A_P_linalg_solver;
    LinalgPartitionSolver_Type<decltype(A_P), decltype(A_P)> A_A_P_linalg_solver_move
        = std::move(A_A_P_linalg_solver_copy);
    A_A_P_linalg_solver = A_A_P_linalg_solver_move;

    auto A_A_P_x = A_A_P_linalg_solver.solve(A_P, A_P, 3);

    Matrix<DefDense, T, 4, 4> A_A_P_x_answer({
        { 1.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F }
        });

    tester.expect_near(A_A_P_x.matrix.data, A_A_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Dense and Dense.");

    A_A_P_x = A_A_P_linalg_solver.cold_solve(A_P, A_P, 3);

    tester.expect_near(A_A_P_x.matrix.data, A_A_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Dense and Dense cold solve.");

    static auto A_B_P_linalg_solver = make_LinalgPartitionSolver<decltype(A_P), decltype(B_P)>();

    auto A_B_P_x = A_B_P_linalg_solver.solve(A_P, B_P, 3);

    Matrix<DefDense, T, 4, 4> A_B_P_x_answer({
        {-6.66666667e-01F, 6.66666667e-01F, 0.0F, 0.0F},
        {6.33333333e-01F, -1.33333333F, 0.9F, 0.0F},
        {1.33333333e-01F, 6.66666667e-01F, -0.6F, 0.0F},
        {0.0F, 0.0F, 0.0F, 0.0F}
        });

    tester.expect_near(A_B_P_x.matrix.data, A_B_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Dense and Diag.");

    A_B_P_x = A_B_P_linalg_solver.cold_solve(A_P, B_P, 3);

    tester.expect_near(A_B_P_x.matrix.data, A_B_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Dense and Diag cold solve.");

    static auto A_C_P_linalg_solver = make_LinalgPartitionSolver<decltype(A_P), decltype(C_P)>();

    auto A_C_P_x = A_C_P_linalg_solver.solve(A_P, C_P, 3);

    Matrix<DefDense, T, 4, 4> A_C_P_x_answer({
        { 3.33333333e-01F, 0.0F, 2.66666667F, 0.0F},
        {-1.36666667F, 0.6F, -4.13333333F, 0.0F},
        {1.13333333F, -0.4F, 1.86666667F, 0.0F},
        {0.0F, 0.0F, 0.0F, 0.0F}
        });

    tester.expect_near(A_C_P_x.matrix.data, A_C_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Dense and Sparse.");

    A_C_P_x = A_C_P_linalg_solver.cold_solve(A_P, C_P, 3);

    tester.expect_near(A_C_P_x.matrix.data, A_C_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Dense and Sparse cold solve.");

    static auto B_A_P_linalg_solver = make_LinalgPartitionSolver<decltype(B_P), decltype(A_P)>();

    auto B_A_P_x = B_A_P_linalg_solver.solve(B_P, A_P, 3);

    Matrix<DefDense, T, 4, 4> B_A_P_x_answer({
        { 1.0F, 2.0F, 3.0F, 0.0F },
        { 2.5F, 2.0F, 3.0F, 0.0F },
        { 3.0F, 2.66666667F, 2.33333333F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F }
        });

    tester.expect_near(B_A_P_x.matrix.data, B_A_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Diag and Dense.");

    B_A_P_x = B_A_P_linalg_solver.cold_solve(B_P, A_P, 3);

    tester.expect_near(B_A_P_x.matrix.data, B_A_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Diag and Dense cold solve.");

    static auto B_B_P_linalg_solver = make_LinalgPartitionSolver<decltype(B_P), decltype(B_P)>();

    auto B_B_P_x = B_B_P_linalg_solver.solve(B_P, B_P, 3);
    auto B_B_P_x_dense = B_B_P_x.create_dense();

    Matrix<DefDense, T, 4, 4> B_B_P_x_answer({
        { 1.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F }
        });

    tester.expect_near(B_B_P_x_dense.matrix.data, B_B_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Diag and Diag.");

    static auto B_C_P_linalg_solver = make_LinalgPartitionSolver<decltype(B_P), decltype(C_P)>();

    auto B_C_P_x = B_C_P_linalg_solver.solve(B_P, C_P, 3);

    Matrix<DefDense, T, 4, 4> B_C_P_x_answer({
        { 1.0F, 0.0F, 0.0F, 0.0F },
        { 1.5F, 0.0F, 4.0F, 0.0F },
        { 0.0F, 0.66666667F, 1.33333333F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F }
        });

    tester.expect_near(B_C_P_x.matrix.data, B_C_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Diag and Sparse.");

    B_C_P_x = B_C_P_linalg_solver.cold_solve(B_P, C_P, 3);

    tester.expect_near(B_C_P_x.matrix.data, B_C_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Diag and Sparse cold solve.");

    static auto C_A_P_linalg_solver = make_LinalgPartitionSolver<decltype(C_P), decltype(A_P)>();

    auto C_A_P_x = C_A_P_linalg_solver.solve(C_P, A_P, 3);

    Matrix<DefDense, T, 4, 4> C_A_P_x_answer({
        { 1.0F, 2.0F, 3.0F, 0.0F },
        { 4.0F, 4.5F, 4.25F, 0.0F },
        {0.25F, -0.25F, -0.375F, 0.0F },
        {0.0F, 0.0F, 0.0F, 0.0F }
        });

    tester.expect_near(C_A_P_x.matrix.data, C_A_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Sparse and Dense.");

    C_A_P_x = C_A_P_linalg_solver.cold_solve(C_P, A_P, 3);

    tester.expect_near(C_A_P_x.matrix.data, C_A_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Sparse and Dense cold solve.");

    static auto C_B_P_linalg_solver = make_LinalgPartitionSolver<decltype(C_P), decltype(B_P)>();

    auto C_B_P_x = C_B_P_linalg_solver.solve(C_P, B_P, 3);

    Matrix<DefDense, T, 4, 4> C_B_P_x_answer({
        { 1.0F, 0.0F, 0.0F, 0.0F },
        { 0.75F, -0.5F, 1.5F, 0.0F },
        { -0.375F, 0.25F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F }
        });

    tester.expect_near(C_B_P_x.matrix.data, C_B_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Sparse and Diag.");

    C_B_P_x = C_B_P_linalg_solver.cold_solve(C_P, B_P, 3);

    tester.expect_near(C_B_P_x.matrix.data, C_B_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Sparse and Diag cold solve.");

    static auto C_C_P_linalg_solver = make_LinalgPartitionSolver<decltype(C_P), decltype(C_P)>();

    auto C_C_P_x = C_C_P_linalg_solver.solve(C_P, C_P, 3);

    Matrix<DefDense, T, 4, 4> C_C_P_x_answer({
        { 1.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F }
        });

    tester.expect_near(C_C_P_x.matrix.data, C_C_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Sparse and Sparse.");

    C_C_P_x = C_C_P_linalg_solver.cold_solve(C_P, C_P, 3);

    tester.expect_near(C_C_P_x.matrix.data, C_C_P_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgPartitionSolver solve Sparse and Sparse cold solve.");


    /* 矩形　左除算 */
    Matrix<DefDense, T, 4, 3> AL({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7}, {2, 2, 3} });
    Matrix<DefDiag, T, 4> BL({ 1, 2, 3, 4 });
    Matrix<DefSparse, T, 4, 3,
        SparseAvailable<
        ColumnAvailable<true, true, false>,
        ColumnAvailable<false, false, true>,
        ColumnAvailable<false, true, true>,
        ColumnAvailable<false, true, false>>
        >
        CL({ 1, 3, 2, 8, 4, 1 });

    static auto AL_AL_lstsq_solver = make_LinalgLstsqSolver<decltype(AL), decltype(AL)>();
    AL_AL_lstsq_solver.set_division_min(static_cast<T>(1.0e-10));
    AL_AL_lstsq_solver.set_decay_rate(static_cast<T>(0));

    auto AL_AL_x = AL_AL_lstsq_solver.solve(AL, AL);

    Matrix<DefDense, T, 3, 3> AL_AL_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(AL_AL_x.matrix.data, AL_AL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Dense and Dense.");

    static auto AL_BL_lstsq_solver = make_LinalgLstsqSolver<decltype(AL), decltype(BL)>();

    auto AL_BL_x = AL_BL_lstsq_solver.solve(AL, BL);

    Matrix<DefDense, T, 3, 4> AL_BL_x_answer({
        {-6.36363636e-01F, 7.27272727e-01F, 0.0F, -3.63636364e-01F },
        {6.36363636e-01F, -1.32727273e+00F, 0.9F, -3.63636364e-02F },
        {9.09090909e-02F, 5.81818182e-01F, -0.6F, 5.09090909e-01F }
        });

    tester.expect_near(AL_BL_x.matrix.data, AL_BL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Dense and Diag.");

    static auto AL_CL_lstsq_solver = make_LinalgLstsqSolver<decltype(AL), decltype(CL)>();

    auto AL_CL_x = AL_CL_lstsq_solver.solve(AL, CL);

    Matrix<DefDense, T, 3, 3> AL_CL_x_answer({
        {-0.63636364F, -2.0F, 0.72727273F },
        { 0.63636364F, 4.3F, -0.12727273F },
        { 0.09090909F, -1.2F, -0.21818182F }
        });

    tester.expect_near(AL_CL_x.matrix.data, AL_CL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Dense and Sparse.");

    static auto CL_AL_lstsq_solver = make_LinalgLstsqSolver<decltype(CL), decltype(AL)>();

    auto CL_AL_x = CL_AL_lstsq_solver.solve(CL, AL);

    Matrix<DefDense, T, 3, 3> CL_AL_x_answer({
        { 0.91304348F, 1.56521739F, 4.08695652F },
        { 0.02898551F, 0.14492754F, -0.36231884F },
        { 2.25362319F, 1.76811594F, 2.57971014F }
        });

    tester.expect_near(CL_AL_x.matrix.data, CL_AL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Sparse and Dense.");

    LinalgLstsqSolver_Type<decltype(CL), decltype(CL )> CL_CL_lstsq_solver
        = make_LinalgLstsqSolver<decltype(CL), decltype(CL)>();

    LinalgLstsqSolver_Type<decltype(CL), decltype(CL)> CL_CL_lstsq_solver_copy = CL_CL_lstsq_solver;
    LinalgLstsqSolver_Type<decltype(CL), decltype(CL)> CL_CL_lstsq_solver_move = std::move(CL_CL_lstsq_solver_copy);
    CL_CL_lstsq_solver = CL_CL_lstsq_solver_move;

    auto CL_CL_x = CL_CL_lstsq_solver.solve(CL, CL);

    Matrix<DefDense, T, 3, 3> CL_CL_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(CL_CL_x.matrix.data, CL_CL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Sparse and Sparse.");

    /* 逆行列 */
    LinalgSolverInv_Type<decltype(A)> A_inv_solver
        = make_LinalgSolverInv<decltype(A)>();

    auto A_Inv = A_inv_solver.inv(A);

    LinalgSolverInv_Type<decltype(A)> A_inv_solver_copy = A_inv_solver;
    LinalgSolverInv_Type<decltype(A)> A_inv_solver_move = std::move(A_inv_solver_copy);
    A_inv_solver = A_inv_solver_move;

    Matrix<DefDense, T, 3, 3> A_Inv_answer({
        {-6.66666667e-01F, 3.33333333e-01F, 0.0F },
        {6.33333333e-01F, -6.66666667e-01F, 3.00000000e-01F },
        {1.33333333e-01F, 3.33333333e-01F, -0.2F }
        });

    tester.expect_near(A_Inv.matrix.data, A_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Dense.");

    static auto B_inv_solver = make_LinalgSolverInv<decltype(B)>();

    LinalgSolverInv_Type<decltype(B)> B_inv_solver_copy = B_inv_solver;
    LinalgSolverInv_Type<decltype(B)> B_inv_solver_move = std::move(B_inv_solver_copy);
    B_inv_solver = B_inv_solver_move;

    auto B_Inv = B_inv_solver.inv(B);
    B_Inv = B_inv_solver.get_answer();
    auto B_Inv_dense = B_Inv.create_dense();

    Matrix<DefDense, T, 3, 3> B_Inv_answer({
        {1.0F, 0.0F, 0.0F},
        {0.0F, 0.5F, 0.0F},
        {0.0F, 0.0F, 0.33333333F}
        });

    tester.expect_near(B_Inv_dense.matrix.data, B_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Diag.");

    static auto C_inv_solver = make_LinalgSolverInv<decltype(C)>();

    auto C_Inv = C_inv_solver.inv(C);

    Matrix<DefDense, T, 3, 3> Inv_answer({
        {1.0F, 0.0F, 0.0F},
        {0.75F, -0.25F, 0.5F},
        {-0.375F, 0.125F, 0}
        });

    tester.expect_near(C_Inv.matrix.data, Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Sparse.");


    /* 逆行列 複素数 */
    Matrix<DefDense, Complex<T>, 3, 3> A_comp({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, Complex<T>, 3> B_comp({ 1, 2, 3 });
    Matrix<DefSparse, Complex<T>, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C_comp({ 1, 3, 8, 2, 4 });

    static auto A_comp_inv_solver = make_LinalgSolverInv<decltype(A_comp)>();

    auto A_comp_Inv = A_comp_inv_solver.inv(A_comp);
    auto A_comp_Inv_real = A_comp_Inv.real();

    Matrix<DefDense, T, 3, 3> A_comp_Inv_answer_real({
        {-6.66666667e-01F, 3.33333333e-01F, 0.0F },
        {6.33333333e-01F, -6.66666667e-01F, 3.00000000e-01F },
        {1.33333333e-01F, 3.33333333e-01F, -0.2F }
        });

    tester.expect_near(A_comp_Inv_real.matrix.data,
        A_comp_Inv_answer_real.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Dense complex real.");

    auto A_comp_Inv_imag = A_comp_Inv.imag();

    Matrix<DefDense, T, 3, 3> A_comp_Inv_answer_imag({
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F}
        });

    tester.expect_near(A_comp_Inv_imag.matrix.data,
        A_comp_Inv_answer_imag.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Dense complex imag.");

    static auto B_comp_inv_solver = make_LinalgSolverInv<decltype(B_comp)>();

    auto B_comp_Inv = B_comp_inv_solver.inv(B_comp);
    auto B_comp_Inv_real = B_comp_Inv.real();

    Matrix<DefDiag, T, 3> B_comp_Inv_answer_real(
        {1.0F, 0.5F, 0.33333333F}
        );

    tester.expect_near(B_comp_Inv_real.matrix.data,
        B_comp_Inv_answer_real.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Diag complex real.");

    auto B_comp_Inv_imag = B_comp_Inv.imag();

    Matrix<DefDiag, T, 3> B_comp_Inv_answer_imag(
        {0.0F, 0.0F, 0.0F}
        );

    tester.expect_near(B_comp_Inv_imag.matrix.data,
        B_comp_Inv_answer_imag.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Diag complex imag.");

    auto C_comp_real = C_comp.real();
    auto C_comp_real_dense = C_comp_real.create_dense();

    Matrix<DefDense, T, 3, 3> C_comp_real_answer({
        {1.0F, 0.0F, 0.0F},
        {3.0F, 0.0F, 8.0F},
        {0.0F, 2.0F, 4.0F}
        });

    tester.expect_near(C_comp_real_dense.matrix.data, C_comp_real_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix get real.");

    auto C_comp_imag = C_comp.imag();
    auto C_comp_imag_dense = C_comp_imag.create_dense();

    Matrix<DefDense, T, 3, 3> C_comp_imag_answer({
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F}
        });

    tester.expect_near(C_comp_imag_dense.matrix.data, C_comp_imag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix get imag.");

    static auto C_comp_inv_solver = make_LinalgSolverInv<decltype(C_comp)>();

    auto C_comp_Inv = C_comp_inv_solver.inv(C_comp);
    auto C_comp_Inv_real = C_comp_Inv.real();

    Matrix<DefDense, T, 3, 3> C_comp_Inv_answer_real({
        {1.0F, 0.0F, 0.0F},
        {0.75F, -0.25F, 0.5F},
        {-0.375F, 0.125F, 0.0F}
        });

    tester.expect_near(C_comp_Inv_real.matrix.data,
        C_comp_Inv_answer_real.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Sparse complex real.");

    auto C_comp_Inv_imag = C_comp_Inv.imag();

    Matrix<DefDense, T, 3, 3> C_comp_Inv_answer_imag({
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F}
        });

    tester.expect_near(C_comp_Inv_imag.matrix.data,
        C_comp_Inv_answer_imag.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Sparse complex imag.");


    /* スカラー左除算 */
    Matrix<DefDense, T, 1, 1> A_scalar({ { 2 } });
    Matrix<DefDiag, T, 1> B_scalar({ 3 });
    Matrix<DefSparse, T, 1, 1,
        SparseAvailable<
        ColumnAvailable<true>>
        > C_scalar({ 4 });

    auto A_scalar_linalg_solver = make_LinalgSolver<decltype(A_scalar), decltype(A_scalar)>();
    auto A_scalar_x = A_scalar_linalg_solver.solve(A_scalar, A_scalar);

    Matrix<DefDense, T, 1, 1> A_scalar_x_answer({
        { 1.0F }
        });

    tester.expect_near(A_scalar_x.matrix.data, A_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Dense scalar.");


    auto A_scalar_B_scalar_linalg_solver = make_LinalgSolver<decltype(A_scalar), decltype(B_scalar)>();
    auto A_scalar_B_scalar_x = A_scalar_B_scalar_linalg_solver.solve(A_scalar, B_scalar);

    Matrix<DefDense, T, 1, 1> A_scalar_B_scalar_x_answer({
        { 1.5F }
        });

    tester.expect_near(A_scalar_B_scalar_x.matrix.data, A_scalar_B_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Diag scalar.");

    auto A_scalar_C_scalar_linalg_solver = make_LinalgSolver<decltype(A_scalar), decltype(C_scalar)>();
    auto A_scalar_C_scalar_x = A_scalar_C_scalar_linalg_solver.solve(A_scalar, C_scalar);

    Matrix<DefDense, T, 1, 1> A_scalar_C_scalar_x_answer({
        { 2.0F }
        });

    tester.expect_near(A_scalar_C_scalar_x.matrix.data, A_scalar_C_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Sparse scalar.");

    auto B_scalar_A_scalar_linalg_solver = make_LinalgSolver<decltype(B_scalar), decltype(A_scalar)>();
    auto B_scalar_A_scalar_x = B_scalar_A_scalar_linalg_solver.solve(B_scalar, A_scalar);

    Matrix<DefDense, T, 1, 1> B_scalar_A_scalar_x_answer({
        { 0.666666667F }
        });

    tester.expect_near(B_scalar_A_scalar_x.matrix.data, B_scalar_A_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Dense scalar.");

    auto B_scalar_B_scalar_linalg_solver = make_LinalgSolver<decltype(B_scalar), decltype(B_scalar)>();
    auto B_scalar_B_scalar_x = B_scalar_B_scalar_linalg_solver.solve(B_scalar, B_scalar);
    auto B_scalar_B_scalar_x_dense = B_scalar_B_scalar_x.create_dense();

    Matrix<DefDense, T, 1, 1> B_scalar_B_scalar_x_answer({
        { 1.0F }
        });

    tester.expect_near(B_scalar_B_scalar_x_dense.matrix.data, B_scalar_B_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Diag scalar.");

    auto B_scalar_C_scalar_linalg_solver = make_LinalgSolver<decltype(B_scalar), decltype(C_scalar)>();
    auto B_scalar_C_scalar_x = B_scalar_C_scalar_linalg_solver.solve(B_scalar, C_scalar);

    Matrix<DefDense, T, 1, 1> B_scalar_C_scalar_x_answer({
        { 1.33333333F }
        });

    tester.expect_near(B_scalar_C_scalar_x.matrix.data, B_scalar_C_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Sparse scalar.");

    auto C_scalar_A_scalar_linalg_solver = make_LinalgSolver<decltype(C_scalar), decltype(A_scalar)>();
    auto C_scalar_A_scalar_x = C_scalar_A_scalar_linalg_solver.solve(C_scalar, A_scalar);

    Matrix<DefDense, T, 1, 1> C_scalar_A_scalar_x_answer({
        { 0.5F }
        });

    tester.expect_near(C_scalar_A_scalar_x.matrix.data, C_scalar_A_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Sparse and Dense scalar.");

    auto C_scalar_B_scalar_linalg_solver = make_LinalgSolver<decltype(C_scalar), decltype(B_scalar)>();
    auto C_scalar_B_scalar_x = C_scalar_B_scalar_linalg_solver.solve(C_scalar, B_scalar);

    Matrix<DefDense, T, 1, 1> C_scalar_B_scalar_x_answer({
        { 0.75F }
        });

    tester.expect_near(C_scalar_B_scalar_x.matrix.data, C_scalar_B_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Sparse and Diag scalar.");

    auto C_scalar_C_scalar_linalg_solver = make_LinalgSolver<decltype(C_scalar), decltype(C_scalar)>();
    auto C_scalar_C_scalar_x = C_scalar_C_scalar_linalg_solver.solve(C_scalar, C_scalar);

    Matrix<DefDense, T, 1, 1> C_scalar_C_scalar_x_answer({
        { 1.0F }
        });

    tester.expect_near(C_scalar_C_scalar_x.matrix.data, C_scalar_C_scalar_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Sparse and Sparse scalar.");

    /* 逆行列 スカラー */
    auto A_scalar_inv_solver = make_LinalgSolverInv<decltype(A_scalar)>();
    auto A_scalar_Inv = A_scalar_inv_solver.inv(A_scalar);

    Matrix<DefDense, T, 1, 1> A_scalar_Inv_answer({
        { 0.5F }
        });

    tester.expect_near(A_scalar_Inv.matrix.data, A_scalar_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Dense scalar.");

    auto B_scalar_inv_solver = make_LinalgSolverInv<decltype(B_scalar)>();
    auto B_scalar_Inv = B_scalar_inv_solver.inv(B_scalar);
    auto B_scalar_Inv_dense = B_scalar_Inv.create_dense();

    Matrix<DefDense, T, 1, 1> B_scalar_Inv_answer({
        { 0.333333333F }
        });

    tester.expect_near(B_scalar_Inv_dense.matrix.data, B_scalar_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Diag scalar.");

    auto C_scalar_inv_solver = make_LinalgSolverInv<decltype(C_scalar)>();
    auto C_scalar_Inv = C_scalar_inv_solver.inv(C_scalar);

    Matrix<DefDense, T, 1, 1> C_scalar_Inv_answer({
        { 0.25F }
        });

    tester.expect_near(C_scalar_Inv.matrix.data, C_scalar_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Sparse scalar.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckPythonNumpy<T>::check_python_numpy_concatenate(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });

    Matrix<DefDense, T, 4, 3> AL({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7}, {2, 2, 3} });
    Matrix<DefDiag, T, 4> BL({ 1, 2, 3, 4 });
    Matrix<DefSparse, T, 4, 3,
        SparseAvailable<
        ColumnAvailable<true, true, false>,
        ColumnAvailable<false, false, true>,
        ColumnAvailable<false, true, true>,
        ColumnAvailable<false, true, false>>
        >
        CL({ 1, 3, 2, 8, 4, 1 });

    Matrix<DefSparse, T, 3, 3,
        SparseAvailableEmpty<3, 3>> Empty;
    Matrix<DefSparse, T, 4, 3,
        SparseAvailableEmpty<4, 3>> EmptyL;


    /* 結合 */
    Matrix<DefDense, T, 1, 1> a({ {1} });
    Matrix<DefDense, T, 1, 1> b({ {2} });
    Matrix<DefDense, T, 2, 1> aa({ {1}, {2} });
    Matrix<DefDense, T, 2, 1> bb({ {3}, {4} });

    auto a_v_b = concatenate_vertically(a, b);

    Matrix<DefDense, T, 2, 1> a_v_b_answer({
        {1},
        {2}
        });

    tester.expect_near(a_v_b.matrix.data, a_v_b_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Dense small, 1 1.");

    auto aa_v_b = concatenate_vertically(aa, b);

    Matrix<DefDense, T, 3, 1> aa_v_b_answer({
        {1},
        {2},
        {2}
        });

    tester.expect_near(aa_v_b.matrix.data, aa_v_b_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Dense small, 2 1.");

    auto a_v_bb = concatenate_vertically(a, bb);

    Matrix<DefDense, T, 3, 1> a_v_bb_answer({
        {1},
        {3},
        {4}
        });

    tester.expect_near(a_v_bb.matrix.data, a_v_bb_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Dense small, 1 2.");


    auto A_v_A = concatenate_vertically(A, A);
    ConcatenateVertically_Type<decltype(A), decltype(A)> A_v_A_t;
    A_v_A_t = A_v_A;

    Matrix<DefDense, T, 6, 3> A_v_A_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 }
        });

    tester.expect_near(A_v_A_t.matrix.data, A_v_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Dense.");

    update_vertically_concatenated_matrix(A_v_A, static_cast<T>(2) * A, A);

    Matrix<DefDense, T, 6, 3> A_v_A_answer_2({
        { 2, 4, 6 },
        { 10, 8, 12 },
        { 18, 16, 14 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 }
        });

    tester.expect_near(A_v_A.matrix.data, A_v_A_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Dense and Dense.");

    auto A_v_B = concatenate_vertically(A, B);
    ConcatenateVertically_Type<decltype(A), decltype(B)> A_v_B_t;
    A_v_B_t = A_v_B;
    auto A_v_B_t_dense = A_v_B_t.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_B_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 }
        });

    tester.expect_near(A_v_B_t_dense.matrix.data, A_v_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Diag.");

    update_vertically_concatenated_matrix(A_v_B, A, static_cast<T>(2) * B);
    auto A_v_B_dense = A_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_B_answer_2({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 2, 0, 0 },
        { 0, 4, 0 },
        { 0, 0, 6 }
        });

    tester.expect_near(A_v_B_dense.matrix.data, A_v_B_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Dense and Diag.");


    auto A_v_C = concatenate_vertically(A, C);
    ConcatenateVertically_Type<decltype(A), decltype(C)> A_v_C_t;
    A_v_C_t = A_v_C;
    auto A_v_C_t_dense = A_v_C_t.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_C_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 }
        });

    tester.expect_near(A_v_C_t_dense.matrix.data, A_v_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Sparse.");

    update_vertically_concatenated_matrix(A_v_C, A, static_cast<T>(2) * C);
    auto A_v_C_dense = A_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_C_answer_2({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 2, 0, 0 },
        { 6, 0, 16 },
        { 0, 4, 8 }
        });

    tester.expect_near(A_v_C_dense.matrix.data, A_v_C_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Dense and Sparse.");

    auto A_v_E = concatenate_vertically(A, Empty);
    ConcatenateVertically_Type<decltype(A), decltype(Empty)> A_v_E_t;
    A_v_E_t = A_v_E;
    auto A_v_E_t_dense = A_v_E_t.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_E_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 }
        });

    tester.expect_near(A_v_E_t_dense.matrix.data, A_v_E_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Empty.");

    update_vertically_concatenated_matrix(A_v_E, static_cast<T>(2) * A, Empty);
    auto A_v_E_dense = A_v_E.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_E_answer_2({
        { 2, 4, 6 },
        { 10, 8, 12 },
        { 18, 16, 14 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 }
        });

    tester.expect_near(A_v_E_dense.matrix.data, A_v_E_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Dense and Empty.");

    auto B_v_A = concatenate_vertically(B, A);
    ConcatenateVertically_Type<decltype(B), decltype(A)> B_v_A_t;
    B_v_A_t = B_v_A;
    auto B_v_A_t_dense = B_v_A_t.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_A_answer({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 }
        });

    tester.expect_near(B_v_A_t_dense.matrix.data, B_v_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Dense.");

    update_vertically_concatenated_matrix(B_v_A, static_cast<T>(2) * B, A);
    auto B_v_A_dense = B_v_A.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_A_answer_2({
        { 2, 0, 0 },
        { 0, 4, 0 },
        { 0, 0, 6 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 }
        });

    tester.expect_near(B_v_A_dense.matrix.data, B_v_A_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Diag and Dense.");

    auto B_v_B = concatenate_vertically(B, B * static_cast<T>(2));
    ConcatenateVertically_Type<decltype(B), decltype(B)> B_v_B_t;
    B_v_B_t = B_v_B;
    auto B_v_B_t_dense = B_v_B_t.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_B_answer({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 2, 0, 0 },
        { 0, 4, 0 },
        { 0, 0, 6 }
        });

    tester.expect_near(B_v_B_t_dense.matrix.data, B_v_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Diag.");

    update_vertically_concatenated_matrix(B_v_B, static_cast<T>(2) * B, B);
    auto B_v_B_dense = B_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_B_answer_2({
        { 2, 0, 0 },
        { 0, 4, 0 },
        { 0, 0, 6 },
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 }
        });

    tester.expect_near(B_v_B_dense.matrix.data, B_v_B_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Diag and Diag.");

    auto B_v_C = concatenate_vertically(B, C);
    ConcatenateVertically_Type<decltype(B), decltype(C)> B_v_C_t;
    B_v_C_t = B_v_C;
    auto B_v_C_t_dense = B_v_C_t.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_C_answer({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 }
        });

    tester.expect_near(B_v_C_t_dense.matrix.data, B_v_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Sparse.");

    update_vertically_concatenated_matrix(B_v_C, B, static_cast<T>(2) * C);
    auto B_v_C_dense = B_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_C_answer_2({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 2, 0, 0 },
        { 6, 0, 16 },
        { 0, 4, 8 }
        });

    tester.expect_near(B_v_C_dense.matrix.data, B_v_C_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Diag and Sparse.");

    auto B_v_E = concatenate_vertically(B, Empty);
    auto B_v_E_dense = B_v_E.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_E_answer({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 }
        });

    tester.expect_near(B_v_E_dense.matrix.data, B_v_E_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Empty.");

    auto C_v_A = concatenate_vertically(C, A);
    ConcatenateVertically_Type<decltype(C), decltype(A)> C_v_A_t;
    C_v_A_t = C_v_A;
    auto C_v_A_t_dense = C_v_A.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_A_answer({
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        });

    tester.expect_near(C_v_A_t_dense.matrix.data, C_v_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Dense.");

    update_vertically_concatenated_matrix(C_v_A, C, static_cast<T>(2) * A);
    auto C_v_A_dense = C_v_A.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_A_answer_2({
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 },
        { 2, 4, 6 },
        { 10, 8, 12 },
        { 18, 16, 14 }
        });

    tester.expect_near(C_v_A_dense.matrix.data, C_v_A_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Sparse and Dense.");

    auto E_v_A = concatenate_vertically(Empty, A);
    auto E_v_A_dense = E_v_A.create_dense();

    Matrix<DefDense, T, 6, 3> E_v_A_answer({
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 }
        });

    tester.expect_near(E_v_A_dense.matrix.data, E_v_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Empty and Dense.");

    auto C_v_B = concatenate_vertically(C, B);
    ConcatenateVertically_Type<decltype(C), decltype(B)> C_v_B_t;
    C_v_B_t = C_v_B;
    auto C_v_B_t_dense = C_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_B_answer({
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 },
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 }
        });

    tester.expect_near(C_v_B_t_dense.matrix.data, C_v_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Diag.");

    update_vertically_concatenated_matrix(C_v_B, C, static_cast<T>(2) * B);
    auto C_v_B_dense = C_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_B_answer_2({
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 },
        { 2, 0, 0 },
        { 0, 4, 0 },
        { 0, 0, 6 }
        });

    tester.expect_near(C_v_B_dense.matrix.data, C_v_B_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Sparse and Diag.");

    auto E_v_B = concatenate_vertically(Empty, B);
    auto E_v_B_dense = E_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> E_v_B_answer({
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 }
        });

    tester.expect_near(E_v_B_dense.matrix.data, E_v_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Empty and Diag.");

    auto C_v_C = concatenate_vertically(C, C * static_cast<T>(2));
    ConcatenateVertically_Type<decltype(C), decltype(C)> C_v_C_t;
    C_v_C_t = C_v_C;
    auto C_v_C_t_dense = C_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_C_answer({
    { 1, 0, 0 },
    { 3, 0, 8 },
    { 0, 2, 4 },
    { 2, 0, 0 },
    { 6, 0, 16 },
    { 0, 4, 8 }
        });

    tester.expect_near(C_v_C_t_dense.matrix.data, C_v_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Sparse.");

    update_vertically_concatenated_matrix(C_v_C, static_cast<T>(2) * C, C);
    auto C_v_C_dense = C_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_C_answer_2({
        { 2, 0, 0 },
        { 6, 0, 16 },
        { 0, 4, 8 },
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 }
        });

    tester.expect_near(C_v_C_dense.matrix.data, C_v_C_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update vertically concatenated matrix Sparse and Sparse.");

    auto E_v_C = concatenate_vertically(Empty, C);
    auto E_v_C_dense = E_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> E_v_C_answer({
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 }
        });

    tester.expect_near(E_v_C_dense.matrix.data, E_v_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Empty and Sparse.");

    auto AL_h_AL = concatenate_horizontally(AL, AL);
    ConcatenateHorizontally_Type<decltype(AL), decltype(AL)> AL_h_AL_t;
    AL_h_AL_t = AL_h_AL;

    Matrix<DefDense, T, 4, 6> AL_h_AL_answer({
        {1, 2, 3, 1, 2, 3},
        {5, 4, 6, 5, 4, 6},
        {9, 8, 7, 9, 8, 7},
        {2, 2, 3, 2, 2, 3}
        });

    tester.expect_near(AL_h_AL_t.matrix.data, AL_h_AL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Dense.");

    update_horizontally_concatenated_matrix(AL_h_AL, static_cast<T>(2) * AL, AL);

    Matrix<DefDense, T, 4, 6> AL_h_AL_answer_2({
        {2, 4, 6, 1, 2, 3},
        {10, 8, 12, 5, 4, 6},
        {18, 16, 14, 9, 8, 7},
        {4, 4, 6, 2, 2, 3}
        });

    tester.expect_near(AL_h_AL.matrix.data, AL_h_AL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Dense and Dense.");

    auto AL_h_BL = concatenate_horizontally(AL, BL);
    ConcatenateHorizontally_Type<decltype(AL), decltype(BL)> AL_h_BL_t;
    AL_h_BL_t = AL_h_BL;
    auto AL_h_BL_t_dense = AL_h_BL_t.create_dense();

    Matrix<DefDense, T, 4, 7> AL_h_BL_answer({
        {1, 2, 3, 1, 0, 0, 0},
        {5, 4, 6, 0, 2, 0, 0},
        {9, 8, 7, 0, 0, 3, 0},
        {2, 2, 3, 0, 0, 0, 4}
        });

    tester.expect_near(AL_h_BL_t_dense.matrix.data, AL_h_BL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Diag.");

    update_horizontally_concatenated_matrix(AL_h_BL, AL, static_cast<T>(2) * BL);
    auto AL_h_BL_dense = AL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 7> AL_h_BL_answer_2({
        {1, 2, 3, 2, 0, 0, 0},
        {5, 4, 6, 0, 4, 0, 0},
        {9, 8, 7, 0, 0, 6, 0},
        {2, 2, 3, 0, 0, 0, 8}
        });

    tester.expect_near(AL_h_BL_dense.matrix.data, AL_h_BL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Dense and Diag.");

    auto AL_h_CL = concatenate_horizontally(AL, CL);
    ConcatenateHorizontally_Type<decltype(AL), decltype(CL)> AL_h_CL_t;
    AL_h_CL_t = AL_h_CL;
    auto AL_h_CL_t_dense = AL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 6> AL_h_CL_answer({
        {1, 2, 3, 1, 3, 0},
        {5, 4, 6, 0, 0, 2},
        {9, 8, 7, 0, 8, 4},
        {2, 2, 3, 0, 1, 0}
        });

    tester.expect_near(AL_h_CL_t_dense.matrix.data, AL_h_CL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Sparse.");

    update_horizontally_concatenated_matrix(AL_h_CL, AL, static_cast<T>(2) * CL);
    auto AL_h_CL_dense = AL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 6> AL_h_CL_answer_2({
        {1, 2, 3, 2, 6, 0},
        {5, 4, 6, 0, 0, 4},
        {9, 8, 7, 0, 16, 8},
        {2, 2, 3, 0, 2, 0}
        });

    tester.expect_near(AL_h_CL_dense.matrix.data, AL_h_CL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Dense and Sparse.");

    auto AL_h_EL = concatenate_horizontally(AL, EmptyL);
    auto AL_h_EL_dense = AL_h_EL.create_dense();

    Matrix<DefDense, T, 4, 6> AL_h_EL_answer({
        {1, 2, 3, 0, 0, 0},
        {5, 4, 6, 0, 0, 0},
        {9, 8, 7, 0, 0, 0},
        {2, 2, 3, 0, 0, 0}
        });

    tester.expect_near(AL_h_EL_dense.matrix.data, AL_h_EL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Empty.");

    auto BL_h_AL = concatenate_horizontally(BL, AL);
    ConcatenateHorizontally_Type<decltype(BL), decltype(AL)> BL_h_AL_t;
    BL_h_AL_t = BL_h_AL;
    auto BL_h_AL_t_dense = BL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_AL_answer({
        {1, 0, 0, 0, 1, 2, 3},
        {0, 2, 0, 0, 5, 4, 6},
        {0, 0, 3, 0, 9, 8, 7},
        {0, 0, 0, 4, 2, 2, 3}
        });

    tester.expect_near(BL_h_AL_t_dense.matrix.data, BL_h_AL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Dense.");

    update_horizontally_concatenated_matrix(BL_h_AL, static_cast<T>(2) * BL, AL);
    auto BL_h_AL_dense = BL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_AL_answer_2({
        {2, 0, 0, 0, 1, 2, 3},
        {0, 4, 0, 0, 5, 4, 6},
        {0, 0, 6, 0, 9, 8, 7},
        {0, 0, 0, 8, 2, 2, 3}
        });

    tester.expect_near(BL_h_AL_dense.matrix.data, BL_h_AL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Diag and Dense.");

    auto BL_h_BL = concatenate_horizontally(BL, BL * static_cast<T>(2));
    ConcatenateHorizontally_Type<decltype(BL), decltype(BL)> BL_h_BL_t;
    BL_h_BL_t = BL_h_BL;
    auto BL_h_BL_t_dense = BL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 8> BL_h_BL_answer({
        {1, 0, 0, 0, 2, 0, 0, 0},
        {0, 2, 0, 0, 0, 4, 0, 0},
        {0, 0, 3, 0, 0, 0, 6, 0},
        {0, 0, 0, 4, 0, 0, 0, 8}
        });

    tester.expect_near(BL_h_BL_t_dense.matrix.data, BL_h_BL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Diag.");

    update_horizontally_concatenated_matrix(BL_h_BL, static_cast<T>(2) * BL, BL);
    auto BL_h_BL_dense = BL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 8> BL_h_BL_answer_2({
        {2, 0, 0, 0, 1, 0, 0, 0},
        {0, 4, 0, 0, 0, 2, 0, 0},
        {0, 0, 6, 0, 0, 0, 3, 0},
        {0, 0, 0, 8, 0, 0, 0, 4}
        });

    tester.expect_near(BL_h_BL_dense.matrix.data, BL_h_BL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Diag and Diag.");

    auto BL_h_CL = concatenate_horizontally(BL, CL);
    ConcatenateHorizontally_Type<decltype(BL), decltype(CL)> BL_h_CL_t;
    BL_h_CL_t = BL_h_CL;
    auto BL_h_CL_t_dense = BL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_CL_answer({
        {1, 0, 0, 0, 1, 3, 0},
        {0, 2, 0, 0, 0, 0, 2},
        {0, 0, 3, 0, 0, 8, 4},
        {0, 0, 0, 4, 0, 1, 0}
        });

    tester.expect_near(BL_h_CL_t_dense.matrix.data, BL_h_CL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Sparse.");

    update_horizontally_concatenated_matrix(BL_h_CL, static_cast<T>(2) * BL, CL);
    auto BL_h_CL_dense = BL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_CL_answer_2({
        {2, 0, 0, 0, 1, 3, 0},
        {0, 4, 0, 0, 0, 0, 2},
        {0, 0, 6, 0, 0, 8, 4},
        {0, 0, 0, 8, 0, 1, 0}
        });

    tester.expect_near(BL_h_CL_dense.matrix.data, BL_h_CL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Diag and Sparse.");

    auto BL_h_EL = concatenate_horizontally(BL, EmptyL);
    auto BL_h_EL_dense = BL_h_EL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_EL_answer({
        {1, 0, 0, 0, 0, 0, 0},
        {0, 2, 0, 0, 0, 0, 0},
        {0, 0, 3, 0, 0, 0, 0},
        {0, 0, 0, 4, 0, 0, 0}
        });

    tester.expect_near(BL_h_EL_dense.matrix.data, BL_h_EL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Empty.");

    auto CL_h_AL = concatenate_horizontally(CL, AL);
    ConcatenateHorizontally_Type<decltype(CL), decltype(AL)> CL_h_AL_t;
    CL_h_AL_t = CL_h_AL;
    auto CL_h_AL_t_dense = CL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 6> CL_h_AL_answer({
        {1, 3, 0, 1, 2, 3},
        {0, 0, 2, 5, 4, 6},
        {0, 8, 4, 9, 8, 7},
        {0, 1, 0, 2, 2, 3}
        });

    tester.expect_near(CL_h_AL_t_dense.matrix.data, CL_h_AL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Dense.");

    update_horizontally_concatenated_matrix(CL_h_AL, CL, static_cast<T>(2) * AL);
    auto CL_h_AL_dense = CL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 6> CL_h_AL_answer_2({
        {1, 3, 0, 2, 4, 6},
        {0, 0, 2, 10, 8, 12},
        {0, 8, 4, 18, 16, 14},
        {0, 1, 0, 4, 4, 6}
        });

    tester.expect_near(CL_h_AL_dense.matrix.data, CL_h_AL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Sparse and Dense.");

    auto EL_h_AL = concatenate_horizontally(EmptyL, AL);
    auto EL_h_AL_dense = EL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 6> EL_h_AL_answer({
        {0, 0, 0, 1, 2, 3},
        {0, 0, 0, 5, 4, 6},
        {0, 0, 0, 9, 8, 7},
        {0, 0, 0, 2, 2, 3}
        });

    tester.expect_near(EL_h_AL_dense.matrix.data, EL_h_AL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Empty and Dense.");

    auto CL_h_BL = concatenate_horizontally(CL, BL);
    ConcatenateHorizontally_Type<decltype(CL), decltype(BL)> CL_h_BL_t;
    CL_h_BL_t = CL_h_BL;
    auto CL_h_BL_t_dense = CL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 7> CL_h_BL_answer({
        {1, 3, 0, 1, 0, 0, 0},
        {0, 0, 2, 0, 2, 0, 0},
        {0, 8, 4, 0, 0, 3, 0},
        {0, 1, 0, 0, 0, 0, 4}
        });

    tester.expect_near(CL_h_BL_t_dense.matrix.data, CL_h_BL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Diag.");

    update_horizontally_concatenated_matrix(CL_h_BL, CL, static_cast<T>(2) * BL);
    auto CL_h_BL_dense = CL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 7> CL_h_BL_answer_2({
        {1, 3, 0, 2, 0, 0, 0},
        {0, 0, 2, 0, 4, 0, 0},
        {0, 8, 4, 0, 0, 6, 0},
        {0, 1, 0, 0, 0, 0, 8}
        });

    tester.expect_near(CL_h_BL_dense.matrix.data, CL_h_BL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Sparse and Diag.");

    auto EL_h_BL = concatenate_horizontally(EmptyL, BL);
    auto EL_h_BL_dense = EL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 7> EL_h_BL_answer({
        {0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 2, 0, 0},
        {0, 0, 0, 0, 0, 3, 0},
        {0, 0, 0, 0, 0, 0, 4}
        });

    tester.expect_near(EL_h_BL_dense.matrix.data, EL_h_BL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Empty and Diag.");

    auto CL_h_CL = concatenate_horizontally(CL, CL);
    ConcatenateHorizontally_Type<decltype(CL), decltype(CL)> CL_h_CL_t;
    CL_h_CL_t = CL_h_CL;
    auto CL_h_CL_t_dense = CL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 6> CL_h_CL_answer({
        {1, 3, 0, 1, 3, 0},
        {0, 0, 2, 0, 0, 2},
        {0, 8, 4, 0, 8, 4},
        {0, 1, 0, 0, 1, 0}
        });

    tester.expect_near(CL_h_CL_t_dense.matrix.data, CL_h_CL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Sparse.");

    update_horizontally_concatenated_matrix(CL_h_CL, static_cast<T>(2) * CL, CL);
    auto CL_h_CL_dense = CL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 6> CL_h_CL_answer_2({
        {2, 6, 0, 1, 3, 0},
        {0, 0, 4, 0, 0, 2},
        {0, 16, 8, 0, 8, 4},
        {0, 2, 0, 0, 1, 0}
        });

    tester.expect_near(CL_h_CL_dense.matrix.data, CL_h_CL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Sparse and Sparse.");

    auto EL_h_CL = concatenate_horizontally(EmptyL, CL);
    auto EL_h_CL_dense = EL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 6> EL_h_CL_answer({
        {0, 0, 0, 1, 3, 0},
        {0, 0, 0, 0, 0, 2},
        {0, 0, 0, 0, 8, 4},
        {0, 0, 0, 0, 1, 0}
        });

    tester.expect_near(EL_h_CL_dense.matrix.data, EL_h_CL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Empty and Sparse.");

    /* Block結合 */
    auto ABCE_block = concatenate_block_2x2(A, B, C, Empty);
    ConcatenateBlock2X2_Type<decltype(A), decltype(B), decltype(C), decltype(Empty)> ABCE_block_t;
    ABCE_block_t = ABCE_block;
    auto ABCE_block_t_dense = ABCE_block_t.create_dense();

    Matrix<DefDense, T, 6, 6> ABCE_block_answer({
        {1, 2, 3, 1, 0, 0},
        {5, 4, 6, 0, 2, 0},
        {9, 8, 7, 0, 0, 3},
        {1, 0, 0, 0, 0, 0},
        {3, 0, 8, 0, 0, 0},
        {0, 2, 4, 0, 0, 0}
        });

    tester.expect_near(ABCE_block_t_dense.matrix.data, ABCE_block_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate block Dense, Diag, Sparse and Empty, 2 x 2.");

    update_block_2x2_concatenated_matrix(ABCE_block, A, static_cast<T>(2) * B, C, Empty);
    auto ABCE_block_dense = ABCE_block.create_dense();

    Matrix<DefDense, T, 6, 6> ABCE_block_answer_2({
        {1, 2, 3, 2, 0, 0},
        {5, 4, 6, 0, 4, 0},
        {9, 8, 7, 0, 0, 6},
        {1, 0, 0, 0, 0, 0},
        {3, 0, 8, 0, 0, 0},
        {0, 2, 4, 0, 0, 0}
        });

    tester.expect_near(ABCE_block_dense.matrix.data, ABCE_block_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update block concatenated matrix Dense, Diag, Sparse and Empty.");


    auto ABCE_block_2 = concatenate_block<2, 2>(A, B, C, Empty);
    ConcatenateBlock_Type<2, 2, decltype(A), decltype(B), decltype(C), decltype(Empty)> ABCE_block_t_2;
    ABCE_block_t_2 = ABCE_block_2;
    auto ABCE_block_t_dense_2 = ABCE_block_t_2.create_dense();

    tester.expect_near(ABCE_block_t_dense_2.matrix.data, ABCE_block_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate block Dense, Diag, Sparse and Empty.");

    update_block_concatenated_matrix<2, 2>(ABCE_block_2, A, static_cast<T>(2)* B, C, Empty);

    auto ABCE_block_dense_2 = ABCE_block_2.create_dense();

    Matrix<DefDense, T, 6, 6> ABCE_block_answer_2_2({
        {1, 2, 3, 2, 0, 0},
        {5, 4, 6, 0, 4, 0},
        {9, 8, 7, 0, 0, 6},
        {1, 0, 0, 0, 0, 0},
        {3, 0, 8, 0, 0, 0},
        {0, 2, 4, 0, 0, 0}
        });

    tester.expect_near(ABCE_block_dense_2.matrix.data, ABCE_block_answer_2_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update block concatenated matrix Dense, Diag, Sparse and Empty.");


    /* tile */
    Tile_Type<2, 3, decltype(C)> R = concatenate_tile<2, 3>(C);
    auto R_dense = R.create_dense();

    Matrix<DefDense, T, 6, 9> R_answer({
        {1, 0, 0, 1, 0, 0, 1, 0, 0},
        {3, 0, 8, 3, 0, 8, 3, 0, 8},
        {0, 2, 4, 0, 2, 4, 0, 2, 4},
        {1, 0, 0, 1, 0, 0, 1, 0, 0},
        {3, 0, 8, 3, 0, 8, 3, 0, 8},
        {0, 2, 4, 0, 2, 4, 0, 2, 4}
        });

    tester.expect_near(R_dense.matrix.data, R_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check repmat Sparse.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckPythonNumpy<T>::check_python_numpy_transpose(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });

    Matrix<DefDense, T, 4, 3> AL({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7}, {2, 2, 3} });
    Matrix<DefDiag, T, 4> BL({ 1, 2, 3, 4 });
    Matrix<DefSparse, T, 4, 3,
        SparseAvailable<
        ColumnAvailable<true, true, false>,
        ColumnAvailable<false, false, true>,
        ColumnAvailable<false, true, true>,
        ColumnAvailable<false, true, false>>
        >
        CL({ 1, 3, 2, 8, 4, 1 });

    /* 転置 */
    Transpose_Type<decltype(A)> AT = A.transpose();

    Matrix<DefDense, T, 3, 3> AT_answer({
        {1, 5, 9},
        {2, 4, 8},
        {3, 6, 7}
        });

    tester.expect_near(AT.matrix.data, AT_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check transpose Dense.");

    Transpose_Type<decltype(CL)> CL_T = CL.transpose();
    auto CL_T_dense = CL_T.create_dense();

    Matrix<DefDense, T, 3, 4> CL_T_answer({
        {1, 0, 0, 0},
        {3, 0, 8, 1},
        {0, 2, 4, 0}
        });

    tester.expect_near(CL_T_dense.matrix.data, CL_T_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check transpose Sparse.");

    auto B_T = B.transpose();
    //std::cout << "B_T = ";
    //for (size_t i = 0; i < B_T.rows(); ++i) {
    //    std::cout << B_T.matrix[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Matrix<DefDiag, T, 3> B_T_answer({ 1, 2, 3 });
    tester.expect_near(B_T.matrix.data, B_T_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check transpose Diag.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckPythonNumpy<T>::check_python_numpy_lu(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });

    /* LU分解 */
    LinalgSolverLU_Type<decltype(A)> A_LU_solver = make_LinalgSolverLU<decltype(A)>();
    A_LU_solver.set_division_min(static_cast<T>(1.0e-10));
    A_LU_solver.solve(A);

    auto A_LU = A_LU_solver.get_L() * A_LU_solver.get_U();
    auto A_LU_dense = A_LU.create_dense();

    Matrix<DefDense, T, 3, 3> A_LU_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    tester.expect_near(A_LU_dense.matrix.data, A_LU_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverLU L multiply U Dense.");


    T det_answer = 30;
    tester.expect_near(A_LU_solver.get_det(), det_answer, NEAR_LIMIT_STRICT,
        "check LinalgSolverLU det.");

    auto B_LU_solver = make_LinalgSolverLU<decltype(B)>();
    B_LU_solver.solve(B);

    auto B_LU = B_LU_solver.get_L() * B_LU_solver.get_U();
    auto B_LU_dense = B_LU.create_dense();

    Matrix<DefDense, T, 3, 3> B_LU_answer({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
        });

    tester.expect_near(B_LU_dense.matrix.data, B_LU_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverLU L multiply U Diag.");




    tester.throw_error_if_test_failed();
}

template<typename T>
void CheckPythonNumpy<T>::check_python_numpy_cholesky(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefDense, T, 3, 3> K({
        {10, 1, 2},
        {1, 20, 4},
        {2, 4, 30}
        });
    Matrix<DefSparse, T, 3, 3, SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<false, true, true>,
        ColumnAvailable<false, true, true>>
        >K_s({ 1, 8, 3, 3, 4 });

    /* コレスキー分解 */
    LinalgSolverCholesky_Type<decltype(K)> Chol_solver;
    Chol_solver.set_division_min(static_cast<T>(1.0e-10));
    Chol_solver = make_LinalgSolverCholesky<decltype(K)>();

    LinalgSolverCholesky_Type<decltype(K)> Chol_solver_copy = Chol_solver;
    LinalgSolverCholesky_Type<decltype(K)> Chol_solver_move = std::move(Chol_solver_copy);
    Chol_solver = std::move(Chol_solver_move);

    auto A_ch = Chol_solver.solve(K);
    auto A_ch_dense = A_ch.create_dense();

    Matrix<DefDense, T, 3, 3> A_ch_answer({
        {3.16228F, 0.316228F, 0.632456F},
        {0, 4.46094F, 0.851838F},
        {0, 0, 5.37349F}
        });
    tester.expect_near(A_ch_dense.matrix.data, A_ch_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverCholesky solve Dense.");

    /* ゼロフラグの確認 */
    auto Zeros = make_DenseMatrixZeros<T, 3, 3>();

    static auto Chol_solver_zero = make_LinalgSolverCholesky<decltype(Zeros)>();
    auto Zeros_ch = Chol_solver_zero.solve(Zeros);
    auto Zeros_ch_dense = Zeros_ch.create_dense();

    Matrix<DefDense, T, 3, 3> Zeros_ch_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(Zeros_ch_dense.matrix.data, Zeros_ch_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverCholesky solve Dense Zero.");

    bool zero_div_flag = Chol_solver_zero.get_zero_div_flag();

    tester.expect_near(zero_div_flag, true, 0,
        "check LinalgSolverCholesky zero_div_flag.");

    /* コレスキー分解　対角 */
    LinalgSolverCholesky_Type<decltype(B)> Chol_solver_B;
    Chol_solver_B = make_LinalgSolverCholesky<decltype(B)>();

    auto B_ch = Chol_solver_B.solve(B);
    auto B_ch_dense = B_ch.create_dense();

    Matrix<DefDense, T, 3, 3> B_ch_answer({
        {1.0F, 0.0F, 0.0F},
        {0.0F, 1.41421356F, 0.0F},
        {0.0F, 0.0F, 1.73205081F}
        });

    tester.expect_near(B_ch_dense.matrix.data, B_ch_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverCholesky solve Diag.");

    /* コレスキー分解 スパース */
    LinalgSolverCholesky_Type<decltype(K_s)> Chol_solver_s;
    Chol_solver_s = make_LinalgSolverCholesky<decltype(K_s)>();

    auto K_s_ch = Chol_solver_s.solve(K_s);
    auto K_s_ch_dense = K_s_ch.create_dense();

    Matrix<DefDense, T, 3, 3> K_s_ch_answer({
        {1.0F, 0.0F, 0.0F},
        {0.0F, 2.82842712F, 1.06066017F},
        {0.0F, 0.0F, 1.6955825F}
        });

    tester.expect_near(K_s_ch_dense.matrix.data, K_s_ch_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverCholesky solve Sparse.");


    tester.throw_error_if_test_failed();
}


template<typename T>
void CheckPythonNumpy<T>::check_python_numpy_transpose_operation(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });


    /* 転置積 */
    A_mul_BTranspose_Type<decltype(A), decltype(A)> A_At
        = A_mul_BTranspose(A, A);

    Matrix<DefDense, T, 3, 3> A_At_answer({
        {14, 31, 46},
        {31, 77, 119},
        {46, 119, 194}
        });

    tester.expect_near(A_At.matrix.data, A_At_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Dense and Dense.");

    auto A_Bt = A_mul_BTranspose(A, B);

    Matrix<DefDense, T, 3, 3> A_Bt_answer({
        {1, 4, 9},
        {5, 8, 18},
        {9, 16, 21}
        });

    tester.expect_near(A_Bt.matrix.data, A_Bt_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Dense and Diag.");

    auto A_Ct = A_mul_BTranspose(A, C);

    Matrix<DefDense, T, 3, 3> A_Ct_answer({
        {1, 27, 16},
        {5, 63, 32},
        {9, 83, 44}
        });

    tester.expect_near(A_Ct.matrix.data, A_Ct_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Dense and Sparse.");

    auto B_At = A_mul_BTranspose(B, A);

    Matrix<DefDense, T, 3, 3> B_At_answer({
        {1, 5, 9},
        {4, 8, 16},
        {9, 18, 21}
        });

    tester.expect_near(B_At.matrix.data, B_At_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Diag and Dense.");

    auto B_Bt = A_mul_BTranspose(B, B);
    auto B_Bt_dense = B_Bt.create_dense();

    Matrix<DefDense, T, 3, 3> B_Bt_answer({
        {1, 0, 0},
        {0, 4, 0},
        {0, 0, 9}
        });

    tester.expect_near(B_Bt_dense.matrix.data, B_Bt_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Diag and Diag.");

    auto B_Ct = A_mul_BTranspose(B, C);
    auto B_Ct_dense = B_Ct.create_dense();

    Matrix<DefDense, T, 3, 3> B_Ct_answer({
        {1, 3, 0},
        {0, 0, 4},
        {0, 24, 12}
        });

    tester.expect_near(B_Ct_dense.matrix.data, B_Ct_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Diag and Sparse.");

    auto C_At = A_mul_BTranspose(C, A);

    Matrix<DefDense, T, 3, 3> C_At_answer({
        {1, 5, 9},
        {27, 63, 83},
        {16, 32, 44}
        });

    tester.expect_near(C_At.matrix.data, C_At_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Sparse and Dense.");

    auto C_Bt = A_mul_BTranspose(C, B);
    auto C_Bt_dense = C_Bt.create_dense();

    Matrix<DefDense, T, 3, 3> C_Bt_answer({
        {1, 0, 0},
        {3, 0, 24},
        {0, 4, 12}
        });

    tester.expect_near(C_Bt_dense.matrix.data, C_Bt_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Sparse and Diag.");

    A_mul_BTranspose_Type<decltype(C), decltype(C)> C_Ct
        = A_mul_BTranspose(C, C);
    auto C_Ct_dense = C_Ct.create_dense();

    Matrix<DefDense, T, 3, 3> C_Ct_answer({
        {1, 3, 0},
        {3, 73, 32},
        {0, 32, 20}
        });

    tester.expect_near(C_Ct_dense.matrix.data, C_Ct_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Sparse and Sparse.");

    ATranspose_mul_B_Type<decltype(A), decltype(A)> At_A
        = ATranspose_mul_B(A, A);

    Matrix<DefDense, T, 3, 3> At_A_answer({
        {107, 94, 96},
        {94, 84, 86},
        {96, 86, 94}
        });

    tester.expect_near(At_A.matrix.data, At_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Dense and Dense.");

    auto At_B = ATranspose_mul_B(A, B);

    Matrix<DefDense, T, 3, 3> At_B_answer({
        {1, 10, 27},
        {2, 8, 24},
        {3, 12, 21}
        });

    tester.expect_near(At_B.matrix.data, At_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Dense and Diag.");

    auto At_C = ATranspose_mul_B(A, C);

    Matrix<DefDense, T, 3, 3> At_C_answer({
        {16, 18, 76},
        {14, 16, 64},
        {21, 14, 76}
        });

    tester.expect_near(At_C.matrix.data, At_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Dense and Sparse.");

    auto Bt_A = ATranspose_mul_B(B, A);

    Matrix<DefDense, T, 3, 3> Bt_A_answer({
        {1, 2, 3},
        {10, 8, 12},
        {27, 24, 21}
        });

    tester.expect_near(Bt_A.matrix.data, Bt_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Diag and Dense.");

    auto Bt_B = ATranspose_mul_B(B, B);
    auto Bt_B_dense = Bt_B.create_dense();

    Matrix<DefDense, T, 3, 3> Bt_B_answer({
        {1, 0, 0},
        {0, 4, 0},
        {0, 0, 9}
        });

    tester.expect_near(Bt_B_dense.matrix.data, Bt_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Diag and Diag.");

    auto Bt_C = ATranspose_mul_B(B, C);
    auto Bt_C_dense = Bt_C.create_dense();

    Matrix<DefDense, T, 3, 3> Bt_C_answer({
        {1, 0, 0},
        {6, 0, 16},
        {0, 6, 12}
        });

    tester.expect_near(Bt_C_dense.matrix.data, Bt_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Diag and Sparse.");

    auto Ct_A = ATranspose_mul_B(C, A);

    Matrix<DefDense, T, 3, 3> Ct_A_answer({
        {16, 14, 21},
        {18, 16, 14},
        {76, 64, 76}
        });

    tester.expect_near(Ct_A.matrix.data, Ct_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Sparse and Dense.");

    auto Ct_B = ATranspose_mul_B(C, B);
    auto Ct_B_dense = Ct_B.create_dense();

    Matrix<DefDense, T, 3, 3> Ct_B_answer({
        {1, 6, 0},
        {0, 0, 6},
        {0, 16, 12}
        });

    tester.expect_near(Ct_B_dense.matrix.data, Ct_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Sparse and Diag.");

    ATranspose_mul_B_Type<decltype(C), decltype(C)> Ct_C
        = ATranspose_mul_B(C, C);
    auto Ct_C_dense = Ct_C.create_dense();

    Matrix<DefDense, T, 3, 3> Ct_C_answer({
        {10, 0, 24},
        {0, 4, 8},
        {24, 8, 80}
        });

    tester.expect_near(Ct_C_dense.matrix.data, Ct_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Sparse and Sparse.");

    /* 転置積 矩形行列 */
    using Ak_S = DenseAvailable<3, 2>;
    Matrix<DefSparse, T, 3, 2, Ak_S> Ak_sparse({ 4, 10, 5, 18, 6, 23 });

    Matrix<DefDense, T, 3, 1> Bk({ {1}, {2}, {3} });

    auto Ak_T_mul_Bk = ATranspose_mul_B(Ak_sparse, Bk);

    Matrix<DefDense, T, 2, 1> Ak_T_mul_Bk_answer({
        {32},
        {115}
        });

    tester.expect_near(Ak_T_mul_Bk.matrix.data, Ak_T_mul_Bk_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Sparse and Dense, 3 x 2 rectangular.");

    Matrix<DefDense, T, 1, 2> Bl({ { 1, 2 } });

    auto Bl_mul_Ak_T = A_mul_BTranspose(Bl, Ak_sparse);

    Matrix<DefDense, T, 1, 3> Bl_mul_Ak_T_answer({
        {24, 41, 52}
        });

    tester.expect_near(Bl_mul_Ak_T.matrix.data, Bl_mul_Ak_T_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Dense and Sparse, 3 x 2 rectangular.");


    tester.throw_error_if_test_failed();
}

template<typename T>
void CheckPythonNumpy<T>::check_python_numpy_qr(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });


    /* QR分解 */
    LinalgSolverQR_Type<decltype(A)> QR_solver_dense = make_LinalgSolverQR<decltype(A)>();
    QR_solver_dense.set_division_min(static_cast<T>(1.0e-10));
    QR_solver_dense.solve(A);

    LinalgSolverQR_Type<decltype(A)> QR_solver_dense_copy = QR_solver_dense;
    LinalgSolverQR_Type<decltype(A)> QR_solver_dense_move = std::move(QR_solver_dense_copy);
    QR_solver_dense = QR_solver_dense_move;

    auto Q = QR_solver_dense.get_Q();
    auto R = QR_solver_dense.get_R();
    auto R_dense = R.create_dense();

    Matrix<DefDense, T, 3, 3> R_answer({
        { -10.34408043F, -9.087323F, -9.28067029F},
        {0.0F, 1.19187279F, 1.39574577F},
        {0.0F, 0.0F, -2.43332132F}
        });

    tester.expect_near(R_dense.matrix.data, R_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR R Dense.");

    auto QR_result = Q * R;

    Matrix<DefDense, T, 3, 3> QR_result_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    tester.expect_near(QR_result.matrix.data, QR_result_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR Q multiply R Dense.");

    LinalgSolverQR_Type<decltype(B)> QR_solver_diag = make_LinalgSolverQR<decltype(B)>();
    QR_solver_diag.solve(B);

    LinalgSolverQR_Type<decltype(B)> QR_solver_diag_copy = QR_solver_diag;
    LinalgSolverQR_Type<decltype(B)> QR_solver_diag_move = std::move(QR_solver_diag_copy);
    QR_solver_diag = QR_solver_diag_move;

    auto R_diag = QR_solver_diag.get_R();
    auto R_diag_dense = R_diag.create_dense();

    Matrix<DefDense, T, 3, 3> R_diag_answer({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
        });

    tester.expect_near(R_diag_dense.matrix.data, R_diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR R Diag.");

    LinalgSolverQR_Type<decltype(C)> QR_solver_sparse = make_LinalgSolverQR<decltype(C)>();
    QR_solver_sparse.set_division_min(static_cast<T>(1.0e-10));

    LinalgSolverQR_Type<decltype(C)> QR_solver_sparse_copy = QR_solver_sparse;
    LinalgSolverQR_Type<decltype(C)> QR_solver_sparse_move = std::move(QR_solver_sparse_copy);
    QR_solver_sparse = QR_solver_sparse_move;

    QR_solver_sparse.solve(C);
    auto C_QR = QR_solver_sparse.get_Q() * QR_solver_sparse.get_R();

    Matrix<DefDense, T, 3, 3> C_QR_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4}
        });

    tester.expect_near(C_QR.matrix.data, C_QR_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR Q multiply R.");

    T division_min = static_cast<T>(1.0e-10);

    /* 後退代入 */
    Matrix<DefDense, T, 3, 4> A_2({
        {1, 2, 3, 10},
        {5, 4, 6, 11},
        {9, 8, 7, 12}
    });

    Matrix<DefDense, T, 3, 4> R_1_A_2 = QR_solver_dense.backward_substitution(A_2);

    Matrix<DefDense, T, 3, 4> R_1_A_2_answer({
        {-4.26873404F, -3.57425466F, -5.09101295F, -9.72349365F},
        {8.52639051F, 7.20611792F, 8.40289246F, 15.00425545F},
        {-3.6986484F, -3.28768747F, -2.87672653F, -4.9315312F}
    });

    tester.expect_near(R_1_A_2.matrix.data, R_1_A_2_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgQR backward_substitution Dense.");

    LinalgSolverQR_Type<decltype(A)> QR_solver_bs;
    QR_solver_bs.set_division_min(division_min);
    QR_solver_bs.solve(A);

    auto R_1_A_bs = QR_solver_bs.backward_substitution(A);

    Matrix<DefDense, T, 3, 3> R_1_A_bs_answer({
        {-4.26873404F, -3.57425466F, -5.09101295F},
        {8.52639051F, 7.20611792F, 8.40289246F},
        {-3.6986484F, -3.28768747F, -2.87672653F}
        });

    tester.expect_near(R_1_A_bs.matrix.data, R_1_A_bs_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR backward_substitution Dense with LinalgSolverQR.");

    auto R_1_B = QR_solver_diag.backward_substitution(A);

    Matrix<DefDense, T, 3, 3> R_1_B_answer({
        {1.0F, 2.0F, 3.0F},
        {2.5F, 2.0F, 3.0F},
        {3.0F, 2.66666667F, 2.33333333F}
        });

    tester.expect_near(R_1_B.matrix.data, R_1_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR backward_substitution Diag.");

    auto R_1_C = QR_solver_sparse.backward_substitution(A);

    Matrix<DefDense, T, 3, 3> R_1_C_answer({
        {8.22192192F, 6.95701085F, 5.69209979F },
        { 4.61512474F, 4.32455532F, 2.53398591F },
        { -3.55756237F, -3.16227766F, -2.76699295F }
    });

    tester.expect_near(R_1_C.matrix.data, R_1_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR backward_substitution Sparse.");

    /* QR分解 矩形行列 */
    Matrix<DefDense, T, 4, 3> AL({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7}, {2, 2, 3} });

    LinalgSolverQR_Type<decltype(AL)> QR_solver_rect = make_LinalgSolverQR<decltype(AL)>();
    QR_solver_rect.set_division_min(static_cast<T>(1.0e-10));
    QR_solver_rect.solve(AL);

    auto AL_QR = QR_solver_rect.get_Q() * QR_solver_rect.get_R();

    tester.expect_near(AL_QR.matrix.data, AL.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR Rect Dense.");

    Matrix<DefSparse, T, 4, 3,
        SparseAvailable<
        ColumnAvailable<true, true, false>,
        ColumnAvailable<false, false, true>,
        ColumnAvailable<false, true, true>,
        ColumnAvailable<true, true, true>>
        >
        CL({ 1, 3, 2, 8, 4, 10, 1, 15 });

    LinalgSolverQR_Type<decltype(CL)> QR_solver_rect_sparse = make_LinalgSolverQR<decltype(CL)>();
    QR_solver_rect_sparse.set_division_min(static_cast<T>(1.0e-10));
    QR_solver_rect_sparse.solve(CL);

    auto CL_QR = QR_solver_rect_sparse.get_Q() * QR_solver_rect_sparse.get_R();

    Matrix<DefDense, T, 4, 3> CL_QR_answer({
        {1.0F, 3.0F, 0.0F},
        {0.0F, 0.0F, 2.0F},
        {0.0F, 8.0F, 4.0F},
        {10.0F, 1.0F, 15.0F}
        });

    tester.expect_near(CL_QR.matrix.data, CL_QR_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR Rect Sparse.");


    tester.throw_error_if_test_failed();
}

template<typename T>
void CheckPythonNumpy<T>::check_python_numpy_eig(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<DefDense, T, 3, 3> A({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefSparse, T, 3, 3,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        > C({ 1, 3, 8, 2, 4 });

    /* 対角行列の固有値 実数 */
    LinalgSolverEigReal_Type<decltype(B)> eig_solver_diag
        = make_LinalgSolverEigReal<decltype(B)>();

    LinalgSolverEigReal_Type<decltype(B)> eig_solver_diag_copy = eig_solver_diag;
    LinalgSolverEigReal_Type<decltype(B)> eig_solver_diag_move = std::move(eig_solver_diag_copy);
    eig_solver_diag = eig_solver_diag_move;

    eig_solver_diag.solve_eigen_values(B);

    auto eigen_values_diag = eig_solver_diag.get_eigen_values();

    Matrix<DefDense, T, 3, 1> eigen_values_diag_answer({ {1}, {2}, {3} });

    tester.expect_near(eigen_values_diag.matrix.data, eigen_values_diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEigReal eigen values Diag.");

    auto eigen_vectors_diag = eig_solver_diag.get_eigen_vectors();

    Matrix<DefDiag, T, 3> eigen_vectors_diag_answer = Matrix<DefDiag, T, 3>::identity();

    tester.expect_near(eigen_vectors_diag.matrix.data, eigen_vectors_diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEigReal eigen vectors Diag.");


    /* スパース行列の固有値 実数 */
    LinalgSolverEigReal_Type<decltype(C)> eig_solver_sparse
        = make_LinalgSolverEigReal<decltype(C)>();
    eig_solver_sparse.set_iteration_max(10);
    eig_solver_sparse.set_division_min(static_cast<T>(1.0e-20));

    LinalgSolverEigReal_Type<decltype(C)> eig_solver_sparse_copy = eig_solver_sparse;
    LinalgSolverEigReal_Type<decltype(C)> eig_solver_sparse_move = std::move(eig_solver_sparse_copy);
    eig_solver_sparse = eig_solver_sparse_move;

    eig_solver_sparse.solve_eigen_values(C);

    auto eigen_values_sparse = eig_solver_sparse.get_eigen_values();

    decltype(eigen_values_sparse) eigen_values_sparse_sorted = eigen_values_sparse;
    Base::Utility::sort(eigen_values_sparse_sorted.matrix.data);

    decltype(eigen_values_sparse) eigen_values_sparse_answer({
        {static_cast<T>(-2.47213595)},
        {static_cast<T>(1.0)},
        {static_cast<T>(6.47213595)}
        });

    tester.expect_near(eigen_values_sparse_sorted.matrix.data, eigen_values_sparse_answer.matrix.data,
        NEAR_LIMIT_SOFT * std::abs(eigen_values_sparse_answer(0, 0)),
        "check LinalgSolverEigReal eigen values Sparse.");

    eig_solver_sparse.solve_eigen_vectors(C);
    auto eigen_vectors_sparse = eig_solver_sparse.get_eigen_vectors();

    auto A_mul_V_sparse = C * eigen_vectors_sparse;
    auto V_mul_D_sparse = eigen_vectors_sparse * Matrix<DefDiag, T, 3>(eigen_values_sparse.matrix);

    tester.expect_near(A_mul_V_sparse.matrix.data, V_mul_D_sparse.matrix.data, NEAR_LIMIT_SOFT * static_cast<T>(10),
        "check LinalgSolverEigReal eigen vectors Sparse.");

    auto result_C = eig_solver_sparse.check_validity(C);

    Matrix<DefDense, T, 3, 3> result_C_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(result_C.matrix.data, result_C_answer.matrix.data, NEAR_LIMIT_SOFT * static_cast<T>(10),
        "check LinalgSolverEigReal check_validity Sparse.");


    /* 実数値のみの固有値 */
    Matrix<DefDense, T, 3, 3> A0({ {6, -3, 5}, {-1, 4, -5}, {-3, 3, -4} });
    //Matrix<DefDense, T, 4, 4> A2({ {11, 8, 5, 10}, {14, 1, 4, 15}, {2, 13, 16, 3}, {7, 12, 9, 6} });
    LinalgSolverEigReal_Type<decltype(A)> eig_solver
        = make_LinalgSolverEigReal<decltype(A0)>();
    eig_solver.set_iteration_max(10);
    eig_solver.solve_eigen_values(A0);

    LinalgSolverEigReal_Type<decltype(A)> eig_solver_copy = eig_solver;
    LinalgSolverEigReal_Type<decltype(A)> eig_solver_move = std::move(eig_solver_copy);
    eig_solver = eig_solver_move;

    auto eigen_values = eig_solver.get_eigen_values();

    decltype(eigen_values) eigen_values_sorted = eigen_values;
    Base::Utility::sort(eigen_values_sorted.matrix.data[0]);

    //Matrix<DefDense, T, 4, 1> eigen_values_answer({ {34}, {8.94427191F}, {0}, {-8.94427191F} });
    Matrix<DefDense, T, 3, 1> eigen_values_answer({ {1}, {2}, {3} });
    tester.expect_near(eigen_values_sorted.matrix.data, eigen_values_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEigReal eigen values.");

    eig_solver.set_iteration_max(5);
    eig_solver.continue_solving_eigen_values();
    eig_solver.continue_solving_eigen_values();
    eig_solver.continue_solving_eigen_values();
    eigen_values = eig_solver.get_eigen_values();

    eigen_values_sorted = eigen_values;
    Base::Utility::sort(eigen_values_sorted.matrix.data[0]);

    tester.expect_near(eigen_values_sorted.matrix.data, eigen_values_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEigReal eigen values, strict.");

    eig_solver.solve_eigen_vectors(A0);
    auto eigen_vectors = eig_solver.get_eigen_vectors();

    //std::cout << "eigen_vectors = " << std::endl;
    //for (size_t j = 0; j < eigen_vectors.cols(); ++j) {
    //    for (size_t i = 0; i < eigen_vectors.rows(); ++i) {
    //        std::cout << eigen_vectors(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    auto A_mul_V = A0 * eigen_vectors;
    auto V_mul_D = eigen_vectors * Matrix<DefDiag, T, 3>(eigen_values.matrix);

    tester.expect_near(A_mul_V.matrix.data, V_mul_D.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEigReal eigen vectors.");

    auto result_A0 = eig_solver.check_validity(A0);

    Matrix<DefDense, T, 3, 3> result_A0_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(result_A0.matrix.data, result_A0_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEigReal check_validity.");


    /* 対角行列の固有値 複素数 */
    static auto eig_solver_diag_comp = make_LinalgSolverEig<decltype(B)>();
    eig_solver_diag_comp.solve_eigen_values(B);

    LinalgSolverEig_Type<decltype(B)> eig_solver_diag_comp_copy = eig_solver_diag_comp;
    LinalgSolverEig_Type<decltype(B)> eig_solver_diag_comp_move = std::move(eig_solver_diag_comp_copy);
    eig_solver_diag_comp = eig_solver_diag_comp_move;

    auto eigen_values_comp_diag = eig_solver_diag_comp.get_eigen_values();

    Matrix<DefDense, T, 3, 1> eigen_values_comp_diag_real(
        Base::Matrix::get_real_matrix_from_complex_matrix(eigen_values_comp_diag.matrix));

    Matrix<DefDense, T, 3, 1> eigen_values_comp_diag_answer({ {1}, {2}, {3} });

    tester.expect_near(eigen_values_comp_diag_real.matrix.data,
        eigen_values_comp_diag_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen values Diag.");

    auto eigen_vectors_comp_diag = eig_solver_diag_comp.get_eigen_vectors();
    Matrix<DefDiag, T, 3> eigen_vectors_comp_diag_real;
    eigen_vectors_comp_diag_real(0) = eigen_vectors_comp_diag(0).real;
    eigen_vectors_comp_diag_real(1) = eigen_vectors_comp_diag(1).real;
    eigen_vectors_comp_diag_real(2) = eigen_vectors_comp_diag(2).real;

    Matrix<DefDiag, T, 3> eigen_vectors_comp_diag_answer = Matrix<DefDiag, T, 3>::identity();

    tester.expect_near(eigen_vectors_comp_diag_real.matrix.data,
        eigen_vectors_comp_diag_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen vectors Diag.");

    /* スパース行列の固有値 複素数 */
    LinalgSolverEig_Type<decltype(C)> eig_solver_comp_sparse
        = make_LinalgSolverEig<decltype(C)>();
    eig_solver_comp_sparse.set_iteration_max(10);
    eig_solver_comp_sparse.set_iteration_max_for_eigen_vector(30);
    eig_solver_comp_sparse.set_division_min(static_cast<T>(1.0e-20));

    LinalgSolverEig_Type<decltype(C)> eig_solver_comp_sparse_copy = eig_solver_comp_sparse;
    LinalgSolverEig_Type<decltype(C)> eig_solver_comp_sparse_move = std::move(eig_solver_comp_sparse_copy);
    eig_solver_comp_sparse = eig_solver_comp_sparse_move;

    eig_solver_comp_sparse.solve_eigen_values(C);

    auto eigen_values_comp_sparse = eig_solver_comp_sparse.get_eigen_values();

    decltype(eigen_values_comp_sparse) eigen_values_comp_sparse_sorted = eigen_values_comp_sparse;
    Base::Utility::sort(eigen_values_comp_sparse_sorted.matrix.data[0]);

    Matrix<DefDense, T, 3, 1> eigen_values_comp_sparse_real(
        Base::Matrix::get_real_matrix_from_complex_matrix(eigen_values_comp_sparse_sorted.matrix));

    decltype(eigen_values_comp_sparse_real) eigen_values_comp_sparse_answer({
        {static_cast<T>(-2.47213595)},
        {static_cast<T>(1.0)},
        {static_cast<T>(6.47213595)}
        });

    tester.expect_near(eigen_values_comp_sparse_real.matrix.data, eigen_values_comp_sparse_answer.matrix.data,
        NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen values Sparse.");

    eig_solver_comp_sparse.solve_eigen_vectors(C);
    auto eigen_vectors_comp_sparse = eig_solver_comp_sparse.get_eigen_vectors();

    auto A_mul_V_comp_sparse = Matrix<DefDense, Base::Matrix::Complex<T>, 3, 3>(
        Base::Matrix::convert_matrix_real_to_complex(Base::Matrix::output_dense_matrix(C.matrix))) 
        * eigen_vectors_comp_sparse;
    auto V_mul_D_comp_sparse = eigen_vectors_comp_sparse
        * Matrix<DefDiag, Base::Matrix::Complex<T>, 3>(eigen_values_comp_sparse.matrix);

    Matrix<DefDense, T, 3, 3> A_mul_V_real_sparse(Base::Matrix::get_real_matrix_from_complex_matrix(A_mul_V_comp_sparse.matrix));
    Matrix<DefDense, T, 3, 3> V_mul_D_real_sparse(Base::Matrix::get_real_matrix_from_complex_matrix(V_mul_D_comp_sparse.matrix));
    tester.expect_near(A_mul_V_real_sparse.matrix.data, V_mul_D_real_sparse.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen vectors real sparse.");

    Matrix<DefDense, T, 3, 3> A_mul_V_imag_sparse(Base::Matrix::get_imag_matrix_from_complex_matrix(A_mul_V_comp_sparse.matrix));
    Matrix<DefDense, T, 3, 3> V_mul_D_imag_sparse(Base::Matrix::get_imag_matrix_from_complex_matrix(V_mul_D_comp_sparse.matrix));
    tester.expect_near(A_mul_V_imag_sparse.matrix.data, V_mul_D_imag_sparse.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen vectors imag sparse.");

    auto result_C_comp = eig_solver_comp_sparse.check_validity(C);

    Matrix<DefDense, T, 3, 3> result_C_real(Base::Matrix::get_real_matrix_from_complex_matrix(result_C_comp.matrix));

    Matrix<DefDense, T, 3, 3> result_C_real_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(result_C_real.matrix.data, result_C_real_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig check_validity Sparse.");

    Matrix<DefDense, T, 3, 3> result_C_imag(Base::Matrix::get_imag_matrix_from_complex_matrix(result_C_comp.matrix));

    Matrix<DefDense, T, 3, 3> result_C_imag_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(result_C_imag.matrix.data, result_C_imag_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig check_validity Sparse.");


    /* 複素数固有値 */
    Matrix<DefDense, T, 3, 3> A1({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });
    Matrix<DefDense, Complex<T>, 3, 3> A1_comp({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });

    LinalgSolverEig_Type<decltype(A1)> eig_solver_comp
        = make_LinalgSolverEig<decltype(A1)>();
    eig_solver_comp.set_iteration_max(5);
    eig_solver_comp.set_iteration_max_for_eigen_vector(15);
    eig_solver_comp.set_division_min(static_cast<T>(1.0e-20F));

    LinalgSolverEig_Type<decltype(A1)> eig_solver_comp_copy = eig_solver_comp;
    LinalgSolverEig_Type<decltype(A1)> eig_solver_comp_move = std::move(eig_solver_comp_copy);
    eig_solver_comp = eig_solver_comp_move;

    eig_solver_comp.solve_eigen_values(A1);

    auto eigen_values_comp = eig_solver_comp.get_eigen_values();

    decltype(eigen_values_comp) eigen_values_comp_sorted = eigen_values_comp;
    Base::Utility::sort(eigen_values_comp_sorted.matrix.data[0]);


    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_real({ {-1.5F}, {-1.5F}, {6.0F} });
    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_imag({ {-0.8660254F}, {0.8660254F}, {0.0F} });

    Matrix<DefDense, T, 3, 1> eigen_values_comp_real(
        Base::Matrix::get_real_matrix_from_complex_matrix(eigen_values_comp_sorted.matrix));
    tester.expect_near(eigen_values_comp_real.matrix.data, eigen_values_comp_answer_real.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen values real.");

    Matrix<DefDense, T, 3, 1> eigen_values_comp_imag(
        Base::Matrix::get_imag_matrix_from_complex_matrix(eigen_values_comp_sorted.matrix));
    tester.expect_near(eigen_values_comp_imag.matrix.data, eigen_values_comp_answer_imag.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen values imag.");

    eig_solver_comp.continue_solving_eigen_values();
    eigen_values_comp = eig_solver_comp.get_eigen_values();

    eigen_values_comp_sorted = eigen_values_comp;
    Base::Utility::sort(eigen_values_comp_sorted.matrix.data[0]);

    tester.expect_near(eigen_values_comp_real.matrix.data, eigen_values_comp_answer_real.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEig eigen values real, strict.");
    tester.expect_near(eigen_values_comp_imag.matrix.data, eigen_values_comp_answer_imag.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEig eigen values imag, strict.");

    eig_solver_comp.solve_eigen_vectors(A1);
    auto eigen_vectors_comp = eig_solver_comp.get_eigen_vectors();

    //std::cout << "eigen_vectors_comp = " << std::endl;
    //for (size_t j = 0; j < eigen_vectors_comp.cols(); ++j) {
    //    for (size_t i = 0; i < eigen_vectors_comp.rows(); ++i) {
    //        std::cout << eigen_vectors_comp(j, i).real << " + " << eigen_vectors_comp(j, i).imag << "j, ";;
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;


    eig_solver_comp.solve_eigen_vectors(A1);
    eigen_vectors_comp = eig_solver_comp.get_eigen_vectors();

    auto A_mul_V_comp = A1_comp * eigen_vectors_comp;
    auto V_mul_D_comp = eigen_vectors_comp * Matrix<DefDiag, Complex<T>, 3>(eigen_values_comp.matrix);

    Matrix<DefDense, T, 3, 3> A_mul_V_real(Base::Matrix::get_real_matrix_from_complex_matrix(A_mul_V_comp.matrix));
    Matrix<DefDense, T, 3, 3> V_mul_D_real(Base::Matrix::get_real_matrix_from_complex_matrix(V_mul_D_comp.matrix));
    tester.expect_near(A_mul_V_real.matrix.data, V_mul_D_real.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen vectors real.");

    Matrix<DefDense, T, 3, 3> A_mul_V_imag(Base::Matrix::get_imag_matrix_from_complex_matrix(A_mul_V_comp.matrix));
    Matrix<DefDense, T, 3, 3> V_mul_D_imag(Base::Matrix::get_imag_matrix_from_complex_matrix(V_mul_D_comp.matrix));
    tester.expect_near(A_mul_V_imag.matrix.data, V_mul_D_imag.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen vectors imag.");

    eig_solver_comp.solve_eigen_vectors(A1);
    eigen_vectors_comp = eig_solver_comp.get_eigen_vectors();
    A_mul_V_comp = A1_comp * eigen_vectors_comp;
    V_mul_D_comp = eigen_vectors_comp * Matrix<DefDiag, Complex<T>, 3>(eigen_values_comp.matrix);

    A_mul_V_real.matrix = Base::Matrix::get_real_matrix_from_complex_matrix(A_mul_V_comp.matrix);
    V_mul_D_real.matrix = Base::Matrix::get_real_matrix_from_complex_matrix(V_mul_D_comp.matrix);
    tester.expect_near(A_mul_V_real.matrix.data, V_mul_D_real.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEig eigen vectors real, strict.");

    A_mul_V_imag.matrix = Base::Matrix::get_imag_matrix_from_complex_matrix(A_mul_V_comp.matrix);
    V_mul_D_imag.matrix = Base::Matrix::get_imag_matrix_from_complex_matrix(V_mul_D_comp.matrix);
    tester.expect_near(A_mul_V_imag.matrix.data, V_mul_D_imag.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEig eigen vectors imag, strict.");

    auto result_A1_comp = eig_solver_comp.check_validity(A1);

    Matrix<DefDense, T, 3, 3> result_A1_real(Base::Matrix::get_real_matrix_from_complex_matrix(result_A1_comp.matrix));

    Matrix<DefDense, T, 3, 3> result_A1_real_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(result_A1_real.matrix.data, result_A1_real_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEig check_validity real.");

    Matrix<DefDense, T, 3, 3> result_A1_imag(Base::Matrix::get_imag_matrix_from_complex_matrix(result_A1_comp.matrix));

    Matrix<DefDense, T, 3, 3> result_A1_imag_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(result_A1_imag.matrix.data, result_A1_imag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEig check_validity imag.");


    tester.throw_error_if_test_failed();
}


#endif // __CHECK_PYTHON_NUMPY_HPP__
