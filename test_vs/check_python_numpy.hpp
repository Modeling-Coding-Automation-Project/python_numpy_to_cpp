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
    void check_python_numpy_left_divide_and_inv(void);
    void check_python_numpy_concatenate(void);
    void check_python_numpy_transpose(void);
    void check_python_numpy_lu(void);
    void check_check_python_numpy_cholesky(void);
    void check_check_python_numpy_transpose_operation(void);
    void check_python_numpy_qr(void);
    void check_python_numpy_eig(void);

    void calc(void);
};

template <typename T>
void CheckPythonNumpy<T>::calc(void) {

    check_python_numpy_base();

    check_python_numpy_left_divide_and_inv();

    check_python_numpy_concatenate();

    check_python_numpy_transpose();

    check_python_numpy_lu();

    check_check_python_numpy_cholesky();

    check_check_python_numpy_transpose_operation();

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

    T a_value = A(1, 2);

    tester.expect_near(a_value, 6, NEAR_LIMIT_STRICT,
        "check Matrix get value.");

    T a_value_outlier = A(100, 100);

    tester.expect_near(a_value_outlier, 7, NEAR_LIMIT_STRICT,
        "check Matrix get value outlier.");

    A(1, 2) = 100;

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

    T b_value = B(1);

    tester.expect_near(b_value, 2, NEAR_LIMIT_STRICT,
        "check DiagMatrix get value.");

    T b_value_outlier = B(100);

    tester.expect_near(b_value_outlier, 3, NEAR_LIMIT_STRICT,
        "check DiagMatrix get value outlier.");

    B(1) = 100;

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

    T c_value = C(2);

    tester.expect_near(c_value, 8, NEAR_LIMIT_STRICT,
        "check SparseMatrix get value.");

    T c_value_outlier = C(100);

    tester.expect_near(c_value_outlier, 4, NEAR_LIMIT_STRICT,
        "check SparseMatrix get value outlier.");

    C(2) = 100;

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

    auto Dense_mul_Diag = AA * B;

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


    tester.throw_error_if_test_failed();
}




template <typename T>
void CheckPythonNumpy<T>::check_python_numpy_left_divide_and_inv(void) {
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

    /* 左除算 */
    Matrix<DefDense, T, 3, 2> b({ { 4, 10 }, { 5, 18 }, { 6, 23 } });

    static auto A_A_linalg_solver = make_LinalgSolver(A, A);

    auto A_A_x = A_A_linalg_solver.solve(A, A);

    Matrix<DefDense, T, 3, 3> A_A_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(A_A_x.matrix.data, A_A_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Dense.");

    static auto A_B_linalg_solver = make_LinalgSolver(A, B);

    auto A_B_x = A_B_linalg_solver.solve(A, B);

    Matrix<DefDense, T, 3, 3> A_B_x_answer({
        {-6.66666667e-01F, 6.66666667e-01F, 0.0F},
        {6.33333333e-01F, -1.33333333F, 0.9F},
        {1.33333333e-01F, 6.66666667e-01F, -0.6F}
        });

    tester.expect_near(A_B_x.matrix.data, A_B_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Diag.");

    static auto A_C_linalg_solver = make_LinalgSolver(A, C);

    auto A_C_x = A_C_linalg_solver.solve(A, C);

    Matrix<DefDense, T, 3, 3> A_C_x_answer({
        { 3.33333333e-01F, 0.0F, 2.66666667F},
        {-1.36666667F, 0.6F, -4.13333333F},
        {1.13333333F, -0.4F, 1.86666667F}
        });

    tester.expect_near(A_C_x.matrix.data, A_C_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Dense and Sparse.");

    static auto B_A_linalg_solver = make_LinalgSolver(B, A);

    auto B_A_x = B_A_linalg_solver.solve(B, A);

    Matrix<DefDense, T, 3, 3> B_A_x_answer({
        { 1.0F, 2.0F, 3.0F },
        { 2.5F, 2.0F, 3.0F },
        { 3.0F, 2.66666667F, 2.33333333F }
        });

    tester.expect_near(B_A_x.matrix.data, B_A_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Dense.");

    static auto B_B_linalg_solver = make_LinalgSolver(B, B);

    auto B_B_x = B_B_linalg_solver.solve(B, B);
    auto B_B_x_dense = B_B_x.create_dense();

    Matrix<DefDense, T, 3, 3> B_B_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(B_B_x_dense.matrix.data, B_B_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Diag.");

    static auto B_C_linalg_solver = make_LinalgSolver(B, C);

    auto B_C_x = B_C_linalg_solver.solve(B, C);

    Matrix<DefDense, T, 3, 3> B_C_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 1.5F, 0.0F, 4.0F },
        { 0.0F, 0.66666667F, 1.33333333F }
        });

    tester.expect_near(B_C_x.matrix.data, B_C_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Diag and Sparse.");

    static auto C_C_linalg_solver = make_LinalgSolver(C, C);

    auto C_C_x = C_C_linalg_solver.solve(C, C);

    Matrix<DefDense, T, 3, 3> C_C_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(C_C_x.matrix.data, C_C_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve Sparse and Sparse.");

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

    static auto AL_AL_lstsq_solver = make_LinalgLstsqSolver(AL, AL);

    auto AL_AL_x = AL_AL_lstsq_solver.solve(AL, AL);

    Matrix<DefDense, T, 3, 3> AL_AL_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(AL_AL_x.matrix.data, AL_AL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Dense and Dense.");

    static auto AL_BL_lstsq_solver = make_LinalgLstsqSolver(AL, BL);

    auto AL_BL_x = AL_BL_lstsq_solver.solve(AL, BL);

    Matrix<DefDense, T, 3, 4> AL_BL_x_answer({
        {-6.36363636e-01F, 7.27272727e-01F, 0.0F, -3.63636364e-01F },
        {6.36363636e-01F, -1.32727273e+00F, 0.9F, -3.63636364e-02F },
        {9.09090909e-02F, 5.81818182e-01F, -0.6F, 5.09090909e-01F }
        });

    tester.expect_near(AL_BL_x.matrix.data, AL_BL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Dense and Diag.");

    static auto AL_CL_lstsq_solver = make_LinalgLstsqSolver(AL, CL);

    auto AL_CL_x = AL_CL_lstsq_solver.solve(AL, CL);

    Matrix<DefDense, T, 3, 3> AL_CL_x_answer({
        {-0.63636364F, -2.0F, 0.72727273F },
        { 0.63636364F, 4.3F, -0.12727273F },
        { 0.09090909F, -1.2F, -0.21818182F }
        });

    tester.expect_near(AL_CL_x.matrix.data, AL_CL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Dense and Sparse.");

    static auto CL_AL_lstsq_solver = make_LinalgLstsqSolver(CL, AL);

    auto CL_AL_x = CL_AL_lstsq_solver.solve(CL, AL);

    Matrix<DefDense, T, 3, 3> CL_AL_x_answer({
        { 0.91304348F, 1.56521739F, 4.08695652F },
        { 0.02898551F, 0.14492754F, -0.36231884F },
        { 2.25362319F, 1.76811594F, 2.57971014F }
        });

    tester.expect_near(CL_AL_x.matrix.data, CL_AL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Sparse and Dense.");

    static auto CL_CL_lstsq_solver = make_LinalgLstsqSolver(CL, CL);

    auto CL_CL_x = CL_CL_lstsq_solver.solve(CL, CL);

    Matrix<DefDense, T, 3, 3> CL_CL_x_answer({
        { 1.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 1.0F }
        });

    tester.expect_near(CL_CL_x.matrix.data, CL_CL_x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve Sparse and Sparse.");

    /* 逆行列 */
    static auto A_inv_solver = make_LinalgSolver(A);

    auto A_Inv = A_inv_solver.inv(A);

    Matrix<DefDense, T, 3, 3> A_Inv_answer({
        {-6.66666667e-01F, 3.33333333e-01F, 0.0F },
        {6.33333333e-01F, -6.66666667e-01F, 3.00000000e-01F },
        {1.33333333e-01F, 3.33333333e-01F, -0.2F }
        });

    tester.expect_near(A_Inv.matrix.data, A_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Dense.");

    static auto B_inv_solver = make_LinalgSolver(B);

    auto B_Inv = B_inv_solver.inv(B);
    auto B_Inv_dense = B_Inv.create_dense();

    Matrix<DefDense, T, 3, 3> B_Inv_answer({
        {1.0F, 0.0F, 0.0F},
        {0.0F, 0.5F, 0.0F},
        {0.0F, 0.0F, 0.33333333F}
        });

    tester.expect_near(B_Inv_dense.matrix.data, B_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Diag.");

    static auto C_inv_solver = make_LinalgSolver(C);

    auto C_Inv = C_inv_solver.inv(C);

    Matrix<DefDense, T, 3, 3> Inv_answer({
        {1.0F, 0.0F, 0.0F},
        {0.75F, -0.25F, 0.5F},
        {-0.375F, 0.125F, 0}
        });

    tester.expect_near(C_Inv.matrix.data, Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv Sparse.");


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
    auto A_v_A = concatenate_vertically(A, A);

    Matrix<DefDense, T, 6, 3> A_v_A_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 }
        });

    tester.expect_near(A_v_A.matrix.data, A_v_A_answer.matrix.data, NEAR_LIMIT_STRICT,
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
    auto A_v_B_dense = A_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_B_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 }
        });

    tester.expect_near(A_v_B_dense.matrix.data, A_v_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Diag.");

    update_vertically_concatenated_matrix(A_v_B, A, static_cast<T>(2) * B);
    A_v_B_dense = A_v_B.create_dense();

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
    auto A_v_C_dense = A_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_C_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 }
        });

    tester.expect_near(A_v_C_dense.matrix.data, A_v_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Sparse.");

    update_vertically_concatenated_matrix(A_v_C, A, static_cast<T>(2) * C);
    A_v_C_dense = A_v_C.create_dense();

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
    auto A_v_E_dense = A_v_E.create_dense();

    Matrix<DefDense, T, 6, 3> A_v_E_answer({
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 }
        });

    tester.expect_near(A_v_E_dense.matrix.data, A_v_E_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Empty.");

    auto B_v_A = concatenate_vertically(B, A);
    auto B_v_A_dense = B_v_A.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_A_answer({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 }
        });

    tester.expect_near(B_v_A_dense.matrix.data, B_v_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Dense.");

    update_vertically_concatenated_matrix(B_v_A, static_cast<T>(2) * B, A);
    B_v_A_dense = B_v_A.create_dense();

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
    auto B_v_B_dense = B_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_B_answer({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 2, 0, 0 },
        { 0, 4, 0 },
        { 0, 0, 6 }
        });

    tester.expect_near(B_v_B_dense.matrix.data, B_v_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Diag.");

    update_vertically_concatenated_matrix(B_v_B, static_cast<T>(2) * B, B);
    B_v_B_dense = B_v_B.create_dense();

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
    auto B_v_C_dense = B_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> B_v_C_answer({
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 },
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 }
        });

    tester.expect_near(B_v_C_dense.matrix.data, B_v_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Sparse.");

    update_vertically_concatenated_matrix(B_v_C, B, static_cast<T>(2) * C);
    B_v_C_dense = B_v_C.create_dense();

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
    auto C_v_A_dense = C_v_A.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_A_answer({
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 },
        { 1, 2, 3 },
        { 5, 4, 6 },
        { 9, 8, 7 },
        });

    tester.expect_near(C_v_A_dense.matrix.data, C_v_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Dense.");

    update_vertically_concatenated_matrix(C_v_A, C, static_cast<T>(2) * A);
    C_v_A_dense = C_v_A.create_dense();

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
    auto C_v_B_dense = C_v_B.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_B_answer({
        { 1, 0, 0 },
        { 3, 0, 8 },
        { 0, 2, 4 },
        { 1, 0, 0 },
        { 0, 2, 0 },
        { 0, 0, 3 }
        });

    tester.expect_near(C_v_B_dense.matrix.data, C_v_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Diag.");

    update_vertically_concatenated_matrix(C_v_B, C, static_cast<T>(2) * B);
    C_v_B_dense = C_v_B.create_dense();

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
    auto C_v_C_dense = C_v_C.create_dense();

    Matrix<DefDense, T, 6, 3> C_v_C_answer({
    { 1, 0, 0 },
    { 3, 0, 8 },
    { 0, 2, 4 },
    { 2, 0, 0 },
    { 6, 0, 16 },
    { 0, 4, 8 }
        });

    tester.expect_near(C_v_C_dense.matrix.data, C_v_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Sparse.");

    update_vertically_concatenated_matrix(C_v_C, static_cast<T>(2) * C, C);
    C_v_C_dense = C_v_C.create_dense();

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

    Matrix<DefDense, T, 4, 6> AL_h_AL_answer({
        {1, 2, 3, 1, 2, 3},
        {5, 4, 6, 5, 4, 6},
        {9, 8, 7, 9, 8, 7},
        {2, 2, 3, 2, 2, 3}
        });

    tester.expect_near(AL_h_AL.matrix.data, AL_h_AL_answer.matrix.data, NEAR_LIMIT_STRICT,
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
    auto AL_h_BL_dense = AL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 7> AL_h_BL_answer({
        {1, 2, 3, 1, 0, 0, 0},
        {5, 4, 6, 0, 2, 0, 0},
        {9, 8, 7, 0, 0, 3, 0},
        {2, 2, 3, 0, 0, 0, 4}
        });

    tester.expect_near(AL_h_BL_dense.matrix.data, AL_h_BL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Diag.");

    update_horizontally_concatenated_matrix(AL_h_BL, AL, static_cast<T>(2) * BL);
    AL_h_BL_dense = AL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 7> AL_h_BL_answer_2({
        {1, 2, 3, 2, 0, 0, 0},
        {5, 4, 6, 0, 4, 0, 0},
        {9, 8, 7, 0, 0, 6, 0},
        {2, 2, 3, 0, 0, 0, 8}
        });

    tester.expect_near(AL_h_BL_dense.matrix.data, AL_h_BL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Dense and Diag.");

    auto AL_h_CL = concatenate_horizontally(AL, CL);
    auto AL_h_CL_dense = AL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 6> AL_h_CL_answer({
        {1, 2, 3, 1, 3, 0},
        {5, 4, 6, 0, 0, 2},
        {9, 8, 7, 0, 8, 4},
        {2, 2, 3, 0, 1, 0}
        });

    tester.expect_near(AL_h_CL_dense.matrix.data, AL_h_CL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Sparse.");

    update_horizontally_concatenated_matrix(AL_h_CL, AL, static_cast<T>(2) * CL);
    AL_h_CL_dense = AL_h_CL.create_dense();

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
    auto BL_h_AL_dense = BL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_AL_answer({
        {1, 0, 0, 0, 1, 2, 3},
        {0, 2, 0, 0, 5, 4, 6},
        {0, 0, 3, 0, 9, 8, 7},
        {0, 0, 0, 4, 2, 2, 3}
        });

    tester.expect_near(BL_h_AL_dense.matrix.data, BL_h_AL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Dense.");

    update_horizontally_concatenated_matrix(BL_h_AL, static_cast<T>(2) * BL, AL);
    BL_h_AL_dense = BL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_AL_answer_2({
        {2, 0, 0, 0, 1, 2, 3},
        {0, 4, 0, 0, 5, 4, 6},
        {0, 0, 6, 0, 9, 8, 7},
        {0, 0, 0, 8, 2, 2, 3}
        });

    tester.expect_near(BL_h_AL_dense.matrix.data, BL_h_AL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Diag and Dense.");

    auto BL_h_BL = concatenate_horizontally(BL, BL * static_cast<T>(2));
    auto BL_h_BL_dense = BL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 8> BL_h_BL_answer({
        {1, 0, 0, 0, 2, 0, 0, 0},
        {0, 2, 0, 0, 0, 4, 0, 0},
        {0, 0, 3, 0, 0, 0, 6, 0},
        {0, 0, 0, 4, 0, 0, 0, 8}
        });

    tester.expect_near(BL_h_BL_dense.matrix.data, BL_h_BL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Diag.");

    update_horizontally_concatenated_matrix(BL_h_BL, static_cast<T>(2) * BL, BL);
    BL_h_BL_dense = BL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 8> BL_h_BL_answer_2({
        {2, 0, 0, 0, 1, 0, 0, 0},
        {0, 4, 0, 0, 0, 2, 0, 0},
        {0, 0, 6, 0, 0, 0, 3, 0},
        {0, 0, 0, 8, 0, 0, 0, 4}
        });

    tester.expect_near(BL_h_BL_dense.matrix.data, BL_h_BL_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check update horizontally concatenated matrix Diag and Diag.");

    auto BL_h_CL = concatenate_horizontally(BL, CL);
    auto BL_h_CL_dense = BL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 7> BL_h_CL_answer({
        {1, 0, 0, 0, 1, 3, 0},
        {0, 2, 0, 0, 0, 0, 2},
        {0, 0, 3, 0, 0, 8, 4},
        {0, 0, 0, 4, 0, 1, 0}
        });

    tester.expect_near(BL_h_CL_dense.matrix.data, BL_h_CL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Sparse.");

    update_horizontally_concatenated_matrix(BL_h_CL, static_cast<T>(2) * BL, CL);
    BL_h_CL_dense = BL_h_CL.create_dense();

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
    auto CL_h_AL_dense = CL_h_AL.create_dense();

    Matrix<DefDense, T, 4, 6> CL_h_AL_answer({
        {1, 3, 0, 1, 2, 3},
        {0, 0, 2, 5, 4, 6},
        {0, 8, 4, 9, 8, 7},
        {0, 1, 0, 2, 2, 3}
        });

    tester.expect_near(CL_h_AL_dense.matrix.data, CL_h_AL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Dense.");

    update_horizontally_concatenated_matrix(CL_h_AL, CL, static_cast<T>(2) * AL);
    CL_h_AL_dense = CL_h_AL.create_dense();

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
    auto CL_h_BL_dense = CL_h_BL.create_dense();

    Matrix<DefDense, T, 4, 7> CL_h_BL_answer({
        {1, 3, 0, 1, 0, 0, 0},
        {0, 0, 2, 0, 2, 0, 0},
        {0, 8, 4, 0, 0, 3, 0},
        {0, 1, 0, 0, 0, 0, 4}
        });

    tester.expect_near(CL_h_BL_dense.matrix.data, CL_h_BL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Diag.");

    update_horizontally_concatenated_matrix(CL_h_BL, CL, static_cast<T>(2) * BL);
    CL_h_BL_dense = CL_h_BL.create_dense();

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
    auto CL_h_CL_dense = CL_h_CL.create_dense();

    Matrix<DefDense, T, 4, 6> CL_h_CL_answer({
        {1, 3, 0, 1, 3, 0},
        {0, 0, 2, 0, 0, 2},
        {0, 8, 4, 0, 8, 4},
        {0, 1, 0, 0, 1, 0}
        });

    tester.expect_near(CL_h_CL_dense.matrix.data, CL_h_CL_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Sparse.");

    update_horizontally_concatenated_matrix(CL_h_CL, static_cast<T>(2) * CL, CL);
    CL_h_CL_dense = CL_h_CL.create_dense();

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
    auto AT = A.transpose();

    Matrix<DefDense, T, 3, 3> AT_answer({
        {1, 5, 9},
        {2, 4, 8},
        {3, 6, 7}
        });

    tester.expect_near(AT.matrix.data, AT_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check transpose Dense.");

    auto CL_T = CL.transpose();
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
    static auto A_LU_solver = make_LinalgSolverLU(A);

    auto A_LU = A_LU_solver.get_L() * A_LU_solver.get_U();
    auto A_LU_dense = A_LU.create_dense();

    Matrix<DefDense, T, 3, 3> A_LU_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    tester.expect_near(A_LU_dense.matrix.data, A_LU_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverLU L multiply U Dense.");

    //std::cout << "det = " << LU_solver.get_det() << std::endl << std::endl;

    T det_answer = 30;
    tester.expect_near(A_LU_solver.get_det(), det_answer, NEAR_LIMIT_STRICT,
        "check LinalgSolverLU det.");

    auto B_LU_solver = make_LinalgSolverLU(B);

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
void CheckPythonNumpy<T>::check_check_python_numpy_cholesky(void) {
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


    /* コレスキー分解 */
    Matrix<DefDense, T, 3, 3> K({
        {10, 1, 2},
        {1, 20, 4},
        {2, 4, 30}
        });
    Matrix<DefSparse, T, 3, 3, SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, true, true>,
        ColumnAvailable<false, true, true>>
        >K_s({ 1, 8, 3, 3, 4 });

    static auto Chol_solver = make_LinalgSolverCholesky(K);

    auto A_ch = Chol_solver.solve(K);
    auto A_ch_d = Base::Matrix::output_dense_matrix(A_ch.matrix);
    //std::cout << "A_ch_d = " << std::endl;
    //for (size_t j = 0; j < A_ch_d.cols(); ++j) {
    //    for (size_t i = 0; i < A_ch_d.rows(); ++i) {
    //        std::cout << A_ch_d(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 3> A_ch_answer({
        {3.16228F, 0.316228F, 0.632456F},
        {0, 4.46094F, 0.851838F},
        {0, 0, 5.37349F}
        });
    tester.expect_near(A_ch_d.data, A_ch_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverCholesky solve.");

    tester.throw_error_if_test_failed();
}


template<typename T>
void CheckPythonNumpy<T>::check_check_python_numpy_transpose_operation(void) {
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
    auto A_At = A_mul_BTranspose(A, A);

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

    auto C_Ct = A_mul_BTranspose(C, C);
    auto C_Ct_dense = C_Ct.create_dense();

    Matrix<DefDense, T, 3, 3> C_Ct_answer({
        {1, 3, 0},
        {3, 73, 32},
        {0, 32, 20}
        });

    tester.expect_near(C_Ct_dense.matrix.data, C_Ct_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check A_mul_BTranspose Sparse and Sparse.");

    auto At_A = ATranspose_mul_B(A, A);

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

    auto Ct_C = ATranspose_mul_B(C, C);
    auto Ct_C_dense = Ct_C.create_dense();

    Matrix<DefDense, T, 3, 3> Ct_C_answer({
        {10, 0, 24},
        {0, 4, 8},
        {24, 8, 80}
        });

    tester.expect_near(Ct_C_dense.matrix.data, Ct_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Sparse and Sparse.");


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
    static auto QR_solver_dense = make_LinalgSolverQR(A);

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

    static auto QR_solver_diag = make_LinalgSolverQR(B);

    auto R_diag = QR_solver_diag.get_R();
    auto R_diag_dense = R_diag.create_dense();

    Matrix<DefDense, T, 3, 3> R_diag_answer({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
        });

    tester.expect_near(R_diag_dense.matrix.data, R_diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR R Diag.");

    static auto QR_solver_sparse = make_LinalgSolverQR(C);
    auto C_QR = QR_solver_sparse.get_Q() * QR_solver_sparse.get_R();

    Matrix<DefDense, T, 3, 3> C_QR_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4}
        });

    tester.expect_near(C_QR.matrix.data, C_QR_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR Q multiply R.");


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
    static auto eig_solver_diag = make_LinalgSolverEigReal(B);

    auto eigen_values_diag = eig_solver_diag.get_eigen_values();

    Matrix<DefDense, T, 3, 1> eigen_values_diag_answer({ {1}, {2}, {3} });

    tester.expect_near(eigen_values_diag.matrix.data, eigen_values_diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEigReal eigen values Diag.");

    auto eigen_vectors_diag = eig_solver_diag.get_eigen_vectors();

    Matrix<DefDiag, T, 3> eigen_vectors_diag_answer = Matrix<DefDiag, T, 3>::identity();

    tester.expect_near(eigen_vectors_diag.matrix.data, eigen_vectors_diag_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverEigReal eigen vectors Diag.");

    /* スパース行列の固有値 実数 */
    static auto eig_solver_sparse = make_LinalgSolverEigReal(C);

    auto eigen_values_sparse = eig_solver_sparse.get_eigen_values();

    decltype(eigen_values_sparse) eigen_values_sparse_answer({
        {static_cast<T>(-2.47213595)},
        {static_cast<T>(1.0)},
        {static_cast<T>(6.47213595)}
        });

    tester.expect_near(eigen_values_sparse.matrix.data, eigen_values_sparse_answer.matrix.data,
        NEAR_LIMIT_SOFT * std::abs(eigen_values_sparse_answer(0, 0)),
        "check LinalgSolverEigReal eigen values Sparse.");

    eig_solver_sparse.solve_eigen_vectors(C);
    auto eigen_vectors_sparse = eig_solver_sparse.get_eigen_vectors();

    auto A_mul_V_sparse = C * eigen_vectors_sparse;
    auto V_mul_D_sparse = eigen_vectors_sparse * Matrix<DefDiag, T, 3>(eigen_values_sparse.matrix);

    tester.expect_near(A_mul_V_sparse.matrix.data, V_mul_D_sparse.matrix.data, NEAR_LIMIT_SOFT * static_cast<T>(10),
        "check LinalgSolverEigReal eigen vectors Sparse.");


    /* 実数値のみの固有値 */
    Matrix<DefDense, T, 3, 3> A0({ {6, -3, 5}, {-1, 4, -5}, {-3, 3, -4} });
    //Matrix<DefDense, T, 4, 4> A2({ {11, 8, 5, 10}, {14, 1, 4, 15}, {2, 13, 16, 3}, {7, 12, 9, 6} });
    static auto eig_solver = make_LinalgSolverEigReal(A0);

    auto eigen_values = eig_solver.get_eigen_values();

    //std::cout << "eigen_values = " << std::endl;
    //for (size_t j = 0; j < eigen_values.cols(); ++j) {
    //    for (size_t i = 0; i < eigen_values.rows(); ++i) {
    //        std::cout << eigen_values(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    //Matrix<DefDense, T, 4, 1> eigen_values_answer({ {34}, {8.94427191F}, {0}, {-8.94427191F} });
    Matrix<DefDense, T, 3, 1> eigen_values_answer({ {1}, {2}, {3} });
    tester.expect_near(eigen_values.matrix.data, eigen_values_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEigReal eigen values.");

    eig_solver.set_iteration_max(5);
    eig_solver.continue_solving_eigen_values();
    eig_solver.continue_solving_eigen_values();
    eig_solver.continue_solving_eigen_values();
    eigen_values = eig_solver.get_eigen_values();
    tester.expect_near(eigen_values.matrix.data, eigen_values_answer.matrix.data, NEAR_LIMIT_STRICT,
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

    /* 対角行列の固有値 複素数 */
    static auto eig_solver_diag_comp = make_LinalgSolverEig(B);
    eig_solver_diag_comp.solve_eigen_values(B);

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
    static auto eig_solver_comp_sparse = make_LinalgSolverEig(C);

    auto eigen_values_comp_sparse = eig_solver_comp_sparse.get_eigen_values();
    Matrix<DefDense, T, 3, 1> eigen_values_comp_sparse_real(
        Base::Matrix::get_real_matrix_from_complex_matrix(eigen_values_comp_sparse.matrix));

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

    /* 複素数固有値 */
    Matrix<DefDense, T, 3, 3> A1({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });
    Matrix<DefDense, Complex<T>, 3, 3> A1_comp({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });

    static auto eig_solver_comp = make_LinalgSolverEig<T, 3, 5>(A1);
    auto eigen_values_comp = eig_solver_comp.get_eigen_values();

    //std::cout << "eigen_values_comp = " << std::endl;
    //for (size_t j = 0; j < eigen_values_comp.cols(); ++j) {
    //    for (size_t i = 0; i < eigen_values_comp.rows(); ++i) {
    //        std::cout << "[" << eigen_values_comp(j, i).real << " ";
    //        std::cout << "+ " <<  eigen_values_comp(j, i).imag << "j] ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    eig_solver_comp.set_iteration_max(5);
    eig_solver_comp.set_iteration_max_for_eigen_vector(15);

    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_real({ {-1.5F}, {-1.5F}, {6.0F} });
    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_imag({ {-0.8660254F}, {0.8660254F}, {0.0F} });

    Matrix<DefDense, T, 3, 1> eigen_values_comp_real(
        Base::Matrix::get_real_matrix_from_complex_matrix(eigen_values_comp.matrix));
    tester.expect_near(eigen_values_comp_real.matrix.data, eigen_values_comp_answer_real.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen values real.");

    Matrix<DefDense, T, 3, 1> eigen_values_comp_imag(
        Base::Matrix::get_imag_matrix_from_complex_matrix(eigen_values_comp.matrix));
    tester.expect_near(eigen_values_comp_imag.matrix.data, eigen_values_comp_answer_imag.matrix.data, NEAR_LIMIT_SOFT,
        "check LinalgSolverEig eigen values imag.");

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

    tester.throw_error_if_test_failed();
}


#endif // __CHECK_PYTHON_NUMPY_HPP__
