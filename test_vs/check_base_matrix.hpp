#ifndef __CHECK_BASE_MATRIX_HPP__
#define __CHECK_BASE_MATRIX_HPP__

#include <type_traits>
#include <iostream>

#include "MCAP_tester.hpp"

#include "base_matrix.hpp"

template<typename T>
class CheckBaseMatrix {
public:
    /* Constructor */
    CheckBaseMatrix() {}

public:
    /* Function */
    void check_matrix_vector_creation(void);
    void check_matrix_swap(void);
    void check_matrix_multiply(void);
    void check_lu_decomposition(void);
    void check_gmres_k_and_inverse(void);
    void check_matrix_transpose_multiply(void);
    void check_determinant_and_trace(void);
    void check_diag_matrix(void);
    void check_sparse_matrix(void);
    void check_matrix_cocatenation(void);
    void check_cholesky_decomposition(void);
    void check_qr_decomposition(void);
    void check_variable_sparse_matrix(void);
    void check_triangular_matrix(void);
    void check_complex(void);
    void check_eigen_values_and_vectors(void);
    void calc(void);
};

using namespace Tester;

template<typename T>
void CheckBaseMatrix<T>::calc(void) {

    check_matrix_vector_creation();

    check_matrix_swap();

    check_matrix_multiply();

    check_lu_decomposition();

    check_gmres_k_and_inverse();

    check_matrix_transpose_multiply();

    check_determinant_and_trace();

    check_determinant_and_trace();

    check_diag_matrix();

    check_sparse_matrix();

    check_matrix_cocatenation();

    check_cholesky_decomposition();

    check_qr_decomposition();

    check_variable_sparse_matrix();

    check_triangular_matrix();

    check_complex();

    check_eigen_values_and_vectors();
}

template <typename T>
void CheckBaseMatrix<T>::check_matrix_vector_creation(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 行列の作成 */
    Matrix<T, 2, 3> A;
    A(0, 0) = 1.0F; A(0, 1) = 2.0F; A(0, 2) = 3.0F;
    A(1, 0) = 4.0F; A(1, 1) = 5.0F; A(1, 2) = 6.0F;

    /* ベクトルの作成 */
    Vector<T, 3> one_vec = Vector<T, 3>::Ones();

    for (size_t i = 0; i < one_vec.size(); ++i) {
        tester.expect_near(one_vec[i], static_cast<T>(1), NEAR_LIMIT_STRICT,
            "check Vector Ones.");
    }

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    Vector<T, 2> c;
    c[0] = 4.0F;
    c[1] = 5.0F;

    Vector<T, 3> b_add_b = b + b;

    Vector<T, 3> b_add_b_answer({ 2, 4, 6 });

    tester.expect_near(b_add_b.data, b_add_b_answer.data, NEAR_LIMIT_STRICT,
        "check Vector add.");

    Matrix<T, 2, 3> A_add_A = A + A;

    Matrix<T, 2, 3> A_add_A_answer({
        {2, 4, 6},
        {8, 10, 12}
        });

    tester.expect_near(A_add_A.data, A_add_A_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix add.");

    Vector<T, 3> r = b * static_cast<T>(3.0F);
    //std::cout << "scalar calc ";
    //for (size_t i = 0; i < r.size(); ++i) {
    //    std::cout << r[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> r_answer({ 3, 6, 9 });
    tester.expect_near(r.data, r_answer.data, NEAR_LIMIT_STRICT,
        "check Vector multiply.");

    T b_dot = b.dot(b);
    //std::cout << "b dot b = ";
    //std::cout << b_dot << " ";
    //std::cout << std::endl;
    //std::cout << std::endl;

    T b_dot_answer = 14.0F;
    tester.expect_near(b_dot, b_dot_answer, NEAR_LIMIT_STRICT, "check Vector dot.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_matrix_swap(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<T, 2, 3> A;
    A(0, 0) = 1.0F; A(0, 1) = 2.0F; A(0, 2) = 3.0F;
    A(1, 0) = 4.0F; A(1, 1) = 5.0F; A(1, 2) = 6.0F;

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    /* スワップ */
    Matrix<T, 3, 3>Test_swap({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} });
    matrix_row_swap(0, 2, Test_swap);
    //std::cout << "Test_swap = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << Test_swap(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3>Test_swap_answer({ {3, 2, 1}, {6, 5, 4}, {9, 8, 7} });
    tester.expect_near(Test_swap.data, Test_swap_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix row swap.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_matrix_multiply(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<T, 2, 3> A;
    A(0, 0) = 1.0F; A(0, 1) = 2.0F; A(0, 2) = 3.0F;
    A(1, 0) = 4.0F; A(1, 1) = 5.0F; A(1, 2) = 6.0F;

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    Vector<T, 2> c;
    c[0] = 4.0F;
    c[1] = 5.0F;

    /* 行列とベクトルの積 */
    Vector<T, 2> x = A * b;
    //std::cout << "A * b = ";
    //for (size_t i = 0; i < x.size(); ++i) {
    //    std::cout << x[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 2> x_answer({ 14, 32 });
    tester.expect_near(x.data, x_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply Vector.");

    /* ベクトルと行列の積 */
    ColVector<T, 2> c_col(c);
    ColVector<T, 3> y = c_col * A;
    //std::cout << "c^T * A = ";
    //for (size_t i = 0; i < y.size(); ++i) {
    //    std::cout << y[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    ColVector<T, 3> y_answer({ 24, 33, 42 });
    tester.expect_near(y.data, y_answer.data, NEAR_LIMIT_STRICT,
        "check Column Vector multiply Matrix.");

    /* ベクトルと行列の積 2 */
    Vector<T, 2> d;
    d[0] = 1.0F; d[1] = 2.0F;
    Matrix<T, 1, 3> BB;
    BB(0, 0) = 3.0F; BB(0, 1) = 4.0F; BB(0, 2) = 5.0F;
    Matrix<T, 2, 3> Y = d * BB;
    //std::cout << "Y = " << std::endl;
    //for (size_t j = 0; j < Y.cols(); ++j) {
    //    for (size_t i = 0; i < Y.rows(); ++i) {
    //        std::cout << Y(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 2, 3> Y_answer({ {3, 4, 5}, { 6, 8, 10 } });
    tester.expect_near(Y.data, Y_answer.data, NEAR_LIMIT_STRICT,
        "check Row Vector multiply Matrix.");

    /* 行列と行列の積 */
    Matrix<T, 2, 3> E;
    E(0, 0) = 1.0F; E(0, 1) = 2.0F; E(0, 2) = 3.0F;
    E(1, 0) = 4.0F; E(1, 1) = 5.0F; E(1, 2) = 6.0F;

    Matrix<T, 3, 2> F;
    F(0, 0) = 7.0F; F(1, 0) = 8.0F; F(2, 0) = 9.0F;
    F(0, 1) = 10.0F; F(1, 1) = 11.0F; F(2, 1) = 12.0F;

    Matrix<T, 3, 3> G = F * E;
    //std::cout << "F * E = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << G(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> G_answer({
        {47, 64, 81},
        {52, 71, 90},
        {57, 78, 99}
        });
    tester.expect_near(G.data, G_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply Matrix.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_lu_decomposition(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    /* LU分解 */
    Matrix<T, 3, 3> H({ {1, 2, 9}, {8, 5, 4}, {6, 3, 7} });

    Matrix<T, 3, 3> H_lu({ {0, 2, 9}, {8, 5, 4}, {6, 3, 7} });

    LUDecomposition<T, 3> lu(H, static_cast<T>(1.0e-10F));
    Matrix<T, 3, 3> L = lu.get_L();
    Vector<T, 3> xx = lu.solve(b);

    //std::cout << "L = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << L(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> L_answer({
        { 1, 0, 0 },
        { 8, 1, 0 },
        { 6, 0.818182F, 1 }
        });
    tester.expect_near(L.data, L_answer.data, NEAR_LIMIT_STRICT,
        "check LU Decomposition L.");

    //std::cout << "A^-1 * b = ";
    //for (size_t i = 0; i < xx.size(); ++i) {
    //    std::cout << xx[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> xx_answer({ 0.652632F, -0.821053F, 0.221053F });
    tester.expect_near(xx.data, xx_answer.data, NEAR_LIMIT_STRICT,
        "check LU Decomposition solve.");

    T det = lu.get_determinant();
    //std::cout << "det = " << det << std::endl;
    //std::cout << std::endl;

    T det_answer = -95.0F;
    tester.expect_near(det, det_answer, NEAR_LIMIT_STRICT,
        "check Matrix determinant.");

    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_gmres_k_and_inverse(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    Matrix<T, 3, 3> H({ {1, 2, 9}, {8, 5, 4}, {6, 3, 7} });

    /* GMRES k */
    T rho;
    size_t rep_num;
    Vector<T, 3> x_gmres_k_0;
    Vector<T, 3> x_gmres_k = gmres_k(H, b, x_gmres_k_0, static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho, rep_num);

    //std::cout << "x_gmres_k = ";
    //for (size_t i = 0; i < x_gmres_k.size(); ++i) {
    //    std::cout << x_gmres_k[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> x_gmres_k_answer({ 0.652632F, -0.821053F, 0.221053F });
    tester.expect_near(x_gmres_k.data, x_gmres_k_answer.data, NEAR_LIMIT_STRICT,
        "check GMRES k.");

    /* GMRES k rect */
    Vector<T, 4> b_2;
    b_2[0] = 1.0F;
    b_2[1] = 2.0F;
    b_2[2] = 3.0F;
    b_2[3] = 4.0F;
    Matrix<T, 4, 3> H_2;
    H_2(0, 0) = 1.0F; H_2(0, 1) = 2.0F; H_2(0, 2) = 9.0F;
    H_2(1, 0) = 8.0F; H_2(1, 1) = 5.0F; H_2(1, 2) = 4.0F;
    H_2(2, 0) = 6.0F; H_2(2, 1) = 3.0F; H_2(2, 2) = 7.0F;
    H_2(3, 0) = 10.0F; H_2(3, 1) = 12.0F; H_2(3, 2) = 11.0F;

    Vector<T, 3> x_gmres_k_rect_0;
    Vector<T, 3> x_gmres_k_rect = gmres_k_rect(H_2, b_2, x_gmres_k_rect_0, static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho, rep_num);

    //std::cout << "x_gmres_k_rect = ";
    //for (size_t i = 0; i < x_gmres_k_rect.size(); ++i) {
    //    std::cout << x_gmres_k_rect[i] << " ";
    //}
    //std::cout << std::endl;

    Vector<T, 3> x_gmres_k_rect_answer({ 0.271401F, -0.0229407F, 0.127345F });
    tester.expect_near(x_gmres_k_rect.data, x_gmres_k_rect_answer.data, NEAR_LIMIT_STRICT,
        "check GMRES k rect.");

    /* 逆行列 */
    Matrix<T, 3, 3> H_inv;

    H_inv = H.inv();

    Matrix<T, 3, 3> H_inv_answer({
        {-0.242105F, -0.136842F, 0.389474F},
        {0.336842F, 0.494737F, -0.715789F},
        {0.0631579F, -0.0947368F, 0.115789F}
        });
    tester.expect_near(H_inv.data, H_inv_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix inverse.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_matrix_transpose_multiply(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    Matrix<T, 2, 3> E;
    E(0, 0) = 1.0F; E(0, 1) = 2.0F; E(0, 2) = 3.0F;
    E(1, 0) = 4.0F; E(1, 1) = 5.0F; E(1, 2) = 6.0F;

    Matrix<T, 3, 3> H({ {1, 2, 9}, {8, 5, 4}, {6, 3, 7} });

    /* 転置積 */
    Matrix<T, 3, 2> Trans = matrix_multiply_A_mul_BTranspose(H, E);
    //std::cout << "Trans = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 2; ++i) {
    //        std::cout << Trans(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 2> Trans_answer({
        {32, 68},
        {30, 81},
        {33, 81}
        });
    tester.expect_near(Trans.data, Trans_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply Matrix transpose.");

    Vector<T, 3> b_T = matrix_multiply_AT_mul_b(H, b);

    //std::cout << "b_T = ";
    //for (size_t i = 0; i < xx.size(); ++i) {
    //    std::cout << b_T[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> b_T_answer({ 35, 21, 38 });
    tester.expect_near(b_T.data, b_T_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix transpose multiply Vector.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_determinant_and_trace(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<T, 3, 3> H({ {1, 2, 9}, {8, 5, 4}, {6, 3, 7} });

    /* 行列式、トレース */
    T trace = H.get_trace();
    //std::cout << "trace = " << trace << std::endl;
    //std::cout << std::endl;

    T trace_answer = 13.0F;
    tester.expect_near(trace, trace_answer, NEAR_LIMIT_STRICT,
        "check Matrix trace.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_diag_matrix(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    Matrix<T, 3, 3> H({ {1, 2, 9}, {8, 5, 4}, {6, 3, 7} });

    /* 対角行列 */
    DiagMatrix<T, 3> D;
    D[0] = 1.0F;
    D[1] = 2.0F;
    D[2] = 3.0F;

    DiagMatrix<T, 3> D_mul_D = D * D;

    DiagMatrix<T, 3> D_mul_D_answer({ 1, 4, 9 });

    tester.expect_near(D_mul_D.data, D_mul_D_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply DiagMatrix.");

    Vector<T, 3> D_d = D * b;

    //std::cout << "D_d = ";
    //for (size_t i = 0; i < D_d.size(); ++i) {
    //    std::cout << D_d[i] << " ";
    //}
    //std::cout << std::endl;

    Vector<T, 3> D_d_answer({ 1, 4, 9 });
    tester.expect_near(D_d.data, D_d_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply Vector.");

    Matrix<T, 3, 3> D_2 = D * H;

    //std::cout << "D_2 = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << D_2(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> D_2_answer({
        {1, 2, 9},
        {16, 10, 8},
        {18, 9, 21}
        });
    tester.expect_near(D_2.data, D_2_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply Matrix.");

    DiagMatrix<T, 3> D_3({ 4, 5, 6 });
    DiagMatrix<T, 3> D_4 = D - D_3;

    //std::cout << "D_4 = ";
    //for (size_t i = 0; i < D_4.rows(); ++i) {
    //    std::cout << D_4[i] << " ";
    //}
    //std::cout << std::endl;

    DiagMatrix<T, 3> D_4_answer({ -3, -3, -3 });
    tester.expect_near(D_4.data, D_4_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix subtract DiagMatrix.");

    T ddd = output_trace(D);
    //std::cout << "trace " << ddd << std::endl;

    T ddd_answer = 6.0F;
    tester.expect_near(ddd, ddd_answer, NEAR_LIMIT_STRICT,
        "check DiagMatrix trace.");

    Matrix<T, 3, 3> D_dense = output_dense_matrix(D);

    Matrix<T, 3, 3> D_dense_answer({
    {1, 0, 0},
    {0, 2, 0},
    {0, 0, 3}
        });

    tester.expect_near(D_dense.data, D_dense_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix create dense.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_sparse_matrix(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    /* スパース行列 */
    std::vector<T> A_value({ 1.0F, 3.0F, 8.0F, 2.0F, 4.0F });
    std::vector<size_t> A_row_indices({ 0, 0, 2, 1, 2 });
    std::vector<size_t> A_row_pointers({ 0, 1, 3, 5 });

    Matrix<T, 3, 3> DenseB({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });
    SparseMatrix<T, 3, 3, 5> SA(A_value, A_row_indices, A_row_pointers);

    Matrix<T, 3, 3> DenseA({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });

    SparseMatrix<T, 3, 3, 5> SparseCn({ 1.0F, 3.0F, 8.0F, 2.0F, 4.0F },
        { 0, 0, 2, 1, 2 },
        { 0, 1, 3, 5 });

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> SparseCc({ 1.0F, 3.0F, 8.0F, 2.0F, 4.0F });

    auto Cc_mul_A = output_matrix_transpose(SparseCc);
    auto Cc_mul_A_dense = output_dense_matrix(Cc_mul_A);
    Matrix<T, 3, 3> Cn_mul_A = SparseCn.transpose();

    tester.expect_near(Cc_mul_A_dense.data, Cn_mul_A.data, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix transpose.");

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> SparseCc_mul_scalar = SparseCc * static_cast<T>(3);
    Matrix<T, 3, 3> SparseCc_mul_scalar_dense = Base::Matrix::output_dense_matrix(SparseCc_mul_scalar);

    Matrix<T, 3, 3> SparseCc_mul_scalar_answer({
        {3, 0, 0},
        {9, 0, 24},
        {0, 6, 12}
        });

    tester.expect_near(SparseCc_mul_scalar_dense.data, SparseCc_mul_scalar_answer.data, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix multiply scalar.");

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> Scalar_mul_SparseCc = static_cast<T>(3) * SparseCc;
    Matrix<T, 3, 3> Scalar_mul_SparseCc_dense = Base::Matrix::output_dense_matrix(Scalar_mul_SparseCc);

    tester.expect_near(Scalar_mul_SparseCc_dense.data, SparseCc_mul_scalar_answer.data, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix multiply scalar.");

    Vector<T, 3> SparseCc_mul_vector = SparseCc * b;
    Vector<T, 3> SparseCc_mul_vector_answer({ 1, 27, 16 });

    tester.expect_near(SparseCc_mul_vector.data, SparseCc_mul_vector_answer.data, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix multiply Vector.");

    Matrix<T, 3, 3> C_Add_A = SparseCc + DenseA;

    Matrix<T, 3, 3> C_Add_A_answer({
        {2, 2, 3},
        {8, 4, 14},
        {9, 10, 11}
        });

    tester.expect_near(C_Add_A.data, C_Add_A_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add Matrix.");

    Matrix<T, 3, 3> A_Add_C = DenseA + SparseCc;

    tester.expect_near(A_Add_C.data, C_Add_A_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix add SparseMatrix.");

    Matrix<T, 3, 3> Sparse_mul_Dense = SparseCc * DenseB;

    //std::cout << "DenseC = " << std::endl;
    //for (size_t j = 0; j < DenseC.cols(); ++j) {
    //    for (size_t i = 0; i < DenseC.rows(); ++i) {
    //        std::cout << DenseC(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> Sparse_mul_Dense_answer({
        {1, 2, 3},
        {75, 70, 65},
        {46, 40, 40}
        });

    tester.expect_near(Sparse_mul_Dense.data, Sparse_mul_Dense_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply Matrix.");

    Matrix<T, 3, 3> Dense_mul_Sparse = DenseB * SparseCc;

    Matrix<T, 3, 3> Dense_mul_Sparse_answer({
        {7, 6, 28},
        {17, 12, 56},
        {33, 14, 92}
        });

    tester.expect_near(Dense_mul_Sparse.data, Dense_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply SparseMatrix.");

    Matrix<T, 5, 3> DenseD({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7},{2, 2, 3}, {1, 9, 8} });
    SparseMatrix<T, 3, 4, 6> SE({ 1, 3, 8, 1, 2, 4 }, { 0, 0, 2, 3, 1, 2 }, { 0, 1, 4, 6 });

    Matrix<T, 5, 4> DenseF = DenseD * SE;

    //std::cout << "DenseF = " << std::endl;
    //for (size_t j = 0; j < DenseF.cols(); ++j) {
    //    for (size_t i = 0; i < DenseF.rows(); ++i) {
    //        std::cout << DenseF(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 5, 4> DenseF_answer({
        {7, 6, 28, 2},
        {17, 12, 56, 4 },
        {33, 14, 92, 8},
        {8, 6, 28, 2},
        {28, 16, 104, 9}
        });
    tester.expect_near(DenseF.data, DenseF_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply SparseMatrix.");

    Matrix<T, 3, 3> DenseG({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7} });

    Matrix<T, 3, 3> Sparse_sub_Dense = SparseCc - DenseG;
    //Matrix<T, 3, 3> DenseH = DenseG - SA;

    //std::cout << "DenseH = " << std::endl;
    //for (size_t j = 0; j < DenseH.cols(); ++j) {
    //    for (size_t i = 0; i < DenseH.rows(); ++i) {
    //        std::cout << DenseH(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> Sparse_sub_Dense_answer({
        {0, -2, -3},
        {-2, -4, 2},
        {-9, -6, -3}
        });
    tester.expect_near(Sparse_sub_Dense.data, Sparse_sub_Dense_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix subtract Matrix.");

    Matrix<T, 3, 3> Dense_sub_Sparse = DenseG - SparseCc;

    Matrix<T, 3, 3> Dense_sub_Sparse_answer({
        {0, 2, 3},
        {2, 4, -2},
        {9, 6, 3}
        });

    tester.expect_near(Dense_sub_Sparse.data, Dense_sub_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix subtract SparseMatrix.");

    CompiledSparseMatrix<T, 3, 4,
        RowIndices<0, 0, 2, 3, 1, 2>,
        RowPointers<0, 1, 4, 6>> SEc({ 1, 3, 8, 1, 2, 4 });

    auto Sparse_mul_Sparse = SparseCc * SEc;
    Matrix<T, 3, 4> Sparse_mul_Sparse_dense = Base::Matrix::output_dense_matrix(Sparse_mul_Sparse);

    Matrix<T, 3, 4> Sparse_mul_Sparse_answer({
        {1, 0, 0, 0},
        {3, 16, 32, 0},
        {6, 8, 32, 2}
        });

    tester.expect_near(Sparse_mul_Sparse_dense.data, Sparse_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply SparseMatrix.");

    auto SparseTranspose_mul_Sparse =
        matrix_multiply_SparseATranspose_mul_SparseB(SparseCc, SEc);
    auto SparseTranspose_mul_Sparse_dense =
        Base::Matrix::output_dense_matrix(SparseTranspose_mul_Sparse);

    Matrix<T, 3, 4> SparseTranspose_mul_Sparse_answer({
        {10, 0, 24, 3},
        {0, 4, 8, 0},
        {24, 8, 80, 8}
        });

    tester.expect_near(SparseTranspose_mul_Sparse_dense.data,
        SparseTranspose_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix transpose multiply SparseMatrix.");

    auto Sparse_mul_SparseTranspose =
        matrix_multiply_SparseA_mul_SparseBTranspose(SparseCc, SparseCc);
    Matrix<T, 3, 3> Sparse_mul_SparseTranspose_dense =
        Base::Matrix::output_dense_matrix(Sparse_mul_SparseTranspose);

    Matrix<T, 3, 3> Sparse_mul_SparseTranspose_answer({
        {1, 3, 0},
        {3, 73, 32},
        {0, 32, 20}
        });

    tester.expect_near(Sparse_mul_SparseTranspose_dense.data,
        Sparse_mul_SparseTranspose_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply SparseMatrix transpose.");

    DiagMatrix<T, 3> DiagJ({ 10, 20, 30 });

    auto Sparse_add_Diag = SparseCc + DiagJ;
    Matrix<T, 3, 3> Sparse_add_Diag_dense = Base::Matrix::output_dense_matrix(Sparse_add_Diag);

    Matrix<T, 3, 3> Sparse_add_Diag_answer({
        {11, 0, 0},
        {3, 20, 8},
        {0, 2, 34}
        });

    tester.expect_near(Sparse_add_Diag_dense.data, Sparse_add_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add DiagMatrix.");

    auto Diag_add_Sparse = DiagJ + SparseCc;
    Matrix<T, 3, 3> Diag_add_Sparse_dense = Base::Matrix::output_dense_matrix(Diag_add_Sparse);

    tester.expect_near(Diag_add_Sparse_dense.data, Sparse_add_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix add SparseMatrix.");

    auto Sparse_sub_Diag = SparseCc - DiagJ;
    Matrix<T, 3, 3> Sparse_sub_Diag_dense = Base::Matrix::output_dense_matrix(Sparse_sub_Diag);

    Matrix<T, 3, 3> Sparse_sub_Diag_answer({
        {-9, 0, 0},
        {3, -20, 8},
        {0, 2, -26}
        });

    tester.expect_near(Sparse_sub_Diag_dense.data, Sparse_sub_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub DiagMatrix.");

    auto Diag_sub_Sparse = DiagJ - SparseCc;
    Matrix<T, 3, 3> Diag_sub_Sparse_dense = Base::Matrix::output_dense_matrix(Diag_sub_Sparse);

    Matrix<T, 3, 3> Diag_sub_Sparse_answer({
        {9, 0, 0},
        {-3, 20, -8},
        {0, -2, 26}
        });

    tester.expect_near(Diag_sub_Sparse_dense.data, Diag_sub_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix sub SparseMatrix.");

    auto Sparse_mul_Diag = SparseCc * DiagJ;
    Matrix<T, 3, 3> Sparse_mul_Diag_dense = Base::Matrix::output_dense_matrix(Sparse_mul_Diag);

    Matrix<T, 3, 3> Sparse_mul_Diag_answer({
        {10, 0, 0},
        {30, 0, 240},
        {0, 40, 120}
        });

    tester.expect_near(Sparse_mul_Diag_dense.data, Sparse_mul_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply DiagMatrix.");

    auto Diag_mul_Sparse = DiagJ * SparseCc;
    Matrix<T, 3, 3> Diag_mul_Sparse_dense = Base::Matrix::output_dense_matrix(Diag_mul_Sparse);

    Matrix<T, 3, 3> Diag_mul_Sparse_answer({
        {10, 0, 0},
        {60, 0, 160},
        {0, 60, 120}
        });

    tester.expect_near(Diag_mul_Sparse_dense.data, Diag_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply SparseMatrix.");

    auto Sparse_add_Sparse = SparseCc + SparseCc;
    auto Sparse_add_Sparse_dense = Base::Matrix::output_dense_matrix(Sparse_add_Sparse);

    Matrix<T, 3, 3> Sparse_add_Sparse_answer({
        {2, 0, 0},
        {6, 0, 16},
        {0, 4, 8}
        });

    tester.expect_near(Sparse_add_Sparse_dense.data, Sparse_add_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add SparseMatrix.");

    auto Sparse_sub_Sparse = SparseCc - SparseCc;
    auto Sparse_sub_Sparse_dense = Base::Matrix::output_dense_matrix(Sparse_sub_Sparse);

    Matrix<T, 3, 3> Sparse_sub_Sparse_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(Sparse_sub_Sparse_dense.data, Sparse_sub_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub SparseMatrix.");

    auto Transpose_Diag_mul_Sparse = matrix_multiply_Transpose_DiagA_mul_SparseB(DiagJ, SparseCc);
    Matrix<T, 3, 3> Transpose_Diag_mul_Sparse_dense = Base::Matrix::output_dense_matrix(Transpose_Diag_mul_Sparse);

    Matrix<T, 3, 3> Transpose_Diag_mul_Sparse_answer({
        {10, 60, 0},
        {0, 0, 60},
        {0, 160, 120}
        });

    tester.expect_near(Transpose_Diag_mul_Sparse_dense.data,
        Transpose_Diag_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check Transpose DiagMatrix multiply SparseMatrix.");

    Matrix<T, 3, 3> Sparse_mul_Dense_T = matrix_multiply_SparseA_mul_BTranspose(SparseCc, DenseG);

    Matrix<T, 3, 3> Sparse_mul_Dense_T_answer({
        {1, 5, 9},
        {27, 63, 83},
        {16, 32, 44}
        });

    tester.expect_near(Sparse_mul_Dense_T.data, Sparse_mul_Dense_T_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply Matrix transpose.");

    Matrix<T, 3, 3> Dense_mul_SparseT = matrix_multiply_A_mul_SparseBTranspose(DenseG, SparseCc);

    //std::cout << "DenseL = " << std::endl;
    //for (size_t j = 0; j < DenseL.cols(); ++j) {
    //    for (size_t i = 0; i < DenseL.rows(); ++i) {
    //        std::cout << DenseL(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> Dense_mul_SparseT_answer({
        {1, 27, 16},
        {5, 63, 32},
        {9, 83, 44}
        });
    tester.expect_near(Dense_mul_SparseT.data, Dense_mul_SparseT_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply SparseMatrix transpose.");

    Matrix<T, 3, 3> DenseT_mul_Sparse = matrix_multiply_ATranspose_mul_SparseB(DenseG, SparseCc);

    //std::cout << "DenseM = " << std::endl;
    //for (size_t j = 0; j < DenseM.cols(); ++j) {
    //    for (size_t i = 0; i < DenseM.rows(); ++i) {
    //        std::cout << DenseM(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> DenseT_mul_Sparse_answer({
        {16, 18, 76},
        {14, 16, 64},
        {21, 14, 76}
        });
    tester.expect_near(DenseT_mul_Sparse.data, DenseT_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix transpose multiply SparseMatrix.");

    Matrix<T, 3, 3> SparseT_mul_Dense = matrix_multiply_SparseAT_mul_B(SparseCc, DenseG);

    Matrix<T, 3, 3> SparseT_mul_Dense_answer({
        {16, 14, 21},
        {18, 16, 14},
        {76, 64, 76}
        });

    tester.expect_near(SparseT_mul_Dense.data, SparseT_mul_Dense_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix transpose multiply Matrix.");

    Vector<T, 3> Dense_n = SA * b;

    //std::cout << "SA * b = ";
    //for (size_t i = 0; i < Dense_n.size(); ++i) {
    //    std::cout << Dense_n[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> Dense_n_answer({ 1, 27, 16 });
    tester.expect_near(Dense_n.data, Dense_n_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply Vector.");

    ColVector<T, 3> b_col(b);
    ColVector<T, 3> Dense_m = colVector_a_mul_SparseB(b_col, SparseCc);

    //std::cout << "b_col * SA = ";
    //for (size_t i = 0; i < Dense_m.size(); ++i) {
    //    std::cout << Dense_m[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    ColVector<T, 3> Dense_m_answer({ 7, 6, 28 });
    tester.expect_near(Dense_m.data, Dense_m_answer.data, NEAR_LIMIT_STRICT,
        "check Column Vector multiply SparseMatrix.");

    auto Dense_to_sparse = create_compiled_sparse(DenseG);

    Matrix<T, 3, 3> A_mul_A_sparse = DenseG * Dense_to_sparse;

    Matrix<T, 3, 3> A_mul_A_sparse_answer({
        {38, 34, 36},
        {79, 74, 81},
        {112, 106, 124}
        });

    tester.expect_near(A_mul_A_sparse.data, A_mul_A_sparse_answer.data, NEAR_LIMIT_STRICT,
        "check DenseMatrix to CompiledSparseMatrix.");

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> SparseCc_set = SparseCc;

    set_sparse_matrix_value<2, 0>(SparseCc_set, static_cast<T>(100));
    set_sparse_matrix_value<2, 2>(SparseCc_set, static_cast<T>(100));

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> SparseCc_set_answer({ 1, 3, 8, 2, 100 });

    tester.expect_near(SparseCc_set.values, SparseCc_set_answer.values, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix set value.");

    T SparseCc_value = get_sparse_matrix_value<1, 0>(SparseCc_set);

    T SparseCc_value_answer = static_cast <T>(3.0F);

    tester.expect_near(SparseCc_value, SparseCc_value_answer, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix get value.");

    auto Diag_to_Sparse = create_compiled_sparse(DiagJ);

    Matrix<T, 3, 3> A_mul_Diag_sparse = DenseG * Diag_to_Sparse;

    Matrix<T, 3, 3> A_mul_Diag_sparse_answer({
        {10, 40, 90},
        {50, 80, 180},
        {90, 160, 210}
        });

    tester.expect_near(A_mul_Diag_sparse.data, A_mul_Diag_sparse_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix to CompiledSparseMatrix.");

    auto C_from_list = create_compiled_sparse<T,
        SparseAvailable<
        ColumnAvailable<true, false, false>,
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>>
        >({ 1, 3, 8, 2, 4 });

    Matrix<T, 3, 3> A_mul_C_from_list = DenseG * C_from_list;

    Matrix<T, 3, 3> A_mul_C_from_list_answer({
        {7, 6, 28},
        {17, 12, 56},
        {33, 14, 92}
        });

    tester.expect_near(A_mul_C_from_list.data, A_mul_C_from_list_answer.data, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix from list.");

    /* スパース行列の逆行列計算 */
    T rho;
    size_t rep_num;

    std::array<T, 3> rho_vec;
    std::array<std::size_t, 3> rep_num_vec;

    Vector<T, 3> x_gmres_k_0;
    Vector<T, 3> x_gmres_k = sparse_gmres_k(SparseCc, b, x_gmres_k_0,
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho, rep_num);

    //std::cout << "x_gmres_k = ";
    //for (size_t i = 0; i < x_gmres_k.size(); ++i) {
    //    std::cout << x_gmres_k[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> x_gmres_k_answer_2({ 1, 1.75, -0.125 });
    tester.expect_near(x_gmres_k.data, x_gmres_k_answer_2.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix GMRES k.");

    CompiledSparseMatrix<T, 4, 3,
        RowIndices<0, 1, 2, 1, 2, 1>,
        RowPointers<0, 2, 3, 5, 6>> SBc({ 1, 3, 2, 8, 4, 1 });

    Vector<T, 4> b_2;
    b_2[0] = 1.0F;
    b_2[1] = 2.0F;
    b_2[2] = 3.0F;
    b_2[3] = 4.0F;
    Matrix<T, 4, 3> H_2;
    H_2(0, 0) = 1.0F; H_2(0, 1) = 2.0F; H_2(0, 2) = 9.0F;
    H_2(1, 0) = 8.0F; H_2(1, 1) = 5.0F; H_2(1, 2) = 4.0F;
    H_2(2, 0) = 6.0F; H_2(2, 1) = 3.0F; H_2(2, 2) = 7.0F;
    H_2(3, 0) = 10.0F; H_2(3, 1) = 12.0F; H_2(3, 2) = 11.0F;

    Vector<T, 3> x_gmres_k_rect_0;
    x_gmres_k = sparse_gmres_k_rect(SBc, b_2, x_gmres_k_rect_0,
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho, rep_num);

    //std::cout << "x_gmres_k = ";
    //for (size_t i = 0; i < x_gmres_k.size(); ++i) {
    //    std::cout << x_gmres_k[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> x_gmres_k_answer_3({ 0.478261F, 0.173913F, 0.521739F });
    tester.expect_near(x_gmres_k.data, x_gmres_k_answer_3.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix GMRES k rect.");

    Matrix<T, 4, 3> SB_dense = Base::Matrix::output_dense_matrix(SBc);

    //std::cout << "SB_dense = " << std::endl;
    //for (size_t j = 0; j < SB_dense.cols(); ++j) {
    //    for (size_t i = 0; i < SB_dense.rows(); ++i) {
    //        std::cout << SB_dense(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 4, 3> SB_dense_answer({
        {1, 3, 0},
        {0, 0, 2},
        {0, 8, 4},
        {0, 1, 0}
        });
    tester.expect_near(SB_dense.data, SB_dense_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix create dense.");

    Matrix<T, 3, 3> X_temp = Matrix<T, 3, 3>::identity();
    Matrix<T, 3, 3> S_inv = sparse_gmres_k_matrix_inv(SparseCc,
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho_vec, rep_num_vec, X_temp);

    //std::cout << "S_inv = " << std::endl;
    //for (size_t j = 0; j < S_inv.cols(); ++j) {
    //    for (size_t i = 0; i < S_inv.rows(); ++i) {
    //        std::cout << S_inv(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> S_inv_answer({
        {1.0F, 0.0F, 0.0F},
        {0.75F, -0.25F, 0.5F},
        {-0.375F, 0.125F, 0.0F}
        });
    tester.expect_near(S_inv.data, S_inv_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix inverse.");


    /* スパース行列を作成 */
    SparseMatrix<T, 3, 3, 5> S_test(Matrix<T, 3, 3>({ {1, 0, 0}, {3, 0, 8}, {0 ,2, 4} }));

    Matrix<T, 3, 3> Dense_T = S_test * DenseB;

    //std::cout << "Dense_T = " << std::endl;
    //for (size_t j = 0; j < Dense_T.cols(); ++j) {
    //    for (size_t i = 0; i < Dense_T.rows(); ++i) {
    //        std::cout << Dense_T(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> Dense_T_answer({
        {1, 2, 3},
        {75, 70, 65},
        {46, 40, 40}
        });
    tester.expect_near(Dense_T.data, Dense_T_answer.data, NEAR_LIMIT_STRICT,
        "create Sparse and check SparseMatrix multiply Matrix.");

    DiagMatrix<T, 3> D;
    D[0] = 1.0F;
    D[1] = 2.0F;
    D[2] = 3.0F;

    SparseMatrix<T, 3, 3, 3> G_s = create_sparse(D);
    Matrix<T, 3, 3> G_s_d = G_s.create_dense();
    //std::cout << "G_s_d = " << std::endl;
    //for (size_t j = 0; j < G_s_d.cols(); ++j) {
    //    for (size_t i = 0; i < G_s_d.rows(); ++i) {
    //        std::cout << G_s_d(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> G_s_d_answer({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
        });
    tester.expect_near(G_s_d.data, G_s_d_answer.data, NEAR_LIMIT_STRICT,
        "check create_sparse command.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_matrix_cocatenation(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    Matrix<T, 3, 3> DenseA({ { 1, 2, 3 }, {5, 4, 6}, {9, 8, 7} });

    DiagMatrix<T, 3> D;
    D[0] = 1.0F;
    D[1] = 2.0F;
    D[2] = 3.0F;

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> SparseCc({ 1.0F, 3.0F, 8.0F, 2.0F, 4.0F });

    /* 行列結合 */
    auto A_v_A = concatenate_vertically(DenseA, DenseA);

    Matrix<T, 6, 3> A_v_A_answer({
    {1, 2, 3},
    {5, 4, 6},
    {9, 8, 7},
    {1, 2, 3},
    {5, 4, 6},
    {9, 8, 7}
        });

    tester.expect_near(A_v_A.data, A_v_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Dense.");

    auto A_v_B = concatenate_vertically(DenseA, D);
    auto A_v_B_dense = Base::Matrix::output_dense_matrix(A_v_B);

    Matrix<T, 6, 3> A_v_B_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7},
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
        });

    tester.expect_near(A_v_B_dense.data, A_v_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Diag.");

    auto A_v_C = concatenate_vertically(DenseA, SparseCc);
    auto A_v_C_dense = Base::Matrix::output_dense_matrix(A_v_C);

    Matrix<T, 6, 3> A_v_C_answer({
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7},
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4}
        });

    tester.expect_near(A_v_C_dense.data, A_v_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Dense and Sparse.");

    auto B_v_A = concatenate_vertically(D, DenseA);
    auto B_v_A_dense = Base::Matrix::output_dense_matrix(B_v_A);

    Matrix<T, 6, 3> B_v_A_answer({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3},
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    tester.expect_near(B_v_A_dense.data, B_v_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Dense.");

    auto B_v_B = concatenate_vertically(D, D * static_cast<T>(2));
    auto B_v_B_dense = Base::Matrix::output_dense_matrix(B_v_B);

    Matrix<T, 6, 3> B_v_B_answer({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3},
        {2, 0, 0},
        {0, 4, 0},
        {0, 0, 6}
        });

    tester.expect_near(B_v_B_dense.data, B_v_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Diag.");

    auto B_v_C = concatenate_vertically(D, SparseCc);
    auto B_v_C_dense = Base::Matrix::output_dense_matrix(B_v_C);

    Matrix<T, 6, 3> B_v_C_answer({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3},
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4}
        });

    tester.expect_near(B_v_C_dense.data, B_v_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Diag and Sparse.");

    auto C_v_A = concatenate_vertically(SparseCc, DenseA);
    auto C_v_A_dense = Base::Matrix::output_dense_matrix(C_v_A);

    Matrix<T, 6, 3> C_v_A_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4},
        {1, 2, 3},
        {5, 4, 6},
        {9, 8, 7}
        });

    tester.expect_near(C_v_A_dense.data, C_v_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Dense.");

    auto C_v_B = concatenate_vertically(SparseCc, D);
    auto C_v_B_dense = Base::Matrix::output_dense_matrix(C_v_B);

    Matrix<T, 6, 3> C_v_B_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4},
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
        });

    tester.expect_near(C_v_B_dense.data, C_v_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Diag.");

    auto C_v_C = concatenate_vertically(SparseCc, SparseCc * static_cast<T>(2));
    auto C_v_C_dense = Base::Matrix::output_dense_matrix(C_v_C);

    Matrix<T, 6, 3> C_v_C_answer({
        {1, 0, 0},
        {3, 0, 8},
        {0, 2, 4},
        {2, 0, 0},
        {6, 0, 16},
        {0, 4, 8}
        });

    tester.expect_near(C_v_C_dense.data, C_v_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate vertically Sparse and Sparse.");

    auto A_h_A = concatenate_horizontally(DenseA, DenseA);

    Matrix<T, 3, 6> A_h_A_answer({
        {1, 2, 3, 1, 2, 3},
        {5, 4, 6, 5, 4, 6},
        {9, 8, 7, 9, 8, 7}
        });

    tester.expect_near(A_h_A.data, A_h_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Dense.");

    auto A_h_B = concatenate_horizontally(DenseA, D);
    auto A_h_B_dense = Base::Matrix::output_dense_matrix(A_h_B);

    Matrix<T, 3, 6> A_h_B_answer({
        {1, 2, 3, 1, 0, 0},
        {5, 4, 6, 0, 2, 0},
        {9, 8, 7, 0, 0, 3}
        });

    tester.expect_near(A_h_B_dense.data, A_h_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Diag.");

    auto A_h_C = concatenate_horizontally(DenseA, SparseCc);
    auto A_h_C_dense = Base::Matrix::output_dense_matrix(A_h_C);

    Matrix<T, 3, 6> A_h_C_answer({
        {1, 2, 3, 1, 0, 0},
        {5, 4, 6, 3, 0, 8},
        {9, 8, 7, 0, 2, 4}
        });

    tester.expect_near(A_h_C_dense.data, A_h_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Sparse.");

    auto B_h_A = concatenate_horizontally(D, DenseA);
    auto B_h_A_dense = Base::Matrix::output_dense_matrix(B_h_A);

    Matrix<T, 3, 6> B_h_A_answer({
        {1, 0, 0, 1, 2, 3},
        {0, 2, 0, 5, 4, 6},
        {0, 0, 3, 9, 8, 7}
        });

    tester.expect_near(B_h_A_dense.data, B_h_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Dense.");

    auto B_h_B = concatenate_horizontally(D, D * static_cast<T>(2));
    auto B_h_B_dense = Base::Matrix::output_dense_matrix(B_h_B);

    Matrix<T, 3, 6> B_h_B_answer({
        {1, 0, 0, 2, 0, 0},
        {0, 2, 0, 0, 4, 0},
        {0, 0, 3, 0, 0, 6}
        });

    tester.expect_near(B_h_B_dense.data, B_h_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Diag.");

    auto B_h_C = concatenate_horizontally(D, SparseCc);
    auto B_h_C_dense = Base::Matrix::output_dense_matrix(B_h_C);

    Matrix<T, 3, 6> B_h_C_answer({
        {1, 0, 0, 1, 0, 0},
        {0, 2, 0, 3, 0, 8},
        {0, 0, 3, 0, 2, 4}
        });

    tester.expect_near(B_h_C_dense.data, B_h_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Sparse.");

    auto C_h_A = concatenate_horizontally(SparseCc, DenseA);
    auto C_h_A_dense = Base::Matrix::output_dense_matrix(C_h_A);

    Matrix<T, 3, 6> C_h_A_answer({
        {1, 0, 0, 1, 2, 3},
        {3, 0, 8, 5, 4, 6},
        {0, 2, 4, 9, 8, 7}
        });

    tester.expect_near(C_h_A_dense.data, C_h_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Dense.");

    auto C_h_B = concatenate_horizontally(SparseCc, D);
    auto C_h_B_dense = Base::Matrix::output_dense_matrix(C_h_B);

    Matrix<T, 3, 6> C_h_B_answer({
        {1, 0, 0, 1, 0, 0},
        {3, 0, 8, 0, 2, 0},
        {0, 2, 4, 0, 0, 3}
        });

    tester.expect_near(C_h_B_dense.data, C_h_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Diag.");

    auto C_h_C = concatenate_horizontally(SparseCc, SparseCc * static_cast<T>(2));
    auto C_h_C_dense = Base::Matrix::output_dense_matrix(C_h_C);

    Matrix<T, 3, 6> C_h_C_answer({
        {1, 0, 0, 2, 0, 0},
        {3, 0, 8, 6, 0, 16},
        {0, 2, 4, 0, 4, 8}
        });

    tester.expect_near(C_h_C_dense.data, C_h_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Sparse.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_cholesky_decomposition(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    DiagMatrix<T, 3> D;
    D[0] = 1.0F;
    D[1] = 2.0F;
    D[2] = 3.0F;

    /* コレスキー分解 */
    Matrix<T, 3, 3> K({ {10, 1, 2}, {1, 20, 4}, {2, 4, 30} });

    Matrix<T, 3, 3> K_ch;
    bool flag = false;
    K_ch = cholesky_decomposition(K, K_ch, flag);

    //std::cout << "K_ch = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << K_ch(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> K_ch_2 = matrix_multiply_AT_mul_B(K_ch, K_ch);
    //std::cout << "K_ch_2 = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << K_ch_2(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> K_ch_2_answer({
        {10, 1, 2},
        {1, 20, 4},
        {2, 4, 30}
        });
    tester.expect_near(K_ch_2.data, K_ch_2_answer.data, NEAR_LIMIT_STRICT,
        "check Cholesky decomposition.");

    DiagMatrix<T, 3> K_diag = cholesky_decomposition_diag(D, D, flag);
    //std::cout << "K_diag = ";
    //for (size_t i = 0; i < K_diag.rows(); ++i) {
    //    std::cout << K_diag[i] << " ";
    //}
    //std::cout << std::endl;

    DiagMatrix<T, 3> K_diag_answer({ 1, 1.41421F, 1.73205F });
    tester.expect_near(K_diag.data, K_diag_answer.data, NEAR_LIMIT_STRICT,
        "check Cholesky decomposition diag.");

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 1, 2, 1, 2 >,
        RowPointers<0, 1, 3, 5>> K_s({ 1, 8, 3, 3, 4 });
    Matrix<T, 3, 3> K_ch_sparse = cholesky_decomposition_sparse(K_s, K_ch, flag);

    Matrix<T, 3, 3> K_ch_sparse_2 = matrix_multiply_AT_mul_B(K_ch_sparse, K_ch_sparse);

    Matrix<T, 3, 3> K_ch_sparse_2_answer({
        {1, 0, 0},
        {0, 8, 3},
        {0, 3, 4}
        });

    tester.expect_near(K_ch_sparse_2.data, K_ch_sparse_2_answer.data, NEAR_LIMIT_STRICT,
        "check Cholesky decomposition sparse.");

    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_qr_decomposition(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    DiagMatrix<T, 3> DiagJ({ 10, 20, 30 });

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> SparseCc({ 1.0F, 3.0F, 8.0F, 2.0F, 4.0F });

    /* QR分解 */
    Matrix<T, 3, 3> C_dense({ {1, 0, 0}, {3, 0, 8}, {0 ,2, 4} });
    QRDecomposition<T, 3, 3> qr(C_dense, static_cast<T>(1.0e-10F));

    Matrix<T, 3, 3> Q = qr.get_Q();
    Matrix<T, 3, 3> R = qr.get_R();

    //std::cout << "Q = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << Q(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> Q_answer({
        {-0.316228F, 0, 0.948683F},
        {-0.948683F, 0, -0.316228F},
        {0, -1, 0}
        });
    tester.expect_near(Q.data, Q_answer.data, NEAR_LIMIT_STRICT,
        "check QR Decomposition Q.");

    QRDecompositionSparse<T, 3, 3, RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> qr_s(SparseCc, static_cast<T>(1.0e-10F));

    Matrix<T, 3, 3> Q_s = qr_s.get_Q();

    Matrix<T, 3, 3> Q_s_answer({
        {-0.316228F, 0.0F, 0.948683F},
        {-0.948683F, 0.0F, -0.316228F},
        {0.0F, -1.0F, 0.0F}
        });

    tester.expect_near(Q_s.data, Q_s_answer.data, NEAR_LIMIT_STRICT,
        "check QR Decomposition Sparse Q.");

    //std::cout << "R = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << R(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> R_answer({
        {-3.16228F, 0, -7.58947F},
        {0, -2, -4},
        {0, 0, -2.52982F}
        });
    tester.expect_near(R.data, R_answer.data, NEAR_LIMIT_STRICT,
        "check QR Decomposition R.");

    QRDecompositionDiag<T, 3> qr_d(DiagJ, static_cast<T>(1.0e-10F));
    DiagMatrix<T, 3> Q_d = qr_d.get_Q();

    DiagMatrix<T, 3> Q_d_answer({ 1, 1, 1 });
    tester.expect_near(Q_d.data, Q_d_answer.data, NEAR_LIMIT_STRICT,
        "check QR Decomposition Diag Q.");

    DiagMatrix<T, 3> R_d = qr_d.get_R();
    DiagMatrix<T, 3> R_d_answer({ 10, 20, 30 });
    tester.expect_near(R_d.data, R_d_answer.data, NEAR_LIMIT_STRICT,
        "check QR Decomposition Diag R.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_variable_sparse_matrix(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    std::vector<T> A_value({ 1.0F, 3.0F, 8.0F, 2.0F, 4.0F });
    std::vector<size_t> A_row_indices({ 0, 0, 2, 1, 2 });
    std::vector<size_t> A_row_pointers({ 0, 1, 3, 5 });

    /* 可変スパース行列 */
    VariableSparseMatrix<T, 3, 3> CV;
    std::copy(A_value.begin(),
        A_value.end(), CV.values.begin());
    std::copy(A_row_indices.begin(),
        A_row_indices.end(), CV.row_indices.begin());
    std::copy(A_row_pointers.begin(),
        A_row_pointers.end(), CV.row_pointers.begin());

    Matrix<T, 3, 3> VS_test = CV * CV;
    //std::cout << "VS_test = " << std::endl;
    //for (size_t j = 0; j < VS_test.cols(); ++j) {
    //    for (size_t i = 0; i < VS_test.rows(); ++i) {
    //        std::cout << VS_test(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> VS_test_answer({
        {1, 0, 0},
        {3, 16, 32},
        {6, 8, 32}
        });
    tester.expect_near(VS_test.data, VS_test_answer.data, NEAR_LIMIT_STRICT,
        "check VariableSparseMatrix multiply VariableSparseMatrix.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_triangular_matrix(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;


    /* 三角スパース行列 */
    Matrix<T, 3, 3> Dense_33({ {1, 2, 3}, {5, 6, 7}, {9, 10, 11} });
    auto Dense_33_Triangular = TriangularSparse<T, 3, 3>::create_upper(Dense_33);

    set_sparse_matrix_value<1, 1>(Dense_33_Triangular, static_cast<T>(0));
    TriangularSparse<T, 3, 3>::set_values_upper(Dense_33_Triangular, Dense_33);

    Matrix<T, 3, 3> Dense_33_mul_33 = Dense_33_Triangular * Dense_33;

    Matrix<T, 3, 3> Dense_33_mul_33_answer({
        {38, 44, 50},
        {93, 106, 119},
        {99, 110, 121}
        });

    tester.expect_near(Dense_33_mul_33.data, Dense_33_mul_33_answer.data, NEAR_LIMIT_STRICT,
        "check Upper TriangularSparse multiply DenseMatrix.");

    auto TS = TriangularSparse<T, 4, 4>::create_lower();
    Matrix<T, 4, 4> Test_ts({ {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} });
    TriangularSparse<T, 4, 4>::set_values_lower(TS, Test_ts);

    Matrix<T, 4, 4> Test_lower = Base::Matrix::output_dense_matrix(TS);
    //for (size_t j = 0; j < Test_lower.cols(); ++j) {
    //    for (size_t i = 0; i < Test_lower.rows(); ++i) {
    //        std::cout << Test_lower(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 4, 4> Test_lower_answer({
        {1, 0, 0, 0},
        {5, 6, 0, 0},
        {9, 10, 11, 0},
        {13, 14, 15, 16}
        });
    tester.expect_near(Test_lower.data, Test_lower_answer.data, NEAR_LIMIT_STRICT,
        "check TriangularSparse create lower.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_complex(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    T rho;
    size_t rep_num;

    /* 複素数 */
    Complex<T> a_comp(1, 2);
    Complex<T> b_comp(3, 4);
    a_comp *= b_comp;

    /* 複素数 GMRES K */
    Matrix<Complex<T>, 3, 3> H_comp({ {1, 2, 9}, {8, 5, 4}, {6, 3, 7} });
    H_comp(0, 1).imag = 1.0F;
    Vector<Complex<T>, 3> b_comp_2({ 1, 2, 3 });

    Vector<Complex<T>, 3> x_gmres_k_0_comp;
    Vector<Complex<T>, 3> x_gmres_k_comp = complex_gmres_k(H_comp, b_comp_2, x_gmres_k_0_comp,
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho, rep_num);

    //std::cout << "x_gmres_k_comp = ";
    //std::cout << "[";
    //for (size_t i = 0; i < x_gmres_k_comp.size(); ++i) {
    //    std::cout << x_gmres_k_comp[i].real << " + " << x_gmres_k_comp[i].imag << "j, ";
    //}
    //std::cout << "]";
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> x_gmres_k_comp_real = get_real_vector_from_complex_vector(x_gmres_k_comp);
    Vector<T, 3> x_gmres_k_comp_answer_real({ 0.592496765847348F, -0.737386804657180F, 0.236739974126779F });
    tester.expect_near(x_gmres_k_comp_real.data, x_gmres_k_comp_answer_real.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k real.");

    Vector<T, 3> x_gmres_k_comp_imag = get_imag_vector_from_complex_vector(x_gmres_k_comp);
    Vector<T, 3> x_gmres_k_comp_answer_imag({ -0.178525226390686F, 0.248382923673997F, 0.046571798188875F });
    tester.expect_near(x_gmres_k_comp_imag.data, x_gmres_k_comp_answer_imag.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k imag.");


    std::array<T, 3> rho_vec;
    std::array<std::size_t, 3> rep_num_vec;

    Matrix<Complex<T>, 3, 3> H_inv_H;

    complex_gmres_k_matrix(H_comp, H_comp, H_inv_H,
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho_vec, rep_num_vec);

    Matrix<T, 3, 3> H_inv_H_real = get_real_matrix_from_complex_matrix(H_inv_H);
    Matrix<T, 3, 3> H_inv_H_answer_real = {
        {1.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 1.0F}
    };
    tester.expect_near(H_inv_H_real.data, H_inv_H_answer_real.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k matrix real.");

    Matrix<T, 3, 3> H_inv_H_imag = get_imag_matrix_from_complex_matrix(H_inv_H);
    Matrix<T, 3, 3> H_inv_H_answer_imag = {
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F}
    };
    tester.expect_near(H_inv_H_imag.data, H_inv_H_answer_imag.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k matrix imag.");

    DiagMatrix<Complex<T>, 3> I_Comp(DiagMatrix<Complex<T>, 3>::identity());
    Matrix<Complex<T>, 3, 3> H_inv_I;

    complex_gmres_k_matrix(H_comp, I_Comp, H_inv_I,
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho_vec, rep_num_vec);

    Matrix<T, 3, 3> I_real = get_real_matrix_from_complex_matrix(H_comp * H_inv_I);

    Matrix<T, 3, 3> I_answer_real({
        {1.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 1.0F}
        });

    tester.expect_near(I_real.data, I_answer_real.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k matrix real, diag.");

    Matrix<T, 3, 3> I_imag = get_imag_matrix_from_complex_matrix(H_comp * H_inv_I);

    Matrix<T, 3, 3> I_answer_imag({
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 0.0F}
        });

    tester.expect_near(I_imag.data, I_answer_imag.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k matrix imag, diag.");


    CompiledSparseMatrix<Complex<T>, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> C_comp({ 1.0F, 3.0F, 8.0F, 2.0F, 4.0F });

    Matrix<Complex<T>, 3, 3> C_inv;

    C_inv = complex_sparse_gmres_k_matrix_inv(C_comp,
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F),
        rho_vec, rep_num_vec, C_inv);

    Matrix<T, 3, 3> C_I_real = get_real_matrix_from_complex_matrix(C_comp * C_inv);

    Matrix<T, 3, 3> C_I_answer_real({
        {1.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 1.0F}
        });

    tester.expect_near(C_I_real.data, C_I_answer_real.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k matrix real, sparse.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void CheckBaseMatrix<T>::check_eigen_values_and_vectors(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 実数値のみの固有値 */
    Matrix<T, 3, 3> A0({ {6, -3, 5}, {-1, 4, -5}, {-3, 3, -4} });
    Matrix<T, 3, 3> A1({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });
    Matrix<T, 4, 4> Ae({ {11, 8, 5, 10}, {14, 1, 4, 15}, {2, 13, 16, 3}, {7, 12, 9, 6} });

    EigenSolverReal<T, 3> eigen_solver(A0, 5, static_cast<T>(1.0e-20F));
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
    std::vector<T> eigen_values = eigen_solver.get_eigen_values();
#else
    std::array<T, 3> eigen_values = eigen_solver.get_eigen_values();
#endif

    //std::cout << "eigen_values = ";
    //for (size_t i = 0; i < eigen_values.size(); ++i) {
    //    std::cout << eigen_values[i] << " ";
    //}
    //std::cout << std::endl << std::endl;

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
    std::vector<T> eigen_values_answer(
#else
    std::array<T, 3> eigen_values_answer(
#endif
        { 1, 2, 3 });

    eigen_solver.continue_solving_eigen_values();
    eigen_values = eigen_solver.get_eigen_values();

    tester.expect_near(eigen_values, eigen_values_answer, NEAR_LIMIT_SOFT,
        "check EigenSolverReal eigen values.");

    eigen_solver.continue_solving_eigen_values();
    eigen_solver.continue_solving_eigen_values();
    eigen_solver.continue_solving_eigen_values();
    eigen_values = eigen_solver.get_eigen_values();
    tester.expect_near(eigen_values, eigen_values_answer, NEAR_LIMIT_STRICT,
        "check EigenSolverReal eigen values, strict.");

    eigen_solver.solve_eigen_vectors(A0);
    Matrix<T, 3, 3> eigen_vectors = eigen_solver.get_eigen_vectors();

    //std::cout << "eigen_vectors = " << std::endl;
    //for (size_t j = 0; j < eigen_vectors.cols(); ++j) {
    //    for (size_t i = 0; i < eigen_vectors.rows(); ++i) {
    //        std::cout << eigen_vectors(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> A_mul_V = A0 * eigen_vectors;
    Matrix<T, 3, 3> V_mul_D = eigen_vectors * DiagMatrix<T, 3>(eigen_values);

    tester.expect_near(A_mul_V.data, V_mul_D.data, NEAR_LIMIT_SOFT,
        "check EigenSolverReal eigen vectors.");

    eigen_solver.solve_eigen_vectors(A0);
    eigen_vectors = eigen_solver.get_eigen_vectors();
    A_mul_V = A0 * eigen_vectors;
    V_mul_D = eigen_vectors * DiagMatrix<T, 3>(eigen_values_answer);

    tester.expect_near(A_mul_V.data, V_mul_D.data, NEAR_LIMIT_STRICT,
        "check EigenSolverReal eigen vectors, strict.");


    /* 複素数 固有値 */
    Matrix<Complex<T>, 3, 3> A1_comp({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });
    EigenSolverComplex<T, 3> eigen_solver_comp(A1, 5, static_cast<T>(1.0e-20F));
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
    std::vector<Complex<T>> eigen_values_comp
#else
    std::array<Complex<T>, 3> eigen_values_comp
#endif
        = eigen_solver_comp.get_eigen_values();

    //std::cout << "eigen_values_comp = ";
    //for (size_t i = 0; i < eigen_values_comp.size(); ++i) {
    //    std::cout << eigen_values_comp[i].real << " + " << eigen_values_comp[i].imag << "j, ";
    //}
    //std::cout << std::endl << std::endl;

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
    std::vector<T> eigen_values_comp_answer_real(
#else
    std::array<T, 3> eigen_values_comp_answer_real(
#endif
        { -1.5F, -1.5F, 6.0F });
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
    std::vector<T> eigen_values_comp_real = get_real_vector_from_complex_vector<T, 3>(eigen_values_comp);
#else
    std::array<T, 3> eigen_values_comp_real = get_real_vector_from_complex_vector<T, 3>(eigen_values_comp);
#endif

    tester.expect_near(eigen_values_comp_real, eigen_values_comp_answer_real, NEAR_LIMIT_SOFT,
        "check EigenSolverComplex real eigen values.");

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
    std::vector<T> eigen_values_comp_answer_imag(
#else
    std::array<T, 3> eigen_values_comp_answer_imag(
#endif
        { -0.8660254F, 0.8660254F, 0.0F });

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
    std::vector<T> eigen_values_comp_imag = get_imag_vector_from_complex_vector<T, 3>(eigen_values_comp);
#else
    std::array<T, 3> eigen_values_comp_imag = get_imag_vector_from_complex_vector<T, 3>(eigen_values_comp);
#endif

    tester.expect_near(eigen_values_comp_imag, eigen_values_comp_answer_imag, NEAR_LIMIT_SOFT,
        "check EigenSolverComplex imag eigen values.");

    eigen_solver_comp.continue_solving_eigen_values();
    eigen_values_comp = eigen_solver_comp.get_eigen_values();

    eigen_values_comp_real = get_real_vector_from_complex_vector<T, 3>(eigen_values_comp);
    tester.expect_near(eigen_values_comp_real, eigen_values_comp_answer_real, NEAR_LIMIT_STRICT,
        "check EigenSolverComplex real eigen values, strict.");
    eigen_values_comp_imag = get_imag_vector_from_complex_vector<T, 3>(eigen_values_comp);

    tester.expect_near(eigen_values_comp_imag, eigen_values_comp_answer_imag, NEAR_LIMIT_STRICT,
        "check EigenSolverComplex imag eigen values, strict.");


    eigen_solver_comp.solve_eigen_vectors(A1);
    Matrix<Complex<T>, 3, 3> eigen_vectors_comp = eigen_solver_comp.get_eigen_vectors();

    //std::cout << "eigen_vectors_comp = " << std::endl;
    //for (size_t j = 0; j < eigen_vectors_comp.cols(); ++j) {
    //    for (size_t i = 0; i < eigen_vectors_comp.rows(); ++i) {
    //        std::cout << eigen_vectors_comp(j, i).real << " + " << eigen_vectors_comp(j, i).imag << "j, ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;


    eigen_solver_comp.solve_eigen_vectors(A1);
    eigen_vectors_comp = eigen_solver_comp.get_eigen_vectors();

    Matrix<Complex<T>, 3, 3> A_mul_V_comp = A1_comp * eigen_vectors_comp;
    Matrix<Complex<T>, 3, 3> V_mul_D_comp = eigen_vectors_comp * DiagMatrix<Complex<T>, 3>(eigen_values_comp);

    Matrix<T, 3, 3> A_mul_V_real = get_real_matrix_from_complex_matrix(A_mul_V_comp);
    Matrix<T, 3, 3> V_mul_D_real = get_real_matrix_from_complex_matrix(V_mul_D_comp);
    tester.expect_near(A_mul_V_real.data, V_mul_D_real.data, NEAR_LIMIT_SOFT,
        "check EigenSolverComplex real eigen vectors.");

    Matrix<T, 3, 3> A_mul_V_imag = get_imag_matrix_from_complex_matrix(A_mul_V_comp);
    Matrix<T, 3, 3> V_mul_D_imag = get_imag_matrix_from_complex_matrix(V_mul_D_comp);
    tester.expect_near(A_mul_V_imag.data, V_mul_D_imag.data, NEAR_LIMIT_SOFT,
        "check EigenSolverComplex imag eigen vectors.");

    eigen_solver_comp.solve_eigen_vectors(A1);
    eigen_vectors_comp = eigen_solver_comp.get_eigen_vectors();
    A_mul_V_comp = A1_comp * eigen_vectors_comp;
    V_mul_D_comp = eigen_vectors_comp * DiagMatrix<Complex<T>, 3>(eigen_values_comp);

    A_mul_V_real = get_real_matrix_from_complex_matrix(A_mul_V_comp);
    V_mul_D_real = get_real_matrix_from_complex_matrix(V_mul_D_comp);
    tester.expect_near(A_mul_V_real.data, V_mul_D_real.data, NEAR_LIMIT_STRICT,
        "check EigenSolverComplex real eigen vectors, strict.");

    A_mul_V_imag = get_imag_matrix_from_complex_matrix(A_mul_V_comp);
    V_mul_D_imag = get_imag_matrix_from_complex_matrix(V_mul_D_comp);
    tester.expect_near(A_mul_V_imag.data, V_mul_D_imag.data, NEAR_LIMIT_STRICT,
        "check EigenSolverComplex imag eigen vectors, strict.");


    tester.throw_error_if_test_failed();
}

#endif // __CHECK_BASE_MATRIX_HPP__
