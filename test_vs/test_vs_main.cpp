#include <type_traits>
#include <iostream>

#include "MCAP_tester.hpp"

#include "python_numpy.hpp"
#include "base_matrix.hpp"

using namespace Tester;

template <typename T>
void check_matrix_vector_creation(void) {
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
void check_matrix_swap(void) {
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
void check_matrix_multiply(void) {
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
void check_lu_decomposition(void) {
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
void check_gmres_k_and_inverse(void) {
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
void check_matrix_transpose_multiply(void) {
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
void check_determinant_and_trace(void) {
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
void check_diag_matrix(void) {
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

    T ddd = D.get_trace();
    //std::cout << "trace " << ddd << std::endl;

    T ddd_answer = 6.0F;
    tester.expect_near(ddd, ddd_answer, NEAR_LIMIT_STRICT,
        "check DiagMatrix trace.");

    Matrix<T, 3, 3> D_dense = D.create_dense();

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
void check_sparse_matrix(void) {
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

    Matrix<T, 3, 3> Cc_mul_A = SparseCc.transpose();
    Matrix<T, 3, 3> Cn_mul_A = SparseCn.transpose();

    tester.expect_near(Cc_mul_A.data, Cn_mul_A.data, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix transpose.");

    CompiledSparseMatrix<T, 3, 3,
        RowIndices<0, 0, 2, 1, 2>,
        RowPointers<0, 1, 3, 5>> SparseCc_mul_scalar = SparseCc * static_cast<T>(3);
    Matrix<T, 3, 3> SparseCc_mul_scalar_dense = SparseCc_mul_scalar.create_dense();

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
    Matrix<T, 3, 3> Scalar_mul_SparseCc_dense = Scalar_mul_SparseCc.create_dense();

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
    Matrix<T, 3, 4> Sparse_mul_Sparse_dense = Sparse_mul_Sparse.create_dense();

    Matrix<T, 3, 4> Sparse_mul_Sparse_answer({
        {1, 0, 0, 0},
        {3, 16, 32, 0},
        {6, 8, 32, 2}
        });

    tester.expect_near(Sparse_mul_Sparse_dense.data, Sparse_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply SparseMatrix.");

    Matrix<T, 3, 4> SparseTranspose_mul_Sparse =
        matrix_multiply_SparseATranspose_mul_SparseB(SparseCc, SEc);

    Matrix<T, 3, 4> SparseTranspose_mul_Sparse_answer({
        {10, 0, 24, 3},
        {0, 4, 8, 0},
        {24, 8, 80, 8}
        });

    tester.expect_near(SparseTranspose_mul_Sparse.data, SparseTranspose_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix transpose multiply SparseMatrix.");

    Matrix<T, 3, 3> Sparse_mul_SparseTranspose =
        matrix_multiply_SparseA_mul_SparseBTranspose(SparseCc, SparseCc);

    Matrix<T, 3, 3> Sparse_mul_SparseTranspose_answer({
        {1, 3, 0},
        {3, 73, 32},
        {0, 32, 20}
        });

    tester.expect_near(Sparse_mul_SparseTranspose.data, Sparse_mul_SparseTranspose_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply SparseMatrix transpose.");

    DiagMatrix<T, 3> DiagJ({ 10, 20, 30 });

    auto Sparse_add_Diag = SparseCc + DiagJ;
    Matrix<T, 3, 3> Sparse_add_Diag_dense = Sparse_add_Diag.create_dense();

    Matrix<T, 3, 3> Sparse_add_Diag_answer({
        {11, 0, 0},
        {3, 20, 8},
        {0, 2, 34}
        });

    tester.expect_near(Sparse_add_Diag_dense.data, Sparse_add_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add DiagMatrix.");

    auto Diag_add_Sparse = DiagJ + SparseCc;
    Matrix<T, 3, 3> Diag_add_Sparse_dense = Diag_add_Sparse.create_dense();

    tester.expect_near(Diag_add_Sparse_dense.data, Sparse_add_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix add SparseMatrix.");

    auto Sparse_sub_Diag = SparseCc - DiagJ;
    Matrix<T, 3, 3> Sparse_sub_Diag_dense = Sparse_sub_Diag.create_dense();

    Matrix<T, 3, 3> Sparse_sub_Diag_answer({
        {-9, 0, 0},
        {3, -20, 8},
        {0, 2, -26}
        });

    tester.expect_near(Sparse_sub_Diag_dense.data, Sparse_sub_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub DiagMatrix.");

    auto Diag_sub_Sparse = DiagJ - SparseCc;
    Matrix<T, 3, 3> Diag_sub_Sparse_dense = Diag_sub_Sparse.create_dense();

    Matrix<T, 3, 3> Diag_sub_Sparse_answer({
        {9, 0, 0},
        {-3, 20, -8},
        {0, -2, 26}
        });

    tester.expect_near(Diag_sub_Sparse_dense.data, Diag_sub_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix sub SparseMatrix.");

    auto Sparse_mul_Diag = SparseCc * DiagJ;
    Matrix<T, 3, 3> Sparse_mul_Diag_dense = Sparse_mul_Diag.create_dense();

    Matrix<T, 3, 3> Sparse_mul_Diag_answer({
        {10, 0, 0},
        {30, 0, 240},
        {0, 40, 120}
        });

    tester.expect_near(Sparse_mul_Diag_dense.data, Sparse_mul_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply DiagMatrix.");

    auto Diag_mul_Sparse = DiagJ * SparseCc;
    Matrix<T, 3, 3> Diag_mul_Sparse_dense = Diag_mul_Sparse.create_dense();

    Matrix<T, 3, 3> Diag_mul_Sparse_answer({
        {10, 0, 0},
        {60, 0, 160},
        {0, 60, 120}
        });

    tester.expect_near(Diag_mul_Sparse_dense.data, Diag_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply SparseMatrix.");

    auto Sparse_add_Sparse = SparseCc + SparseCc;
    auto Sparse_add_Sparse_dense = Sparse_add_Sparse.create_dense();

    Matrix<T, 3, 3> Sparse_add_Sparse_answer({
        {2, 0, 0},
        {6, 0, 16},
        {0, 4, 8}
        });

    tester.expect_near(Sparse_add_Sparse_dense.data, Sparse_add_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add SparseMatrix.");

    auto Sparse_sub_Sparse = SparseCc - SparseCc;
    auto Sparse_sub_Sparse_dense = Sparse_sub_Sparse.create_dense();

    Matrix<T, 3, 3> Sparse_sub_Sparse_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(Sparse_sub_Sparse_dense.data, Sparse_sub_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub SparseMatrix.");

    Matrix<T, 3, 3> Transpose_Diag_mul_Sparse = matrix_multiply_Transpose_DiagA_mul_SparseB(DiagJ, SparseCc);

    Matrix<T, 3, 3> Transpose_Diag_mul_Sparse_answer({ {10, 60, 0}, {0, 0, 60}, {0, 160, 120} });

    tester.expect_near(Transpose_Diag_mul_Sparse.data, Transpose_Diag_mul_Sparse_answer.data, NEAR_LIMIT_STRICT,
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

    Matrix<T, 4, 3> SB_dense = SBc.create_dense();

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
        static_cast<T>(0.0F), static_cast<T>(1.0e-10F), X_temp);

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
void check_matrix_cocatenation(void) {
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
    auto A_v_B_dense = A_v_B.create_dense();

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
    auto A_v_C_dense = A_v_C.create_dense();

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
    auto B_v_A_dense = B_v_A.create_dense();

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
    auto B_v_B_dense = B_v_B.create_dense();

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
    auto B_v_C_dense = B_v_C.create_dense();

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
    auto C_v_A_dense = C_v_A.create_dense();

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
    auto C_v_B_dense = C_v_B.create_dense();

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
    auto C_v_C_dense = C_v_C.create_dense();

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
    auto A_h_B_dense = A_h_B.create_dense();

    Matrix<T, 3, 6> A_h_B_answer({
        {1, 2, 3, 1, 0, 0},
        {5, 4, 6, 0, 2, 0},
        {9, 8, 7, 0, 0, 3}
        });

    tester.expect_near(A_h_B_dense.data, A_h_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Diag.");

    auto A_h_C = concatenate_horizontally(DenseA, SparseCc);
    auto A_h_C_dense = A_h_C.create_dense();

    Matrix<T, 3, 6> A_h_C_answer({
        {1, 2, 3, 1, 0, 0},
        {5, 4, 6, 3, 0, 8},
        {9, 8, 7, 0, 2, 4}
        });

    tester.expect_near(A_h_C_dense.data, A_h_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Dense and Sparse.");

    auto B_h_A = concatenate_horizontally(D, DenseA);
    auto B_h_A_dense = B_h_A.create_dense();

    Matrix<T, 3, 6> B_h_A_answer({
        {1, 0, 0, 1, 2, 3},
        {0, 2, 0, 5, 4, 6},
        {0, 0, 3, 9, 8, 7}
        });

    tester.expect_near(B_h_A_dense.data, B_h_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Dense.");

    auto B_h_B = concatenate_horizontally(D, D * static_cast<T>(2));
    auto B_h_B_dense = B_h_B.create_dense();

    Matrix<T, 3, 6> B_h_B_answer({
        {1, 0, 0, 2, 0, 0},
        {0, 2, 0, 0, 4, 0},
        {0, 0, 3, 0, 0, 6}
        });
    
    tester.expect_near(B_h_B_dense.data, B_h_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Diag.");
    
    auto B_h_C = concatenate_horizontally(D, SparseCc);
    auto B_h_C_dense = B_h_C.create_dense();
    
    Matrix<T, 3, 6> B_h_C_answer({
        {1, 0, 0, 1, 0, 0},
        {0, 2, 0, 3, 0, 8},
        {0, 0, 3, 0, 2, 4}
        });
    
    tester.expect_near(B_h_C_dense.data, B_h_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Diag and Sparse.");
    
    auto C_h_A = concatenate_horizontally(SparseCc, DenseA);
    auto C_h_A_dense = C_h_A.create_dense();
    
    Matrix<T, 3, 6> C_h_A_answer({
        {1, 0, 0, 1, 2, 3},
        {3, 0, 8, 5, 4, 6},
        {0, 2, 4, 9, 8, 7}
        });
    
    tester.expect_near(C_h_A_dense.data, C_h_A_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Dense.");
    
    auto C_h_B = concatenate_horizontally(SparseCc, D);
    auto C_h_B_dense = C_h_B.create_dense();
    
    Matrix<T, 3, 6> C_h_B_answer({
        {1, 0, 0, 1, 0, 0},
        {3, 0, 8, 0, 2, 0},
        {0, 2, 4, 0, 0, 3}
        });
    
    tester.expect_near(C_h_B_dense.data, C_h_B_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Diag.");
    
    auto C_h_C = concatenate_horizontally(SparseCc, SparseCc * static_cast<T>(2));
    auto C_h_C_dense = C_h_C.create_dense();
    
    Matrix<T, 3, 6> C_h_C_answer({
        {1, 0, 0, 2, 0, 0},
        {3, 0, 8, 6, 0, 16},
        {0, 2, 4, 0, 4, 8}
        });
    
    tester.expect_near(C_h_C_dense.data, C_h_C_answer.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally Sparse and Sparse.");
    
    Matrix<T, 2, 2> Aa;
    Aa(0, 0) = 1.0F; Aa(0, 1) = 2.0F;
    Aa(1, 0) = 3.0F; Aa(1, 1) = 4.0F;
    
    Matrix<T, 2, 2> Ba;
    Ba(0, 0) = 5.0F; Ba(0, 1) = 6.0F;
    Ba(1, 0) = 7.0F; Ba(1, 1) = 8.0F;
    
    Matrix<T, 3, 2> Ca;
    Ca(0, 0) = 9.0F; Ca(0, 1) = 10.0F;
    Ca(1, 0) = 11.0F; Ca(1, 1) = 12.0F;
    Ca(2, 0) = 11.0F; Ca(2, 1) = 12.0F;
    
    Matrix<T, 3, 2> Da;
    Da(0, 0) = 13.0F; Da(0, 1) = 14.0F;
    Da(1, 0) = 15.0F; Da(1, 1) = 16.0F;
    Da(2, 0) = 15.0F; Da(2, 1) = 16.0F;
    
    Matrix<T, 5, 4> C = concatenate_square(Aa, Ba, Ca, Da);
    //std::cout << "Square = " << std::endl;
    //for (size_t j = 0; j < C.cols(); ++j) {
    //    for (size_t i = 0; i < C.rows(); ++i) {
    //        std::cout << C(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;
    
    Matrix<T, 5, 4> C_answer({
        {1, 2, 5, 6},
        {3, 4, 7, 8},
        {9, 10, 13, 14},
        {11, 12, 15, 16},
        {11, 12, 15, 16} });
    tester.expect_near(C.data, C_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix concatenate.");
    
    
    tester.throw_error_if_test_failed();
}

template <typename T>
void check_cholesky_decomposition(void) {
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
void check_qr_decomposition(void) {
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
void check_variable_sparse_matrix(void) {
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
void check_triangular_matrix(void) {
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

    Matrix<T, 4, 4> Test_lower = TS.create_dense();
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
void check_complex(void) {
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


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_eigen_values_and_vectors(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 実数値のみの固有値 */
    Matrix<T, 3, 3> A0({ {6, -3, 5}, {-1, 4, -5}, {-3, 3, -4} });
    Matrix<T, 3, 3> A1({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });
    Matrix<T, 4, 4> Ae({ {11, 8, 5, 10}, {14, 1, 4, 15}, {2, 13, 16, 3}, {7, 12, 9, 6} });

    EigenSolverReal<T, 3> eigen_solver(A0, 5, static_cast<T>(1.0e-20F));
#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<T> eigen_values = eigen_solver.get_eigen_values();
#else
    std::array<T, 3> eigen_values = eigen_solver.get_eigen_values();
#endif

    //std::cout << "eigen_values = ";
    //for (size_t i = 0; i < eigen_values.size(); ++i) {
    //    std::cout << eigen_values[i] << " ";
    //}
    //std::cout << std::endl << std::endl;

#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<T> eigen_values_answer(
#else
    std::array<T, 3> eigen_values_answer(
#endif
        { 3, 2, 1 });

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
#ifdef BASE_MATRIX_USE_STD_VECTOR
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

#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<T> eigen_values_comp_answer_real(
#else
    std::array<T, 3> eigen_values_comp_answer_real(
#endif
        { 6, -1.5F, -1.5F });
#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<T> eigen_values_comp_real = get_real_vector_from_complex_vector<T, 3>(eigen_values_comp);
#else
    std::array<T, 3> eigen_values_comp_real = get_real_vector_from_complex_vector<T, 3>(eigen_values_comp);
#endif

    tester.expect_near(eigen_values_comp_real, eigen_values_comp_answer_real, NEAR_LIMIT_SOFT,
        "check EigenSolverComplex real eigen values.");

#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<T> eigen_values_comp_answer_imag(
#else
    std::array<T, 3> eigen_values_comp_answer_imag(
#endif
        { 0, 0.8660254F, -0.8660254F });

#ifdef BASE_MATRIX_USE_STD_VECTOR
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


template <typename T>
void check_base_matrix_calc(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    check_matrix_vector_creation<T>();

    check_matrix_swap<T>();

    check_matrix_multiply<T>();

    check_lu_decomposition<T>();

    check_gmres_k_and_inverse<T>();

    check_matrix_transpose_multiply<T>();

    check_determinant_and_trace<T>();

    check_determinant_and_trace<T>();

    check_diag_matrix<T>();

    check_sparse_matrix<T>();

    check_matrix_cocatenation<T>();

    check_cholesky_decomposition<T>();

    check_qr_decomposition<T>();

    check_variable_sparse_matrix<T>();

    check_triangular_matrix<T>();

    check_complex<T>();

    check_eigen_values_and_vectors<T>();
}


template <typename T>
void check_python_numpy_base(void) {
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

    /* 演算 */
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
        >> CL({1, 3, 8, 1, 2, 4});

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


    tester.throw_error_if_test_failed();
}




template <typename T>
void check_python_numpy_left_divide_and_inv(void) {
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

    Matrix<DefDiag, T, 3> B_Inv_answer({ 1, 0.5F, 0.333333F });

    tester.expect_near(B_Inv.matrix.data, B_Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
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
void check_python_numpy_concatenate(void) {
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


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_python_numpy_transpose(void) {
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
    //std::cout << "AL_T = " << std::endl;
    //for (size_t j = 0; j < AL_T.cols(); ++j) {
    //    for (size_t i = 0; i < AL_T.rows(); ++i) {
    //        std::cout << AL_T(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 4> CL_T_answer({
        {1, 0, 0, 0},
        {3, 0, 8, 1},
        {0, 2, 4, 0}
        });
    tester.expect_near(CL_T.matrix.data, CL_T_answer.matrix.data, NEAR_LIMIT_STRICT,
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
void check_python_numpy_lu(void) {
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
void check_check_python_numpy_cholesky(void) {
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
    auto A_ch_d = A_ch.matrix.create_dense();
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
void check_check_python_numpy_transpose_operation(void) {
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

    Matrix<DefDense, T, 3, 3> B_Ct_answer({
        {1, 3, 0},
        {0, 0, 4},
        {0, 24, 12}
        });

    tester.expect_near(B_Ct.matrix.data, B_Ct_answer.matrix.data, NEAR_LIMIT_STRICT,
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

    Matrix<DefDense, T, 3, 3> C_Ct_answer({
        {1, 3, 0},
        {3, 73, 32},
        {0, 32, 20}
        });

    tester.expect_near(C_Ct.matrix.data, C_Ct_answer.matrix.data, NEAR_LIMIT_STRICT,
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

    Matrix<DefDense, T, 3, 3> Ct_B_answer({
        {1, 6, 0},
        {0, 0, 6},
        {0, 16, 12}
        });

    tester.expect_near(Ct_B.matrix.data, Ct_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Sparse and Diag.");

    auto Ct_C = ATranspose_mul_B(C, C);

    Matrix<DefDense, T, 3, 3> Ct_C_answer({
        {10, 0, 24},
        {0, 4, 8},
        {24, 8, 80}
        });

    tester.expect_near(Ct_C.matrix.data, Ct_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check ATranspose_mul_B Sparse and Sparse.");


    tester.throw_error_if_test_failed();
}

template<typename T>
void check_python_numpy_qr(void) {
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
void check_python_numpy_eig(void) {
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
    Matrix<DefDense, T, 3, 1> eigen_values_answer({ {3}, {2}, {1} });
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

    /* 複素数固有値 */
    Matrix<DefDense, T, 3, 3> A1({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });
    Matrix<DefDense, Complex<T>, 3, 3> A1_comp({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });

    static auto eig_solver_comp = make_LinalgSolverEig(A1);
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

    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_real({ {6}, {-1.5F}, {-1.5F} });
    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_imag({ {0}, {0.8660254F}, {-0.8660254F} });

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

template <typename T>
void check_python_numpy_calc(void) {

    check_python_numpy_base<T>();

    check_python_numpy_left_divide_and_inv<T>();

    check_python_numpy_concatenate<T>();

    check_python_numpy_transpose<T>();

    check_python_numpy_lu<T>();

    check_check_python_numpy_cholesky<T>();

    check_check_python_numpy_transpose_operation<T>();

    check_python_numpy_qr<T>();

    check_python_numpy_eig<T>();
}


int main() {

    check_base_matrix_calc<double>();

    check_base_matrix_calc<float>();

    check_python_numpy_calc<double>();

    check_python_numpy_calc<float>();


    return 0;
}
