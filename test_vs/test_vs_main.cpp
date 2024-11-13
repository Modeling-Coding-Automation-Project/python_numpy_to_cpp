#include <type_traits>
#include <iostream>

#include "MCAP_tester.hpp"

#include "python_numpy.hpp"
#include "base_matrix.hpp"

using namespace Tester;

template <typename T>
void check_base_matrix_calc(void) {
    using namespace Base::Matrix;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 行列の作成 */
    Matrix<T, 2, 3> A;
    A(0, 0) = 1.0F; A(0, 1) = 2.0F; A(0, 2) = 3.0F;
    A(1, 0) = 4.0F; A(1, 1) = 5.0F; A(1, 2) = 6.0F;

    /* ベクトルの作成 */
    Vector<T, 3> b;
    b[0] = 1.0F;
    b[1] = 2.0F;
    b[2] = 3.0F;

    Vector<T, 2> c;
    c[0] = 4.0F;
    c[1] = 5.0F;

    Vector<T, 3> b_add_b = b + b;

    Matrix<T, 2, 3> A_add_A = A + A;

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

    /* 行列結合 */
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

    /* 転置積 */
    Matrix<T, 3, 2> Trans = matrix_multiply_A_mul_BT(H, E);
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
    Matrix<T, 3, 3> X_temp = Matrix<T, 3, 3>::identity();

    //H_inv = gmres_k_matrix_inv(H, 0.0, 1.0e-10, X_temp);
    H_inv = H.inv();

    //std::cout << "H_inv = " << std::endl;
    //for (size_t j = 0; j < 3; ++j) {
    //    for (size_t i = 0; i < 3; ++i) {
    //        std::cout << H_inv(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> H_inv_answer({
        {-0.242105F, -0.136842F, 0.389474F},
        {0.336842F, 0.494737F, -0.715789F},
        {0.0631579F, -0.0947368F, 0.115789F}
        });
    tester.expect_near(H_inv.data, H_inv_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix inverse.");

    /* 行列式、トレース */
    T det = lu.get_determinant();
    //std::cout << "det = " << det << std::endl;
    //std::cout << std::endl;

    T det_answer = -95.0F;
    tester.expect_near(det, det_answer, NEAR_LIMIT_STRICT,
        "check Matrix determinant.");

    T trace = H.get_trace();
    //std::cout << "trace = " << trace << std::endl;
    //std::cout << std::endl;

    T trace_answer = 13.0F;
    tester.expect_near(trace, trace_answer, NEAR_LIMIT_STRICT,
        "check Matrix trace.");

    /* 対角行列 */
    DiagMatrix<T, 3> D;
    D[0] = 1.0F;
    D[1] = 2.0F;
    D[2] = 3.0F;

    DiagMatrix<T, 3> D_add_D = D * D;

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

    /* コレスキー分解 */
    Matrix<T, 3, 3> K({ {10, 1, 2}, {1, 20, 4}, {2, 4, 30} });
    SparseMatrix<T, 3, 3, 6> K_s({ 1, 8, 3, 3, 4 }, { 0, 1, 2, 1, 2 }, { 0, 1, 3, 5 });

    Matrix<T, 3, 3> K_ch;
    bool flag = false;
    K_ch = cholesky_decomposition(K, K_ch, flag);
    //K_ch = cholesky_decomposition_sparse(K_s, K_ch, flag);

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

    Matrix<T, 3, 3> C_Add_A = SparseCc + DenseA;

    Matrix<T, 3, 3> C_Add_A_answer({
        {2, 2, 3},
        {8, 4, 14},
        {9, 10, 11}
        });

    tester.expect_near(C_Add_A.data, C_Add_A_answer.data, NEAR_LIMIT_STRICT,
        "check CompiledSparseMatrix add Matrix.");

    Matrix<T, 3, 3> DenseC = SparseCc * DenseB;

    //std::cout << "DenseC = " << std::endl;
    //for (size_t j = 0; j < DenseC.cols(); ++j) {
    //    for (size_t i = 0; i < DenseC.rows(); ++i) {
    //        std::cout << DenseC(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> DenseC_answer({
        {1, 2, 3},
        {75, 70, 65},
        {46, 40, 40}
        });
    tester.expect_near(DenseC.data, DenseC_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply Matrix.");

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

    Matrix<T, 3, 3> DenseH = SparseCc - DenseG;
    //Matrix<T, 3, 3> DenseH = DenseG - SA;

    //std::cout << "DenseH = " << std::endl;
    //for (size_t j = 0; j < DenseH.cols(); ++j) {
    //    for (size_t i = 0; i < DenseH.rows(); ++i) {
    //        std::cout << DenseH(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> DenseH_answer({
        {0, -2, -3},
        {-2, -4, 2},
        {-9, -6, -3}
        });
    tester.expect_near(DenseH.data, DenseH_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix subtract Matrix.");

    Matrix<T, 3, 4> DenseI = SA * SE;

    //std::cout << "DenseI = " << std::endl;
    //for (size_t j = 0; j < DenseI.cols(); ++j) {
    //    for (size_t i = 0; i < DenseI.rows(); ++i) {
    //        std::cout << DenseI(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 4> DenseI_answer({
        {1, 0, 0, 0},
        {3, 16, 32, 0},
        {6, 8, 32, 2}
        });
    tester.expect_near(DenseI.data, DenseI_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix multiply SparseMatrix.");

    DiagMatrix<T, 3> DiagJ({ 10, 20, 30 });

    Matrix<T, 3, 3> Sparse_add_Diag = SparseCc + DiagJ;

    Matrix<T, 3, 3> Sparse_add_Diag_answer({
        {11, 0, 0},
        {3, 20, 8},
        {0, 2, 34}
        });

    tester.expect_near(Sparse_add_Diag.data, Sparse_add_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add DiagMatrix.");

    Matrix<T, 3, 3> Sparse_sub_Diag = SparseCc - DiagJ;

    Matrix<T, 3, 3> Sparse_sub_Diag_answer({
        {-9, 0, 0},
        {3, -20, 8},
        {0, 2, -26}
        });

    tester.expect_near(Sparse_sub_Diag.data, Sparse_sub_Diag_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub DiagMatrix.");

    Matrix<T, 3, 3> DenseK = DiagJ * SA;

    Matrix<T, 3, 3> Sparse_add_Sparse = SparseCc + SparseCc;

    Matrix<T, 3, 3> Sparse_add_Sparse_answer({
        {2, 0, 0},
        {6, 0, 16},
        {0, 4, 8}
        });

    tester.expect_near(Sparse_add_Sparse.data, Sparse_add_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add SparseMatrix.");

    Matrix<T, 3, 3> Sparse_sub_Sparse = SparseCc - SparseCc;

    Matrix<T, 3, 3> Sparse_sub_Sparse_answer({
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        });

    tester.expect_near(Sparse_sub_Sparse.data, Sparse_sub_Sparse_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix sub SparseMatrix.");

    //std::cout << "DenseK = " << std::endl;
    //for (size_t j = 0; j < DenseK.cols(); ++j) {
    //    for (size_t i = 0; i < DenseK.rows(); ++i) {
    //        std::cout << DenseK(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> DenseK_answer({
        {10, 0, 0},
        {60, 0, 160},
        {0, 60, 120}
        });
    tester.expect_near(DenseK.data, DenseK_answer.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply SparseMatrix.");

    Matrix<T, 3, 3> DenseL = matrix_multiply_A_mul_SparseBT(DenseG, SA);

    //std::cout << "DenseL = " << std::endl;
    //for (size_t j = 0; j < DenseL.cols(); ++j) {
    //    for (size_t i = 0; i < DenseL.rows(); ++i) {
    //        std::cout << DenseL(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> DenseL_answer({
        {1, 27, 16},
        {5, 63, 32},
        {9, 83, 44}
        });
    tester.expect_near(DenseL.data, DenseL_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix multiply SparseMatrix transpose.");

    Matrix<T, 3, 3> DenseM = matrix_multiply_AT_mul_SparseB(DenseG, SA);

    //std::cout << "DenseM = " << std::endl;
    //for (size_t j = 0; j < DenseM.cols(); ++j) {
    //    for (size_t i = 0; i < DenseM.rows(); ++i) {
    //        std::cout << DenseM(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> DenseM_answer({
        {16, 18, 76},
        {14, 16, 64},
        {21, 14, 76}
        });
    tester.expect_near(DenseM.data, DenseM_answer.data, NEAR_LIMIT_STRICT,
        "check Matrix transpose multiply SparseMatrix.");

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
    ColVector<T, 3> Dense_m = colV_mul_SB(b_col, SA);

    //std::cout << "b_col * SA = ";
    //for (size_t i = 0; i < Dense_m.size(); ++i) {
    //    std::cout << Dense_m[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    ColVector<T, 3> Dense_m_answer({ 7, 6, 28 });
    tester.expect_near(Dense_m.data, Dense_m_answer.data, NEAR_LIMIT_STRICT,
        "check Column Vector multiply SparseMatrix.");

    /* スパース行列の逆行列計算 */
    x_gmres_k = sparse_gmres_k(SA, b, x_gmres_k_0, static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho, rep_num);

    //std::cout << "x_gmres_k = ";
    //for (size_t i = 0; i < x_gmres_k.size(); ++i) {
    //    std::cout << x_gmres_k[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> x_gmres_k_answer_2({ 1, 1.75, -0.125 });
    tester.expect_near(x_gmres_k.data, x_gmres_k_answer_2.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix GMRES k.");

    SparseMatrix<T, 4, 3, 6> SB({ 1, 3, 2, 8, 4, 1 }, { 0, 1, 2, 1, 2, 1 }, { 0, 2, 3, 5, 6 });
    x_gmres_k = sparse_gmres_k_rect(SB, b_2, x_gmres_k_rect_0, static_cast<T>(0.0F), static_cast<T>(1.0e-10F), rho, rep_num);

    //std::cout << "x_gmres_k = ";
    //for (size_t i = 0; i < x_gmres_k.size(); ++i) {
    //    std::cout << x_gmres_k[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Vector<T, 3> x_gmres_k_answer_3({ 0.478261F, 0.173913F, 0.521739F });
    tester.expect_near(x_gmres_k.data, x_gmres_k_answer_3.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix GMRES k rect.");

    Matrix<T, 4, 3> SB_dense = SB.create_dense();

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

    Matrix<T, 3, 3> S_inv = sparse_gmres_k_matrix_inv(SA, static_cast<T>(0.0F), static_cast<T>(1.0e-10F), X_temp);

    //std::cout << "S_inv = " << std::endl;
    //for (size_t j = 0; j < S_inv.cols(); ++j) {
    //    for (size_t i = 0; i < S_inv.rows(); ++i) {
    //        std::cout << S_inv(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<T, 3, 3> S_inv_answer({
        {1, 0, 0},
        {0.75, -0.25, 0.5},
        {-0.375, 0.125, 0}
        });
    tester.expect_near(S_inv.data, S_inv_answer.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix inverse.");

    SparseMatrix<T, 4, 3, 6> SB_test({ 1, 3, 2, 8, 4, 1 }, { 0, 1, 2, 1, 2, 1 }, { 0, 2, 3, 5, 6 });
    SparseMatrix<T, 4, 3, 6> SB_test_2 = std::move(SB_test);


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


    /* QR分解 */
    Matrix<T, 3, 3> C_dense({ {1, 0, 0}, {3, 0, 8}, {0 ,2, 4} });
    //QRDecomposition<T, 3, 3> qr(C_dense, static_cast<T>(1.0e-10F));
    QRDecompositionSparse<T, 3, 3, 5> qr(SA, static_cast<T>(1.0e-10F));

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

    /* 可変スパース行列 */
    VariableSparseMatrix<T, 3, 3> CV;
    std::memcpy(&CV.values[0], &A_value[0], 5 * sizeof(CV.values[0]));
    std::memcpy(&CV.row_indices[0], &A_row_indices[0], 5 * sizeof(CV.row_indices[0]));
    std::memcpy(&CV.row_pointers[0], &A_row_pointers[0], 4 * sizeof(CV.row_pointers[0]));

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

    /* 三角スパース行列 */
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

    Vector<T, 3> x_gmres_k_comp_imag = get_real_vector_from_complex_vector(x_gmres_k_comp);
    Vector<T, 3> x_gmres_k_comp_answer_imag({ -0.178525226390686F, 0.248382923673997F, 0.046571798188875F });
    tester.expect_near(x_gmres_k_comp_real.data, x_gmres_k_comp_answer_real.data, NEAR_LIMIT_STRICT,
        "check Complex GMRES k imag.");


    /* 実数値のみの固有値 */
    Matrix<T, 3, 3> A0({ {6, -3, 5}, {-1, 4, -5}, {-3, 3, -4} });
    Matrix<T, 3, 3> A1({ {1, 2, 3}, {3, 1, 2}, {2, 3, 1} });
    Matrix<T, 4, 4> Ae({ {11, 8, 5, 10}, {14, 1, 4, 15}, {2, 13, 16, 3}, {7, 12, 9, 6} });

    EigenSolverReal<T, 3> eigen_solver(A0, 5, static_cast<T>(1.0e-10F));
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
    EigenSolverComplex<T, 3> eigen_solver_comp(A1, 5, static_cast<T>(1.0e-10F));
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
        { 0, -0.8660254F, 0.8660254F });

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

#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<Complex<T>> eigen_values_answer_comp(3);
#else
    std::array<Complex<T>, 3> eigen_values_answer_comp;
#endif
    eigen_values_answer_comp[0] = Complex<T>(6, 0);
    eigen_values_answer_comp[1] = Complex<T>(-1.5F, -0.8660254F);
    eigen_values_answer_comp[2] = Complex<T>(-1.5F, 0.8660254F);

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
void check_python_numpy_calc(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    const T NEAR_LIMIT_SOFT = 1.0e-2F;

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

    Matrix<DefDiag, T, 3> B({ 1, 2, 3 });
    Matrix<DefDense, T, 4, 2> BB({ { 1, 2 }, {3, 4}, {5, 6}, {7, 8} });

    Matrix<DefSparse, T, 3, 3, 5> C(
        { 1.0F, 3.0F, 8.0F, 2.0F, 4.0F },
        { 0, 0, 2, 1, 2 },
        { 0, 1, 3, 5 }
    );

    auto A_add_B = A - B;

    auto D = C + C;

    //std::cout << "D = " << std::endl;
    //for (size_t j = 0; j < D.matrix.cols(); ++j) {
    //    for (size_t i = 0; i < D.matrix.rows(); ++i) {
    //        std::cout << D.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 3> D_answer({ {2, 0, 0}, {6, 0, 16}, {0, 4, 8} });
    tester.expect_near(D.matrix.data, D_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check SparseMatrix add SparseMatrix.");

    auto E = B * B;

    //std::cout << "E = " << std::endl;
    //for (size_t i = 0; i < E.matrix.cols(); ++i) {
    //    std::cout << E.matrix[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Matrix<DefDiag, T, 3> E_answer({ 1, 4, 9 });
    tester.expect_near(E.matrix.data, E_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiagMatrix multiply DiagMatrix.");


    /* 左除算 */
    Matrix<DefDense, T, 3, 2> b({ { 4, 10 }, { 5, 18 }, { 6, 23 } });

    static auto solver = make_LinalgSolver(C, C);

    auto x = solver.solve(C, C);
    //auto x = solver.get_answer();
    //std::cout << "x = " << std::endl;
    //for (size_t j = 0; j < x.matrix.cols(); ++j) {
    //    for (size_t i = 0; i < x.matrix.rows(); ++i) {
    //        std::cout << x.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix <DefDense, T, 3, 3> x_answer({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} });
    tester.expect_near(x.matrix.data, x_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve.");

    /* 対角　左除算 */
    static auto solver_diag = make_LinalgSolver(B, B);

    auto x_d = solver_diag.solve(B, B);
    //std::cout << "x_d = ";
    //for (size_t i = 0; i < x_d.matrix.rows(); ++i) {
    //    std::cout << x_d.matrix[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Matrix<DefDiag, T, 3> x_d_answer({ 1, 1, 1 });
    tester.expect_near(x_d.matrix.data, x_d_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver solve diag.");

    /* 矩形　左除算 */
    Matrix<DefDense, T, 4, 3> AL({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7}, {2, 2, 3} });
    Matrix<DefDiag, T, 4> BL({ 1, 2, 3, 4 });
    Matrix<DefSparse, T, 4, 3, 6> CL(Matrix<DefDense, T, 4, 3>({ {1, 3, 0 }, { 0, 0, 2 }, { 0, 8, 4 }, { 0, 1, 0 } }).matrix);

    static auto lstsq_solver = make_LinalgLstsqSolver(CL, CL);

    auto XX = lstsq_solver.solve(CL, CL);
    //std::cout << "XX = " << std::endl;
    //for (size_t j = 0; j < XX.matrix.cols(); ++j) {
    //    for (size_t i = 0; i < XX.matrix.rows(); ++i) {
    //        std::cout << XX.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 3> XX_answer({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} });
    tester.expect_near(XX.matrix.data, XX_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgLstsqSolver solve.");

    /* 逆行列 */
    static auto inv_solver = make_LinalgSolver(C);
    auto Inv = inv_solver.inv(C);
    //std::cout << "Inv = " << std::endl;
    //for (size_t j = 0; j < Inv.matrix.cols(); ++j) {
    //    for (size_t i = 0; i < Inv.matrix.rows(); ++i) {
    //        std::cout << Inv.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 3> Inv_answer({ {1, 0, 0}, {0.75F, -0.25F, 0.5F}, {-0.375F, 0.125F, 0} });
    tester.expect_near(Inv.matrix.data, Inv_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv.");

    /* 対角用 逆行列 */
    static auto inv_solver_d = make_LinalgSolver(B);
    auto Inv_d = inv_solver_d.inv(B);
    //std::cout << "Inv_d = ";
    //for (size_t i = 0; i < Inv_d.matrix.rows(); ++i) {
    //    std::cout << Inv_d.matrix[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Matrix<DefDiag, T, 3> Inv_d_answer({ 1, 0.5F, 0.333333F });
    tester.expect_near(Inv_d.matrix.data, Inv_d_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolver inv diag.");

    /* 結合 */
    auto B_C = concatenate_vertically(B, C);
    auto B_C_dense = B_C.matrix.create_dense();

    auto A_A = concatenate_horizontally(CL, CL);
    Base::Matrix::Matrix<T, 4, 6> A_A_dense = A_A.matrix.create_dense();
    //std::cout << "A_A_dense = " << std::endl;
    //for (size_t j = 0; j < A_A_dense.cols(); ++j) {
    //    for (size_t i = 0; i < A_A_dense.rows(); ++i) {
    //        std::cout << A_A_dense(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 4, 6> A_A_answer({
        {1, 3, 0, 1, 3, 0},
        {0, 0, 2, 0, 0, 2},
        {0, 8, 4, 0, 8, 4},
        {0, 1, 0, 0, 1, 0} });
    tester.expect_near(A_A_dense.data, A_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check concatenate horizontally.");

    /* 転置 */
    auto AL_T = CL.transpose();
    //std::cout << "AL_T = " << std::endl;
    //for (size_t j = 0; j < AL_T.cols(); ++j) {
    //    for (size_t i = 0; i < AL_T.rows(); ++i) {
    //        std::cout << AL_T.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 4> AL_T_answer({
        {1, 0, 0, 0},
        {3, 0, 8, 1},
        {0, 2, 4, 0}
        });
    tester.expect_near(AL_T.matrix.data, AL_T_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check transpose.");

    auto B_T = B.transpose();
    //std::cout << "B_T = ";
    //for (size_t i = 0; i < B_T.rows(); ++i) {
    //    std::cout << B_T.matrix[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Matrix<DefDiag, T, 3> B_T_answer({ 1, 2, 3 });
    tester.expect_near(B_T.matrix.data, B_T_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check transpose diag.");

    /* LU分解 */
    static auto LU_solver = make_LinalgSolverLU(A);

    auto A_LU = LU_solver.get_L() * LU_solver.get_U();
    //std::cout << "A_LU = " << std::endl;
    //for (size_t j = 0; j < A_LU.cols(); ++j) {
    //    for (size_t i = 0; i < A_LU.rows(); ++i) {
    //        std::cout << A_LU.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 3> A_LU_answer({ {1, 2, 3}, {5, 4, 6}, {9, 8, 7} });
    tester.expect_near(A_LU.matrix.data, A_LU_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverLU L multiply U.");

    //std::cout << "det = " << LU_solver.get_det() << std::endl << std::endl;

    T det_answer = 30;
    tester.expect_near(LU_solver.get_det(), det_answer, NEAR_LIMIT_STRICT,
        "check LinalgSolverLU det.");

    /* コレスキー分解 */
    Matrix<DefDense, T, 3, 3> K({ {10, 1, 2}, {1, 20, 4}, {2, 4, 30} });
    Matrix<DefSparse, T, 3, 3, 6> K_s({ 1, 8, 3, 3, 4 }, { 0, 1, 2, 1, 2 }, { 0, 1, 3, 5 });

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

    /* 転置積 */
    auto AB = AT_mul_B(C, B);

    //std::cout << "AB = " << std::endl;
    //for (size_t j = 0; j < AB.cols(); ++j) {
    //    for (size_t i = 0; i < AB.rows(); ++i) {
    //        std::cout << AB.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 3> AB_answer({ {1, 6, 0}, {0, 0, 6}, {0, 16, 12} });
    tester.expect_near(AB.matrix.data, AB_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check AT_mul_B.");

    auto AB_d = AT_mul_B(B, B);
    //std::cout << "AB_d = ";
    //for (size_t i = 0; i < AB_d.rows(); ++i) {
    //    std::cout << AB_d.matrix[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;

    Matrix<DefDiag, T, 3> AB_d_answer({ 1, 4, 9 });
    tester.expect_near(AB_d.matrix.data, AB_d_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check AT_mul_B diag.");

    /* QR分解 */
    static auto QR_solver = make_LinalgSolverQR(C);
    auto A_QR = QR_solver.get_Q() * QR_solver.get_R();

    //std::cout << "A_QR = " << std::endl;
    //for (size_t j = 0; j < A_QR.cols(); ++j) {
    //    for (size_t i = 0; i < A_QR.rows(); ++i) {
    //        std::cout << A_QR.matrix(j, i) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    Matrix<DefDense, T, 3, 3> A_QR_answer({ {1, 0, 0}, {3, 0, 8}, {0, 2, 4} });
    tester.expect_near(A_QR.matrix.data, A_QR_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinalgSolverQR Q multiply R.");

    /* 実数値のみの固有値 */
    Matrix<DefDense, T, 3, 3> A0({ {6, -3, 5}, {-1, 4, -5}, {-3, 3, -4} });
    //Matrix<DefDense, T, 4, 4> A2({ {11, 8, 5, 10}, {14, 1, 4, 15}, {2, 13, 16, 3}, {7, 12, 9, 6} });
    static auto eig_solver = make_LinalgSolverEigReal(A0);

    auto eigen_values = eig_solver.get_eigen_values();

    //std::cout << "eigen_values = " << std::endl;
    //for (size_t j = 0; j < eigen_values.cols(); ++j) {
    //    for (size_t i = 0; i < eigen_values.rows(); ++i) {
    //        std::cout << eigen_values.matrix(j, i) << " ";
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
    //        std::cout << eigen_vectors.matrix(j, i) << " ";
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
    //        std::cout << "[" << eigen_values_comp.matrix(j, i).real << " ";
    //        std::cout << "+ " <<  eigen_values_comp.matrix(j, i).imag << "j] ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    eig_solver_comp.set_iteration_max(5);
    eig_solver_comp.set_iteration_max_for_eigen_vector(15);

    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_real({ {6}, {-1.5F}, {-1.5F} });
    Matrix<DefDense, T, 3, 1> eigen_values_comp_answer_imag({ {0}, {-0.8660254F}, {0.8660254F} });

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
    //        std::cout << eigen_vectors_comp.matrix(j, i).real << " + " << eigen_vectors_comp.matrix(j, i).imag << "j, ";;
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    std::vector<Complex<T>> eigen_values_answer_comp(3);
    eigen_values_answer_comp[0] = Complex<T>(6, 0);
    eigen_values_answer_comp[1] = Complex<T>(-1.5F, -0.8660254F);
    eigen_values_answer_comp[2] = Complex<T>(-1.5F, 0.8660254F);

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


int main() {

    check_base_matrix_calc<double>();

    check_base_matrix_calc<float>();

    check_python_numpy_calc<double>();

    check_python_numpy_calc<float>();


    return 0;
}
