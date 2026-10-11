# 05. Python NumPy サンプルとの対応表

この章は `sample/*.py` と `sample/*.cpp` の対応を一覧化したものです。  
「どの NumPy 記述をどの C++ API に置き換えるか」を素早く確認できます。

## 5.1 ファイル対応

| Python サンプル | C++ サンプル | 主題 |
|---|---|---|
| `arithmetic_operation.py` | `arithmetic_operation.cpp` | 行列生成・積・転置・ゼロ/1/定数埋め |
| `concatenation.py` | `concatenation.cpp` | 縦横連結、block、tile |
| `matrix_action.py` | `matrix_action.cpp` | reshape、部分置換、norm、要素積 |
| `left_divide.py` | `left_divide.cpp` | 連立方程式 `solve` |
| `inverse.py` | `inverse.cpp` | 逆行列 `inv` |
| `cholesky.py` | `cholesky.cpp` | Cholesky 分解 |
| `qr.py` | `qr.cpp` | QR 分解 |
| `eigen_values_vectors_real.py` | `eigen_values_vectors_real.cpp` | 実固有値・固有ベクトル |
| `eigen_values_vectors_complex.py` | `eigen_values_vectors_complex.cpp` | 複素固有値・固有ベクトル |
| `trigonometric.py` | `trigonometric.cpp` | 行列に対する `sin` |
| （なし） | `augmented_matrix.cpp` | AugmentedMatrix の C++ 専用例 |

---

## 5.2 API 対応（抜粋）

| NumPy | PythonNumpy (C++) |
|---|---|
| `np.array(...)` | `make_DenseMatrix<M,N>(...)` |
| `np.diag([...])` | `make_DiagMatrix<M>(...)` |
| 手書き疎行列 | `SparseAvailable` + `make_SparseMatrix<...>(...)` |
| `A @ B` | `A * B` |
| `A.T` | `A.transpose()` |
| `A.T @ B` | `ATranspose_mul_B(A, B)` |
| `A @ B.T` | `A_mul_BTranspose(A, B)` |
| `np.concatenate([A,B], axis=0)` | `concatenate_vertically(A, B)` |
| `np.concatenate([A,B], axis=1)` | `concatenate_horizontally(A, B)` |
| `np.block([[A,B],[C,D]])` | `concatenate_block<2,2>(A, B, C, D)` |
| `np.tile(C, (2,3))` | `concatenate_tile<2,3>(C)` |
| `A.reshape(8,2,order='F')` | `reshape<8,2>(A)` |
| `A[1:4,1:4] = C` | `substitute_part_matrix<1,1>(A, C)` |
| `np.linalg.norm(C)` | `norm(C)` |
| `E = C * D`（要素積） | `element_wise_multiply(E, C, D)` |
| `np.linalg.solve(A,C)` | `make_LinalgSolver<...>().solve(A,C)` |
| `np.linalg.inv(A)` | `make_LinalgSolverInv<...>().inv(A)` |
| `np.linalg.cholesky(A)` | `make_LinalgSolverCholesky<...>().solve(A)` |
| `np.linalg.qr(A)` | `make_LinalgSolverQR<...>()` + `solve/get_Q/get_R` |
| `np.linalg.eig(A)` | `make_LinalgSolverEig*<...>()` + `solve_eigen_*/get_*` |
| `np.sin(2*np.pi*time)` | `sin(2.0 * PI * time)` |

---

## 5.3 `numpy_deploy.py` の使いどころ

`sample/arithmetic_operation.py` では `NumpyDeploy.generate_matrix_cpp_code(...)` を使って、NumPy 配列を C++ 行列定義コードへ変換できます。

- Dense / Diag / Sparse を自動判定
- 生成コードは `python_numpy.hpp` の API 形式（`DenseMatrix_Type` など）で出力
- Python 側で作った行列定義を C++ 側へ移す初期作業を減らせる

