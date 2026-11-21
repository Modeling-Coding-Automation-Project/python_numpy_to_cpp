# python_numpy ドキュメント

NumPy のコードを C++ に移すときは「使い方とサンプル」を、ソルバがどのように答えを求めるかを調べるときは「線形代数アルゴリズム」を参照してください。

## 使い方とサンプル

1. [位置づけと最短スタート](./01_scope_and_quickstart.md) — ヘッダーの読み込みからサンプルの実行まで
2. [行列の定義と生成](./02_matrix_definition.md) — 密行列・対角行列・疎行列の作り方
3. [行列操作](./03_operations_and_transformations.md) — 積、転置、連結、reshape、置換、ノルム、数学関数、畳み込み
4. [線形代数ソルバの使い方](./04_linear_algebra_samples.md) — 連立一次方程式、逆行列、コレスキー分解、QR 分解、固有値・固有ベクトル
5. [Python と C++ のサンプル対応表](./05_python_cpp_sample_mapping.md) — NumPy の式に対応する C++ API とコード例

## 線形代数アルゴリズム

| 解説 | 実装で行う計算 |
| --- | --- |
| [GMRES 法](./detail/01_gmres.md) | 連立一次方程式の残差を小さくする。探索方向の作り方と、反復を終える条件を追う。 |
| [LU 分解](./detail/02_lu.md) | 行を入れ替えながら消去し、下三角行列と上三角行列を得る。行交換は行列式の符号にも反映される。 |
| [QR 分解](./detail/03_qr.md) | ギブンス回転で対角線の下の要素を消し、直交行列と上三角部分に分ける。 |
| [固有値](./detail/04_eigenvalues.md) | 一般の実行列をヘッセンベルグ形に変換し、シフト付き QR 反復で固有値を求める。 |
| [固有ベクトル](./detail/05_eigenvectors.md) | 求めた固有値ごとに逆反復を行う。得られたベクトルが固有値に対応するかを残差で確かめる。 |

## 実装と依存ライブラリ

公開 API の入口は [python_numpy.hpp](../python_numpy/python_numpy.hpp) です。計算手順は [base_matrix](../base_matrix/) に、呼び出し方の例は [sample](../sample/) にあります。

| サブモジュール | 役割 |
| --- | --- |
| [base_utility_cpp](../external_libraries/base_utility_cpp/) | ゼロ割回避、ゼロ近傍の判定、配列コピーなどの共通処理 |
| [python_math_to_cpp](../external_libraries/python_math_to_cpp/) | 平方根、三角関数、指数関数などの数値計算 |
