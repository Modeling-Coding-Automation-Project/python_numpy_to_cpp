# python_numpy 機能ドキュメント（サンプル中心）

このドキュメント群は、本リポジトリの **`python_numpy` インターフェースの使い方**を、`sample` のコードをベースに説明したものです。
`base_matrix` の内部インターフェースやアルゴリズム詳細は扱いません。

## 目次

1. [01: 位置づけと最短スタート](./01_scope_and_quickstart.md)
2. [02: 行列の定義と生成（Dense/Diag/Sparse）](./02_matrix_definition.md)
3. [03: 行列操作（積・転置・連結・reshape・置換・ノルム）](./03_operations_and_transformations.md)
4. [04: 線形代数ソルバ（solve/inv/cholesky/qr/eig）](./04_linear_algebra_samples.md)
5. [05: Python NumPy サンプルとの対応表](./05_python_cpp_sample_mapping.md)

## このドキュメントで重視していること

- `sample/*.cpp` に書かれている実装パターン
- どの API をどう呼び出すか
- NumPy から C++ へ置き換えるときの対応関係

## 依存ライブラリの要点（本リポジトリ内で完結）

本ライブラリは外部サブモジュールも利用しますが、使い方の観点では次を押さえれば十分です。

| ライブラリ | 役割（要約） |
|---|---|
| `external_libraries/base_utility_cpp` | ゼロ割回避、近傍ゼロ判定、配列コピーなどの基礎ユーティリティ |
| `external_libraries/python_math_to_cpp` | `sin`/`cos`/`exp` など数学関数を C++ で NumPy 風に扱うための基盤 |

