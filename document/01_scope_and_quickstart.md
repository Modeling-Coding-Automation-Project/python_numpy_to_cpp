# 01. 位置づけと最短スタート

## 対象範囲

このドキュメントの対象は次のとおりです。

- 公開インターフェース: `python_numpy/python_numpy.hpp`
- 実装例: `sample/*.cpp`（必要に応じて `sample/*.py` と対比）

対象外:

- `base_matrix` フォルダ内 API の説明
- 各分解・ソルバのアルゴリズム内部

---

## 全体像（利用者目線）

```mermaid
flowchart LR
    A[ユーザーコード sample/*.cpp] --> B[#include \"python_numpy.hpp\"]
    B --> C[行列生成 make_DenseMatrix / make_DiagMatrix / make_SparseMatrix]
    C --> D[演算 + - * transpose concatenate reshape]
    C --> E[線形代数 solve inv cholesky qr eig]
    D --> F[結果表示 rows cols operator()]
    E --> F
```

---

## 最小コード

```cpp
#include "python_numpy.hpp"
#include <iostream>

using namespace PythonNumpy;

int main() {
  auto A = make_DenseMatrix<2, 2>(1.0, 2.0, 3.0, 4.0);
  auto B = make_DiagMatrix<2>(10.0, 20.0);
  auto C = A * B;

  for (std::size_t i = 0; i < C.rows(); ++i) {
    for (std::size_t j = 0; j < C.cols(); ++j) {
      std::cout << C(i, j) << " ";
    }
    std::cout << std::endl;
  }
}
```

---

## サンプル実行の基本手順

GitHub Actions の `run_samples.yml` と同じ流れで、各サンプルは次のように実行できます。

1. Python 側の NumPy サンプルを実行（比較用）
2. 同名の C++ サンプルをコンパイル
3. 実行して結果を確認

```bash
python sample/arithmetic_operation.py
g++ -std=c++11 -Wall ^
  -I ./python_numpy ^
  -I ./base_matrix ^
  -I ./external_libraries/python_math_to_cpp/base_math ^
  -I ./external_libraries/python_math_to_cpp/python_math ^
  -I ./external_libraries/base_utility_cpp/base_utility ^
  -o ./sample/app ./sample/arithmetic_operation.cpp
./sample/app
```

> 注: Windows の PowerShell では行継続の記法が異なるため、必要に応じて 1 行コマンドにしてください。

