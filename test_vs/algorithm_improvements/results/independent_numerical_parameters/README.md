# 用途別の独立パラメータの調査データ

[結果と運用設計の調査報告](../../../../document/10_independent_numerical_parameters.md)を参照。

対象は commit `8e22c7b327fa4d783a0db33527cc25b5d941b8ba`。サブモジュールは `python_math_to_cpp=2a2c19b91e9dfe69de532abebbcbdd25f8fbbd5e`、`base_utility_cpp=6d4807de5eb5bfbad8bff950caf3fd20757a9b30`。ヘッダを一時領域へ展開して測定し、本体・依存先の作業ツリーは変更していない。

前回の[共通固定値の調査](../epsilon_parameter_sweep/README.md)に対し、今回は GMRES breakdown 係数、固有値・固有ベクトルの許容誤差、逆反復シフト、内部 GMRES 許容誤差を分離した。係数は epsilon の乗数ではなく、判定に直接使用する最終値である。

## 保存ファイル

| ファイル | 内容 |
| --- | --- |
| [gmres_cases.csv](gmres_cases.csv) | 672件の前進誤差・実残差・反復数・収束・breakdown |
| [gmres_summary.csv](gmres_summary.csv) | 数学設定・型・β・τ・行列集合別の集計 |
| [gmres_exact_inputs.json](gmres_exact_inputs.json) | 参照解計算に使う格納入力・出力の16進表現 |
| [gmres_metadata.json](gmres_metadata.json) | commit、ビルド条件、ソース hash、for版との一致確認 |
| [gmres_notes.md](gmres_notes.md) | 条件・解釈・代表的な比較 |
| [eigen_metrics.csv](eigen_metrics.csv) | 11,616件の固有値誤差・固有対残差・収束・QR sweep・格納量 |
| [eigen_summary.csv](eigen_summary.csv) | 132設定 × 数学設定2種 × 型2種の528行の集計 |
| [eigen_reference_summary.csv](eigen_reference_summary.csv) | 前回の machine / 共通1e-7を同じ固定品質ゲートで再集計 |
| `eigen_*.raw.csv` | 入力・固有値・固有ベクトルの16進出力。基本108組と追加24組に分かれる |
| [eigen_metadata.json](eigen_metadata.json) | 全グリッド、追加設定、commit、NumPy版、ソース hash |
| [eigen_notes.md](eigen_notes.md) | 係数ごとの比較、品質ゲート、測定の限界 |
| [prototype_validation/summary.csv](prototype_validation/summary.csv) | 候補Aを既定値へ適用した一時試作の既存回帰試験6実行 |
| [prototype_validation/metadata.json](prototype_validation/metadata.json) | 一時試作の固定値・commit・ビルド条件 |
| `prototype_validation/*.txt` | ビルド・試験ログ。assertは元のまま |
| [validation.json](validation.json) | 件数、有限値、反復上限、格納量、ソース hash の検査 |

`default` は既定数学、`std` はホスト専用の標準数学比較。いずれも float/double、C++11、g++ 15.2.0、`-O2 -Wall -Wextra -Werror`。マイコンの実時間は測定していない。

GMRES は相対前進誤差・相対実残差を独立計算した。通常集合の精度集計は float `1e-6`、double `1e-12` 以下を正解基準にする。固有値側の品質ゲートは正規化固有値誤差と固有対残差がともに `2e-6` 以下、std double だけ `1e-10` 以下。`quality_cases` は品質だけ、`usable_cases` はさらに収束状態も満たした件数。ゲートは試験対象の許容誤差に連動させていない。

CSV の `requested_*` は指定した実数係数、`relative_tolerance`・`vector_tolerance`・`shift` は型変換後の格納値。Jacobi のベクトル専用フラグ・QR更新数は非該当の `-1`。意図的に厳しい設定・不適切な設定も含むため、全ケースの収束・品質が成立しているという意味ではない。

## 再実行

リポジトリのルートで実行する。Git、g++、Python 3.12以降が必要。固有値スクリプトは NumPy も使用する（保存結果は2.5.4）。依存する参照処理は同じディレクトリの `run_epsilon_*.py`、固有値の基準比較は保存済みの前回調査データを使用する。

```sh
python3 test_vs/algorithm_improvements/run_independent_gmres_parameters.py \
  --output /tmp/independent_parameters
python3 test_vs/algorithm_improvements/run_independent_eigen_parameters.py \
  --output /tmp/independent_parameters
python3 test_vs/algorithm_improvements/run_independent_parameter_validation.py \
  --output /tmp/independent_parameters/prototype_validation
```

`--revision` は上記 commit が既定値で、対応する Git オブジェクトが必要。固有値の基準比較データも同じ commit であることを検証するため、別 revision では比較データの準備も必要になる。

固有値のフル実行は6バイナリで全132設定を測定する。保存済みデータは基本グリッド実行後に `--extras-only` で24設定を追加したため、生出力が12ファイルに分かれる。フル実行でも同じ11,616件を得る。出力先を `/tmp` にすることで保存結果の上書きを避けられる。
