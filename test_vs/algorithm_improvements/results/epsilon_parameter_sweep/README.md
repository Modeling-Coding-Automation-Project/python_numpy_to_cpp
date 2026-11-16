# 固定しきい値基準の調査データ

[比較・解釈・定数化の設計案](../../../../document/09_epsilon_parameter_sweep.md)を参照。

対象commitは`8e22c7b327fa4d783a0db33527cc25b5d941b8ba`。機械epsilonを使う現状と、固定`1e-6`〜`1e-16`を比較した。C++11、g++ 15.2.0、`-O2 -Wall -Wextra -Werror`、固定配列・テンプレート展開の構成。`default`は既定数学、`std`はホストでの標準数学比較。マイコンの実時間は測定していない。

依存先は比較対象commitのgitlinkに合わせる。

- `python_math_to_cpp`: `2a2c19b91e9dfe69de532abebbcbdd25f8fbbd5e`
- `base_utility_cpp`: `6d4807de5eb5bfbad8bff950caf3fd20757a9b30`

## ファイル

| ファイル | 内容 |
| --- | --- |
| [gmres_cases.csv](gmres_cases.csv) | GMRES 1536件の前進誤差・実残差・反復数・収束状態 |
| [gmres_summary.csv](gmres_summary.csv) | full K／許容誤差0／再始動／near-breakdown別集計、現状比の改善・悪化件数 |
| [gmres_exact_inputs.json](gmres_exact_inputs.json) | 行列・右辺・出力の16進数による正確な格納値 |
| [gmres_metadata.json](gmres_metadata.json) | コンパイラ・commit・ソースhash・Decimal参照計算の条件 |
| [eigen_metrics.csv](eigen_metrics.csv) | 固有値1056件の誤差・ベクトル残差・収束状態・QR更新数 |
| [eigen_summary.csv](eigen_summary.csv) | 型・数学構成・固定値別集計 |
| `eigen_*.raw.csv` | 固有値probeの行列・固有値・固有ベクトルの16進出力 |
| [eigen_metadata.json](eigen_metadata.json) | NumPy版・入力集合・反復上限・ソースhash |
| [eigen_roles/eigen_roles_metrics.csv](eigen_roles/eigen_roles_metrics.csv) | 1用途だけを1e-6／1e-7に変更した440件 |
| `eigen_roles/*.raw.csv` | 用途別変更の16進出力 |
| [eigen_roles/eigen_roles_metadata.json](eigen_roles/eigen_roles_metadata.json) | 用途別変更の置換条件 |
| [regression_summary.csv](regression_summary.csv) | 既存試験の型別失敗数。assertや許容誤差を変更していない |
| `regression_logs/*.txt` | 既存試験の型別診断出力 |
| [regression_metadata.json](regression_metadata.json) | 回帰試験の元ヘッダhash・コンパイラ等 |
| [validation.json](validation.json) | 件数・有限値・反復上限・格納量と、固定revisionの同一性検査 |

機械epsilonの行はGMRESデータでは`baseline`、Eigen／回帰試験データでは`machine`と表記する。どちらも同じ比較元である。Jacobiには独立したベクトル収束状態・実QR更新数のAPIがないため、該当フィールドは`-1`。最大残差には未収束の出力も含む。

## 再実行

リポジトリのルートで実行する。Python 3.12以降、Git、g++が必要。Eigenの2スクリプトはNumPyも必要で、保存結果はNumPy 2.5.4で評価した。数学構成・丸め・コンパイラが異なれば数値結果は変わり得る。

```sh
python3 test_vs/algorithm_improvements/run_epsilon_gmres_sweep.py \
  --output /tmp/epsilon_sweep_reproduction
python3 test_vs/algorithm_improvements/run_epsilon_eigen_sweep.py \
  --output /tmp/epsilon_sweep_reproduction
python3 test_vs/algorithm_improvements/run_epsilon_eigen_roles.py \
  --output /tmp/epsilon_sweep_reproduction/eigen_roles
python3 test_vs/algorithm_improvements/run_epsilon_regression_sweep.py \
  --output /tmp/epsilon_sweep_reproduction
```

各スクリプトの`--revision`の既定値は上記commit。対応する本体・サブモジュールのGitオブジェクトがローカルに必要となる。本体ヘッダと依存先を一時領域へ展開し、製品ファイル・submodule checkoutを変更しない。保存結果を上書きしないよう、上記では出力先を`/tmp`に指定している。

初回の測定は当該commitと一致する作業ツリーの隔離コピーを用いた。その後、固定revisionをarchiveする再現処理を追加し、ヘッダの同一性と代表条件の出力一致を確認した。再実行手順の整備によって既存測定値を置き換えてはいない。

GMRESの参照解は16進出力から復元した厳密な格納入力をDecimalで80桁計算し、120桁計算と照合。Eigenの固有値参照と独立残差は、同じ格納入力をNumPyのbinary64で評価している。収束フラグは入力したしきい値に対する判定であり、独立な解誤差とは別項目として扱う。
