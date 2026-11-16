# 行列アルゴリズムの回帰試験と比較

実装・採否は [結果レポート](../../document/07_matrix_algorithm_improvement_results.md)、保存した最終証跡は [results](results/README.md) を参照。

```sh
python3 test_vs/algorithm_improvements/run_validation.py \
  --output /tmp/matrix_algorithm_results --legacy-configs --sanitize --repeats 3
```

基準commitとそのsubmoduleを一時ディレクトリへarchiveし、作業ツリーをcheckoutし直さず比較する。`--baseline` に既存の固定基準ツリーも指定できる。g++、Git、Python 3.12以降を想定する。NumPyを利用する固有値の独立oracleは別コマンド:

```sh
python3 test_vs/algorithm_improvements/eigen_numpy_oracle.py
```

試験ソースの標準数学関数、chrono、メモリ監視、高精度参照はホスト専用。ターゲット実装の依存ではない。`*_benchmark.cpp` は同じソースを改善前後でビルドし、新しいAPIの行だけ `ALGORITHM_IMPROVEMENTS` で切り替える。不採用の試作も比較用コードとして残す。固定配列構成の禁止ヘッダ・ヒープ監視もrunnerが行う。

`<algorithm>` の禁止は本リポジトリのライブラリ実装（`base_matrix/`、`python_numpy/`）をソース検査する。外部リポジトリ `base_utility_cpp` 内での `<algorithm>`、`std::copy` の使用は許容するため、依存先を含む一律のヘッダ禁止にはしない。`<cmath>` は引き続きデフォルト構成の依存先を含め、禁止ヘッダによるビルド検査を行う。`results/` は2026-11-10の実装・条件に対する記録であり、2026-11-11の外部リポジトリ復元後の再計測値ではない。

平方根・逆平方根は数学ライブラリの公開APIに委譲する。現行benchmarkは `QRDecompositionHouseholder<T,M,N>` / `PackedCholesky<T,N>` を使い、クラス側で平方根の反復数や計算方式を選ばない。成分別レポートと `results/` の旧測定値には廃止した精度設定も含まれるため、現行コードを再実行した結果と区別する。

runnerは既定の数学ライブラリ構成に加え、ホスト確認用の `BASE_MATH_USE_STD_MATH_` 構成でも数値回帰試験を実行する。後者は `<cmath>` を使う比較用であり、組み込み構成の代わりにはしない。既定構成の精度assertは依存先の平方根精度を考慮し、標準数学構成では従来の厳密なassertを維持する。構造・零入力・失敗状態の検証は両構成で行う。

`epsilon()`を固定値`1e-6`〜`1e-16`へ置換した比較は、[調査報告](../../document/09_epsilon_parameter_sweep.md)と[測定データ・再実行手順](results/epsilon_parameter_sweep/README.md)を参照。各`run_epsilon_*.py`は比較対象commitと依存先を一時領域へ展開し、製品ヘッダを変更せず測定する。

用途別に収束許容誤差・breakdown・逆反復シフトを分離した追加比較は、[独立パラメータの調査報告](../../document/10_independent_numerical_parameters.md)と[測定データ・再実行手順](results/independent_numerical_parameters/README.md)を参照。各`run_independent_*.py`も一時コピー上で測定する。候補値を既定値にした試作は、元のassertを維持した回帰試験と固定メモリ・禁止ヘッダの検査で確認している。

候補Aを本体に反映した後の作業ツリーは、次のコマンドで検証できる。標準数学doubleの厳密な精度試験は公開メンバーで精度を明示し、既定値での品質試験も別途行う。

```sh
python3 test_vs/algorithm_improvements/run_fixed_parameter_validation.py \
  --output /tmp/fixed_parameter_validation --legacy-configs --sanitize
```

2026-11-17の[採用後の検証結果](results/fixed_parameter_defaults/README.md)に、全32実行の結果と対象ソースのhashを保存した。

前回性能測定時からの実行時間・RAMの変化は、[再評価レポート](../../document/11_execution_time_and_peak_ram_recheck.md)と[全測定データ・再実行手順](results/performance_recheck_2026_11_17/README.md)を参照。既定値変更による時間・精度の悪化例と、テンプレート構成でのスタック増加を確認している。

収束許容誤差の調整と採用値は、[調整結果レポート](../../document/12_convergence_tolerance_tuning.md)と[定数選定・採用後検証](results/balanced_tolerance_selection/README.md)を参照。機械イプシロン由来の値との比較、TMP/forの実命令追跡、精度・時間の改善と悪化、採用後のRAM・24実行の回帰結果を記録した。[初回探索データ](results/convergence_tolerance_tuning/README.md)も保存している。
