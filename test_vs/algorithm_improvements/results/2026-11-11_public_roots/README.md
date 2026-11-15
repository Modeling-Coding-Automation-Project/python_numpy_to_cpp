# 平方根の公開APIへの委譲後の検証

2026-11-11の変更後に取得した記録。[結果レポート9節](../../../../document/07_matrix_algorithm_improvement_results.md#9-2026-11-11の平方根委譲方針への変更)に対応する。親ディレクトリの2026-11-10の記録は更新していない。

- `environment.json` / `source_hashes.json`: 構成と検証したソース。行列側で平方根の方式・精度を指定せず、外部数学ライブラリの公開APIを使用する。
- `legacy_results.json` / `legacy_*.log`: C++11の既存8構成。成功時は無出力。
- `*_regression_default.log` / `*_regression_std.log`: 数値回帰4種類の既定数学構成とホスト標準数学構成。標準数学構成は精度と委譲の検証用で、組み込み構成ではない。
- `*_regression_sanitized.log`: 5試験のASan・UBSan・float-divide-by-zero検証。GMRESは最終版の厳密な解・残差の許容値を維持した試験結果。
- `constraints_default.log` / `constraints_std_copy.log` / `constraints_memcpy.log` / `constraints_both.log`: 固定配列APIの動的確保は全構成0回。依存先を含む `<cmath>` 禁止と `-fno-exceptions` を併用。外部utilityの `<algorithm>` は許可。
- `eigen_numpy_policy.log`: NumPyとの独立比較。両数学構成で45例すべて収束。誤差は固有値の最適対応による最大絶対差を `max(1, ||A||₂)` で割った値。
- `precision_probe.cpp` / `precision_probe_final.log`: 平方根、GMRES、固有値の精度確認。GMRESの解誤差と残差は絶対L2ノルム、固有値は最大絶対誤差。
- `qr_benchmark.log` / `lu_chol_benchmark.log`: 現行APIの精度確認に使用したホスト測定。今回、時間の同条件比較を完了した証拠としては使わない。
- `gmres_oracle.csv` / `*_exact_hex.csv` / `gmres_oracle_metadata.json`: 格納された入力に対するDecimal80桁・120桁の独立参照。252比較（旧18、現行234）、18入力。全結果が非有限値を含まないことを確認したが、全設定で高精度解が得られたという意味ではない。

float Hilbert8では現行の全12設定が旧版の相対解誤差を改善せず、旧版約0.746に対して現行最良約0.991。収束状態は使用型で再計算した残差に基づき、実数の厳密な残差ゼロや真値との一致を保証しない。

再実行方法は[試験README](../../README.md)を参照。既定数学構成ではQR・固有値・ノルム／回転の直接評価に数学ライブラリの近似誤差を含め、標準数学構成では従来の厳密な許容値を維持する。GMRESの解・残差の許容値は両構成で従来どおり。
