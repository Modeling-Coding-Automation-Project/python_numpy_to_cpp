# 最終統合評価の証跡

[結果レポート](../../../document/07_matrix_algorithm_improvement_results.md)に対応する2026-11-10のホスト評価記録。2026-11-11の外部utility復元と平方根の公開APIへの委譲・クラス側精度設定廃止より前の証跡であり、現行コードの再計測値ではない。ソースハッシュとCSVは当時のまま保持する。2026-11-11の変更後の検証は [2026-11-11_public_roots](2026-11-11_public_roots/README.md) に分けて保存する。

- `environment.json`: 基準commit、submodule、コンパイラ、ホスト、固定した論理CPU、検証の構成。
- `source_hashes.json`: 評価した実装・試験ソースのSHA-256。レポートだけの編集は含めない。
- `*_benchmark_before.0.csv`〜`.2.csv` / `*_benchmark_after.0.csv`〜`.2.csv`: 同一CPUで改善前後を交互に実行した3プロセスの結果。QRはプロセス内3組の最小平均、eigenは5組の最小平均、LU/Choleskyは5組の中央値、GMRESは1000回の平均。中央レポートのGMRES時間は3プロセスの中央値。
- `*_benchmark_*.su`: g++の関数ごとのスタックフレーム。呼出し側のオブジェクト、出力、同時に存在する呼出し先を別途評価する。単体の値は最大RAMではない。
- `gmres_oracle.csv`: 正確に格納されたA/bに対するDecimal80桁の真値比較。120桁での一致も確認。
- `gmres_before_exact_hex.csv` / `gmres_after_exact_hex.csv`: oracleへ渡したbinary32/64の入力と結果。
- `*_regression*.log`: 回帰試験とsanitizerの結果。
- `legacy_0.log`〜`legacy_7.log`: 既存CIの8マクロ構成。既存テスタは成功時に無出力のため空ファイルとなる。全プロセスの終了コードは0。
- `constraints_*.log` / `utility_audit_*.log`: 禁止ヘッダ・例外無効・確保監視、utility4構成。確保監視は全て0回。
- `eigen_numpy_oracle.log`: 独立したNumPy参照との45例比較。

GMRESの性能CSVの `rms_target_error` は、右辺の生成に使ったonesからの距離。丸め後のA/bに対する解誤差は `gmres_oracle.csv` の `relative_solution_error` を使う。Hilbertでは両者が大きく異なる。

`configured` のGMRES時間は構築済みworkspaceを再利用し、前処理の準備を含まない。`legacy` は関数内部でworkspaceを構築する。矩形版の `explicit` は比較用のGram構築＋平方GMRES、`rect_api` は実装API、`matrix_free` は任意の演算子経路。

現行APIでの再実行（廃止した平方根精度設定の再現ではない）:

```sh
python3 test_vs/algorithm_improvements/run_validation.py \
  --output /tmp/matrix_algorithm_results --legacy-configs --sanitize --repeats 3
python3 test_vs/algorithm_improvements/eigen_numpy_oracle.py
```

NumPyは2行目だけに必要。Decimal oracleは標準Pythonで動作し、runnerが実行する。成分別の `qr_results.md` / `eigen_results.md` / `lu_chol_results.md` は各試作を評価した際の値であり、この最終統合測定と時間が一致するとは限らない。実機MCUでの測定値ではない。
