# Balanced Eigen default selection artifacts

推薦は float/double 共通 `rtol=1e-6, vtol=4e-6, inner=1e-3`。主比較では β=`8e-12`、shift=`3e-4` と予算を固定した。品質・時間の改善と悪化は [balanced_eigen_notes.md](balanced_eigen_notes.md) に記載している。

- `balanced_eigen_grid_*`: 固定値 24 組＋旧 epsilon 参照 2 組。各 140 ケース。
- `balanced_eigen_nearby_*`: 近傍 6 組。各 140 ケース。
- `balanced_eigen_selection.json`: 新次数検証前に保存した推薦と、最終決定の追記。元の選定日時・hash を残している。
- `balanced_eigen_validation_*`: 新次数 N6/10/14。A・推薦・前回候補・旧参照 2 組、各 48 ケース。
- `*_raw/*.csv`: C++ の出力。入力・固有値・固有ベクトルは 16 進浮動小数点表記で保存。
- `*_metrics.csv`: 独立した NumPy 照合、外部残差、収束フラグ、実際のパラメータを含む。
- `*_summary.csv`: 型別/両型のケース集計と、同じ入力への複数 API を重複計数しない集計。
- `*_metadata.json`: revision、snapshot hash、依存 Gitlink、probe/runner/binary hash、コンパイラ、旧 β の一時置換。
- `balanced_eigen_selected_changes.csv`: A/推薦の全 188 ケース比較。
- `balanced_eigen_comparison.json`: 両方成功・新成功・新失敗の数。
- `timing/`: 同一バイナリで A と inner 候補 3 組、7 process ずつの CPU 固定直列計時。品質試験との状態照合を含む。
- `timing/balanced_eigen_timing_groups.csv`: 成功/失敗集合を分けた計時集計。
- `timing/balanced_eigen_inner_comparison.csv`: inner=1e-3 と 1e-4/2e-3 の直接比較。

本体は `6b1d627edf2de6e3e56deca59b83b37e942a97aa` から固定している。以前の探索 92＋holdout 48 を今回すべて選定に使用し、新次数を後で評価した。C++ の本体はこの probe/runner では変更していない。旧全係数比較の β だけ、隔離 snapshot のヘッダを一時置換・コンパイル後に復元した。

再現はリポジトリルートから、NumPy が利用できる Python で行う。ここでは前回と同じ `/tmp/eigen_oracle_venv/bin/python` を使った。既存保存結果を上書きしたくない場合、`--output` で別ディレクトリを指定する。ただし timing は品質データを本 README のディレクトリから読む。

```sh
/tmp/eigen_oracle_venv/bin/python test_vs/algorithm_improvements/balanced_eigen_sweep.py
/tmp/eigen_oracle_venv/bin/python test_vs/algorithm_improvements/balanced_eigen_sweep.py --label nearby --profiles test_vs/algorithm_improvements/results/balanced_tolerance_selection/eigen/balanced_eigen_nearby_profiles.json --reuse-build
/tmp/eigen_oracle_venv/bin/python test_vs/algorithm_improvements/balanced_eigen_sweep.py --label validation --validation --profiles test_vs/algorithm_improvements/results/balanced_tolerance_selection/eigen/balanced_eigen_validation_profiles.json
/tmp/eigen_oracle_venv/bin/python test_vs/algorithm_improvements/balanced_eigen_timing.py
python3 test_vs/algorithm_improvements/balanced_eigen_analysis.py
```

`--work` を再利用する場合も `verify_snapshot` が固定 Git tree の全ヘッダと照合する。通常 build の `--reuse-build` は revision・snapshot digest・probe hash・compiler flags の sidecar も検査する。今回の旧 β バイナリは grid と validation の初回コンパイルで作り、旧 β の `--reuse-build` は実施していない。

品質 probe の引数は `rtol vtol inner reference_mode`。mode 0 は固定係数、1 は旧許容誤差だけ、2 は旧 epsilon 関連全係数。mode 2 には別に旧 β をコンパイルしたバイナリを使用する必要があり、runner が選ぶ。`BALANCED_EIGEN_VALIDATION` は新次数入力へ切り替える。

計時 probe の引数は `profile [repetitions]`。profile 0=A、1=inner1e-4、2=inner1e-3、3=inner2e-3。候補の rtol/vtol は共通で `1e-6/4e-6`。同一 binary、constructor 込み、7 回の process 順を回転・反転する。全 140 ケースの数値状態が先の品質 probe と一致することを runner で検査する。計時に新次数 48 ケースは含まない。

保存元 runner/probe の hash は各 metadata の値が測定時の正本。後続の集計用 `balanced_eigen_analysis.py` は raw/metrics から集計だけを再生成し、再コンパイルや計時は行わない。ホスト専用の probe であり、`<limits>`、動的な Python データ、表示・時計は本体の組み込み実装制約に含めない。
