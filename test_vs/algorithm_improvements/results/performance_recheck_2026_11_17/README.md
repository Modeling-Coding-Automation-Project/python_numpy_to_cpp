# 2026-11-17 実行時間・RAMの再測定

[結果レポート](../../../../document/11_execution_time_and_peak_ram_recheck.md)を参照。実行時間・観測スタック・一部の固有ベクトル品質で悪化例を確認した。本体とサブモジュールは変更していない。

## 比較対象

- `last_timing`：`879a31981d4cb99173939bd4a12df96bf8ccdf1a`。最後の同条件時間測定後。
- `before_fixed`：`ab31b0d`。固定しきい値導入直前。
- `current`：`b599e8461d0a4b07aa007bc6376746744767e671`。調査対象。

各manifest・metadataに完全commitと依存先commitを保存した。全版で数学・utilityのgitlinkが同じ。C++11、g++ 15.2.0、x86-64、`-O2`、固定配列、既定数学。旧記録からの比較元特定と古いRAM記録の限界は[baseline_notes.md](baseline_notes.md)、照合結果は[baseline_audit.json](baseline_audit.json)を参照。

## ファイル

| 場所 | 内容 |
| --- | --- |
| [timing/manifest.json](timing/manifest.json) | 3版・共通ソースhash・flags・CPU・プロセス実行順 |
| [timing/timing_summary.csv](timing/timing_summary.csv) | 既存413設定×比較元2種＝826行の時間比較 |
| [timing/quality_and_storage.csv](timing/quality_and_storage.csv) | 413設定×3版＝1,239行の数値出力・保持量 |
| `timing/raw/` | 5benchmark×3版×7回＝105 CSV |
| [timing/binary_sections.json](timing/binary_sections.json) | ELF `.text` / `.rodata`等のサイズとhash。QR/LU-Cholは前回と現行で一致 |
| `timing/diagnostics/` | GMRES108設定×3版。各1000試行の実反復数・状態・独立残差 |
| [eigen_probe/timing_summary.csv](eigen_probe/timing_summary.csv) | 補助probe144設定×比較元2種＝288行 |
| [eigen_probe/quality_and_storage.csv](eigen_probe/quality_and_storage.csv) | 補助probe144設定×3版＝432行の精度・収束・予算・格納量 |
| `eigen_probe/raw/` | 3版×7回＝21 CSV |
| [eigen_analysis.md](eigen_analysis.md) | 固有値の詳細分析、初期ベクトルの零射影との関係 |
| [eigen_parameter_attribution.cpp](eigen_parameter_attribution.cpp) / [CSV](eigen_parameter_attribution.csv) / [metadata](eigen_parameter_attribution.json) | 8次floatの4パラメータを1個ずつ戻す12行の診断と再現情報 |
| [timing_for/timing_summary.csv](timing_for/timing_summary.csv) | for構成のGMRES・矩形GMRES140設定×比較元2種＝280行 |
| `timing_for/raw/` | 2benchmark×3版×7回＝42 CSV |
| [memory/README.md](memory/README.md) | RAMの全増加例・測定方法・限界 |
| [memory/peak_stack_cases.csv](memory/peak_stack_cases.csv) | 56処理＋校正2種×3版×2構成＝348行 |
| `memory/*.raw.csv`, `*.su`, `*.ci`, `call_chains.json` | 書込み深さ、関数フレーム、最大既知呼出し列と未解決呼出し |
| `*/stack_frames/` | 時間benchmarkの`.su`。品質計算や測定用オブジェクトも含み、ライブラリ単独ピークと扱わない |
| [historical_timing/README.md](historical_timing/README.md) | 回収した前回の4benchmark×前後×5回の記録。今回の再測定とは別 |
| [validation.json](validation.json) | 件数、反復上限、ソースhash、数値出力の安定性、作業ツリーの確認 |

時間は各版7プロセスを直列実行した。CSVの`before_median_ns`・`after_median_ns`はプロセス中央値、`paired_ratio_median`は各測定組の比の中央値。`material_slowdown`は中央値比>1.10かつ全組比>1.05を表す抽出条件であり、誤差保証や有意性検定ではない。小さな差も全件CSVに保存している。

`quality_and_storage.csv`の`metrics`は元のCSVの時間以外の列をJSONとして保存したもの。同じ版の7回で数値・状態・格納量が一致することをrunnerが確認している。GMRESのtarget errorは製造した解との差であり、丸めた悪条件入力の厳密解との前進誤差ではない。

## 再実行

リポジトリrootで実行する。Python 3.12以降、Git、g++、LinuxのCPU affinityが必要。実行時間測定中は他のビルドや重い処理を走らせない。

```sh
python3 test_vs/algorithm_improvements/run_performance_recheck.py \
  --output /tmp/performance_recheck --phase build \
  --benchmarks qr_benchmark lu_chol_benchmark gmres_benchmark gmres_rect_benchmark eigen_benchmark
python3 test_vs/algorithm_improvements/run_performance_recheck.py \
  --output /tmp/performance_recheck --phase run --repeats 7
python3 test_vs/algorithm_improvements/run_performance_recheck.py \
  --output /tmp/performance_recheck --phase diagnostics

python3 test_vs/algorithm_improvements/run_performance_recheck.py \
  --output /tmp/eigen_performance_recheck --benchmarks eigen_performance_probe --repeats 7
python3 test_vs/algorithm_improvements/run_performance_recheck.py \
  --output /tmp/performance_for_recheck --loop-mode for \
  --benchmarks gmres_benchmark gmres_rect_benchmark --repeats 7
```

RAM測定は時間測定の終了後に別途実行する。x86-64/GCC専用のホスト計測コードを使い、本体の動的確保禁止方針とは分けて扱う。

```sh
python3 test_vs/algorithm_improvements/memory/peak_stack_run.py \
  --stage all --output /tmp/peak_stack_results --build-dir /tmp/peak_stack_build
```

RAMの測定方法は[peak_stack_method.md](../../memory/peak_stack_method.md)、時間・精度の意味は[benchmark監査ノート](../../performance_benchmark_audit.md)を参照。バイナリ、展開ヘッダ、測定用pthread stackは結果保存ディレクトリへ含めていない。
