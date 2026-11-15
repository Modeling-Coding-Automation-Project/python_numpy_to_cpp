# GMRES反復上限の確認

2026-11-12、[GMRESの改善前後と反復回数の上限](../../../../document/08_gmres_algorithm_changes.md)の根拠資料。C++実装は変更していない。

- `check.cpp` / `check.log`: 現行 `879a319` の既存関数・既定クラス・partition・複数右辺・明示再始動の更新数を確認したソースと結果。通常の16×16行列、同じ右辺を使い、既定設定では最大16以内であることを確認した。
- `iteration_probe.cpp` / `iteration_probe_before.log` / `iteration_probe_after.log`: 前回の計算時間比較と同じ、右辺に交互の微小変更を加える1000試行について、実更新数・状態を数えた結果。比較前は `f50060e`、比較後は `879a319`。`check.cpp` の固定右辺による単発確認とは区別する。
- `gmres_benchmark_before.*.csv` / `gmres_benchmark_after.*.csv`: 前回測定した4種類のbenchmarkのうち、GMRESの生データを抜粋。前後を同じCPUで交互に5プロセスずつ実行した。
- `environment.json`: 上限確認時のソース・構成。
- `timing_environment.json`: 前回の時間測定の構成。比較は `b7faabf` の元のアルゴリズムとの比較ではない。

約52%増加の行は `configured,float,16,0,8,4,1,1`。列の意味は順にmode、型、N、行列種別、K、C、P、対角前処理の有無であり、両版とも `K×C=32` が上限。5プロセスの時間中央値は5.402271 μs→8.195366 μs、平均更新数は15.5→24だった。

再実行例（リポジトリのルートから）:

```sh
g++ -std=c++11 -O2 -Wall -Wextra -Werror \
  -Ibase_matrix \
  -Iexternal_libraries/python_math_to_cpp/base_math \
  -Iexternal_libraries/python_math_to_cpp/python_math \
  -Iexternal_libraries/base_utility_cpp/base_utility \
  test_vs/algorithm_improvements/results/2026-11-12_gmres_iterations/check.cpp \
  -o /tmp/gmres_iteration_check
/tmp/gmres_iteration_check
```

`iteration_probe.cpp` も同じinclude条件でビルドできる。旧版との比較時は、`git archive f50060e base_matrix` で別ディレクトリに取り出したヘッダを使用し、作業ツリーを戻す必要はない。検証ソースの `<cmath>`・`<chrono>` はホスト専用。
