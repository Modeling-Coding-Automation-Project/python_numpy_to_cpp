# 呼出し先を含めたRAM比較の方法

`peak_stack_run.py` は、ライブラリ本体を変更せず、各Gitコミットとそのコミットが参照するサブモジュールを一時ディレクトリへ展開して比較する。対象はLinux x86-64、GCC、C++11、`-O2`、固定配列、既定の数学ライブラリ構成。組み込み実機のRAM上限を直接求める測定ではない。

## 対象と保存単位

`float` / `double`、N=8 / 16 の各組合せで次の14処理を測る。入力は同一の決定的な対称正定値行列で、測定領域の外で生成する。

- GMRES: K=N・1 cycle、およびK=N/2・4 cycles。直交化1回、収束許容値は各版の既定値。
- 一般固有値: 実数版／複素数版の固有値のみ、および固有値＋固有ベクトル。QR上限64、逆反復上限8。
- Jacobi: 値のみのpacked版、ベクトル保存版。8 sweeps。
- QR: Givens、Householder因子のみ、Householder＋明示的Q生成。
- LU: 分解＋右辺1本の解法。
- Cholesky: dense出力先API、packed因子API。

一般固有値の複素数版も対称正定値入力であり、複素固有値を持つ全分岐の網羅試験ではない。入力依存の実測と、全到達可能呼出しを調べる静的補助指標を分ける。

主比較は最後の同条件時間測定後の `879a31981d4cb99173939bd4a12df96bf8ccdf1a` と `b599e84`。`ab31` は定数化直前の変化の切り分けに使う。**2026-11-10の保存済み `.su` は、この879a319版の再測定値ではない。** 親ディレクトリの `results/environment.json` に記載された、さらに以前の比較・当時のworking treeに属する。新しい比較はこのprobeを各版へ同じ条件で再コンパイルした値だけで行う。

## 三つのRAM指標

| 列 | 含むもの | 含まないもの／注意点 |
| --- | --- | --- |
| `solver_bytes` | `sizeof(solver)` | 入力、出力、一時変数、呼出し先のstack |
| `context_bytes` | 呼出し元が保持する入力、出力、solver、padding | 演算中だけの一時stack。`solver_bytes`はすでに含まれる |
| `stack_written_max` | 演算入口から最深の書込み位置まで。呼出し先、戻り番地、leafのred zoneを含む | 書き込まれなかった予約領域は検出できない。最大6回の観測値で、全入力の上限ではない |
| `known_call_chain_bytes` | GCC call graphの入口から到達可能な最大frame合計 | 解析不能な外部関数・間接呼出し・無界frameがあれば完全な上限にはならない |
| `known_chain_plus_red_zone` | 上記frame合計＋128 byte | x86-64 leafのred zoneを保守的に加えた補助指標 |

`context_plus_written_max` は `context_bytes + stack_written_max`、`context_plus_known_chain` は `context_bytes + known_chain_plus_red_zone`。solverをもう一度加算しない。対象となる1演算が呼出し元に必要とする保存領域と一時stackを比較する指標であり、ホストプロセスのRSS、pthread管理領域、測定用stack全2 MiB、アプリケーションの他の変数を足した値ではない。

## 動的測定: 専用stackを演算の直前に塗る

1. `mmap`で専用2 MiB stackと両端のguard pageを用意し、`pthread_attr_setstack`へ渡す。pthreadの初期化後、workerが現在のRSPを読む。stackを利用者が指定すると通常のguard設定は適用されないため、guardもprobeが確保する。[Linuxのpthread_attr_setstack仕様](https://man7.org/linux/man-pages/man3/pthread_attr_setstack.3.html)
2. workerの翻訳単位だけを `-mno-red-zone -fno-shrink-wrap -fno-shrink-wrap-separate` でコンパイルする。RSPを読む前にworkerのframeを確保させ、RSPより下の未使用領域をvolatile byteで塗り、関数呼出しやmemsetを挟まずに演算を呼ぶ。演算側の翻訳単位は通常のコンパイル条件のまま。x86-64のred zoneはRSPの下128 byteなので、演算側がそこを使った場合も測定領域に入る。[GCCのred zone説明](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html)
3. 演算終了直後、他の関数を呼ぶ前に最深の変更byteを調べる。worker自身、pthread起動・終了処理、入力生成、printfはこの区間に入らない。RSPが戻ったこととstack下端に到達していないことも確認する。
4. `0xa5` / `0x5a` の2パターンを各3回実行し、最大値と最小値を残す。パターンとデータの偶然の一致による過小評価を減らす。空関数の8 byteと、外側256 byte＋呼出し先4096 byteの書込みをcalibrationとして毎バイナリ実行する。

演算入口の`noinline`境界は全版共通。ライブラリ内部のinline・最適化は阻害しない。演算内へ関数トレース用hookを挿入しない。warm-upと即時bindingで動的リンクの初回処理を除き、子threadへ引き継ぐsignal maskで非同期handlerによる混入も抑える。workerの逆アセンブルを保存し、paintからscanまで演算以外のcallが入っていないことを確認できる。

この方式は**書込み水位**である。未使用の予約領域、別入力だけで通る分岐、OS割込みstack、別アーキテクチャのABIは保証しない。測定用バッファの確保やpthreadはホスト専用で、本ライブラリの動的確保禁止方針を変更しない。

## 静的補助: `.su`の単一行で終わらせない

同じ演算objectへ `-fstack-usage -fcallgraph-info=su` を付ける。GCCはVCG形式の関数間call graphへstack使用量を付加できる。[GCCのcall graphオプション](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html)

各`peak_*`入口から到達する関数をたどり、「入口frame＋最も深い呼出し先frame列」を求める。反復数をstackへ掛けない。分岐先の同時成立を仮定するため保守的になり、tail callの扱いでも実際より多くなる場合がある。leafのred zoneに備えて最後に128 byteを加える。

`call_graph_complete=1` は到達先のframe値がすべて取得でき、再帰cycleや無界の動的frameがない場合だけ付く。`unknown_calls` と `unbounded_frames` は省略せずCSVへ保存する。解析不能な外部callがある場合、既知frameの合計は全体の上限とは呼ばない。完全なgraphで観測水位が静的補助値を超えた場合はrunnerを失敗させる。`call_chains.json` に最大frame列と各frameの元情報を残す。

## 再実行

リポジトリrootで以下を実行する。時間測定と同時にビルド・実行しない。

```sh
python3 test_vs/algorithm_improvements/memory/peak_stack_run.py --stage compile
python3 test_vs/algorithm_improvements/memory/peak_stack_run.py --stage measure
python3 test_vs/algorithm_improvements/memory/peak_stack_run.py --stage analyze
```

`--loop-modes template` で主構成だけ、既定の `template,for` でfor構成も比較できる。`--revisions label=commit ...`、`--output DIR`、`--build-dir DIR`で測定対象と保存先を指定する。`--stage all` は3段階を順に実行する。

保存先の `metadata.json` には完全commit、依存commit、コンパイラ、flags、probeのhashを記録する。`.raw.csv` はpaintの原記録、`.su` / `.ci` はコンパイラの原記録、`peak_stack_cases.csv` は統合値、`peak_stack_comparison.csv` は最初のrevisionに対する差分である。ビルド生成物は指定した一時buildディレクトリへ保存し、ライブラリのworking treeを変更しない。
