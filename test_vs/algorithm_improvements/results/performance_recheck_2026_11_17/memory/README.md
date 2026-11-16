# 2026-11-17: 呼出し先を含めたRAM再測定

`sizeof`は全56処理で不変だったが、**通常のテンプレート構成では14処理の観測stack書込み水位が増加した**。したがって、オブジェクトの大きさだけを根拠に「RAMへの影響はない」とは判断できない。

主比較は最後の同条件時間測定後の `879a319` → 現在の `b599e84`。定数化直前の `ab31b0d` を補助比較に含めた。各版で同じprobe・GCC 15.2.0・C++11・`-O2`・固定配列・既定数学ライブラリを使用し、サブモジュールも各commitのgitlinkから展開した。テンプレート構成とfor構成を、それぞれfloat/double・N=8/16・14処理の56例で比較した。

測定方法、対象API、指標の意味、再実行手順は [peak_stack_method.md](../../../memory/peak_stack_method.md) に記載した。2026-11-10の保存済み `.su` は879a319の記録ではないため、この表へ転用していない。

## 観測された増加

表はテンプレート構成の増加例すべて。単位はbyte。`context`は入力・出力・solverを含む呼出し元の固定保存領域で、全版共通。`stack`は呼出し先を含む最大書込み深さ。`既知frame列`はGCC call graphから集計した最大frame合計で、表ではred zoneの保守的な128 byte加算前の値を示す。`合計増加率`はcontext＋観測stackに対する比率。

| 処理・型・N | context | stack 879→b599 | stack増加 | 既知frame列 879→b599 | 合計増加率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GMRES full float 8 | 976 | 440 → 464 | 24 | 480 → 520 | 1.69% |
| GMRES restart float 8 | 664 | 376 → 400 | 24 | 416 → 456 | 2.31% |
| Eigen real float 8 | 896 | 1592 → 1600 | 8 | 1632 → 1656 | 0.32% |
| Eigen complex float 8 | 1440 | 2520 → 2612 | 92 | 2592 → 2600 | 2.32% |
| GMRES full double 8 | 1912 | 616 → 688 | 72 | 688 → 776 | 2.85% |
| GMRES restart double 8 | 1288 | 536 → 560 | 24 | 608 → 648 | 1.32% |
| Eigen real double 8 | 1728 | 2712 → 2784 | 72 | 2784 → 2872 | 1.62% |
| Eigen complex double 8 | 2816 | 4680 → 4816 | 136 | 4816 → 4760 | 1.81% |
| GMRES restart float 16 | 2096 | 536 → 568 | 32 | 608 → 648 | 1.22% |
| Eigen complex float 16 | 5344 | 7160 → 7368 | 208 | 7296 → 7368 | 1.66% |
| GMRES full double 16 | 6296 | 968 → 992 | 24 | 1104 → 1128 | 0.33% |
| GMRES restart double 16 | 4152 | 856 → 912 | 56 | 992 → 1048 | 1.12% |
| Eigen real double 16 | 6400 | 7448 → 7488 | 40 | 7584 → 7624 | 0.29% |
| Eigen complex double 16 | 10624 | 13928 → 14328 | 400 | 14192 → 14328 | 1.63% |

最大の観測増加はEigen complex double16の400 byte。contextとの合計は24,552 → 24,952 byte。合計の最大増加率はGMRES full double8の約2.85%。GMRES full float16、Eigen real float16、固有値のみ、Jacobi、QR、LU、Choleskyでは観測stackの増加はなかった。

## 変更時期の切り分け

- **879→ab31:** テンプレート構成ですでに同じ14処理の観測stackが増加しており、最大400 byte。for構成は56例すべてでcontext・観測stack・既知frame合計が同じ。
- **ab31→b599:** テンプレート構成の観測差はEigen complex double8の追加16 byteだけ。GMRES float8の既知frame合計には追加56 byteが出るが、この入力での書込み水位は変わらない。for構成の観測差はEigen complex float8の16 byteだけ。
- **879→b599:** for構成の増加例は上記の1例のみ。template構成に集中した増加の大部分は、固定値化以前に導入されたテンプレート展開とそれに伴うコード生成の変化によるものと考えられる。ただし、この比較だけで各変更1件の因果を証明したわけではない。

反復数が増えても同じstackを再利用するため、RAMへ単純に反復数を掛けることはしていない。逆反復内にGMRESが入る場合は、呼出し元と呼出し先のframeが同時に存在する分を測定・集計している。

## 測定の確認と限界

6バイナリのcalibrationはすべて成功した。空関数は8 byte、外側256 byte＋内側4096 byteの入れ子呼出しは4,416 byte。各ケースは2種類のpaint値×3回の最大で比較した。計348行（56処理＋calibration2種、3版×2構成）のうち、最小値と最大値が違ったのは5行だけで、差はいずれも1 byte。パターンの偶然一致を2種類の最大で補う目的に対応する。

workerは別翻訳単位に分け、`-mno-red-zone -fno-shrink-wrap -fno-shrink-wrap-separate` を適用した。演算側へこれらのflagsや計測hookは追加していない。逆アセンブルで、RSP採取前にworkerのframeが確保され、paint開始からscan終了まで演算以外のcallがないことを確認した。初期試作ではGCCのshrink wrappingによりRSP採取後にframeが作られたが、RSP復帰検査で検出し、上記worker設定を修正してから全件測定した。

書込み水位は未書込みの予約領域を検出しないため、全入力に対するstack上限ではない。補助のcall graphでは、336演算行中144行が外部未解決callなし。残りには`__stack_chk_fail`、`memmove`、`memset`が到達先として残る。stack canary失敗時の`__stack_chk_fail`も隠さず記録した。外部callの内部frameをゼロと仮定して完全な上限とは扱わない。既知frameの保守的合計が減っていても、実際に通るpathやred zone使用の変化で観測書込み水位が増える例がある。

これらはこのホスト・コンパイラ・入力でのRAM指標であり、組み込みマイコンのABI、割込みstack、全入力での最悪値を保証しない。組み込み側でRAM上限が厳しい場合は、対象toolchainで同じ呼出し関係を調べ、実機のstack水位も確認する必要がある。

## 保存物

- `metadata.json`: 完全commit、submodule、compiler、flags、probe hash。
- `*.raw.csv`: 各バイナリの観測値。`*.su` / `*.ci`: 同じ演算objectのコンパイラ出力。
- `peak_stack_cases.csv`: sizeof・観測stack・既知frame列・解析不能呼出しを統合した値。
- `peak_stack_comparison.csv`: 879a319に対する差分。
- `call_chains.json`: 各入口からの最大既知frame列と未解決call一覧。
- `worker_disassembly.txt` / `worker.su`: 計測workerの監査用出力。
