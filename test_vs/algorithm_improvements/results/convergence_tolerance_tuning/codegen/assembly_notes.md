# テンプレート展開後のstack増加を生成コードで確認

調査日: 2026-11-17。ライブラリ本体を変更せず、前回のRAM測定に使った実行ファイルを逆アセンブルした。

ソース上の配列・solverの大きさは増えていない。一方、今回のGCC出力では、**関数ごとのstack予約量、inline化される範囲、途中の計算結果をstackへ保存する命令が変わっている**。テンプレート展開が計算式や加算順序を保っていても、これらの機械語上の性質までは同じにならない。以下の2例には、この変化を確認できる命令がある。

また、前回の「stack +400 byte」は**書込み水位の増加**であり、予約量が400 byte増えたことを意味しない。以前から予約されていて書かれなかった領域もあるため、両者は分けて評価する。

## 1. 比較対象と指標

比較した版は `879a31981d4cb99173939bd4a12df96bf8ccdf1a`、固定値導入直前の `ab31b0d25f5b081435864541d882f67571db74dd`、`b599e8461d0a4b07aa007bc6376746744767e671`。調査時HEAD `6b1d627edf2de6e3e56deca59b83b37e942a97aa` の `base_matrix` treeはb599と完全に一致する。サブモジュールも3版で同じ。計測probeのSHA-256は計測時の記録と一致する。

環境はx86-64、GCC 15.2.0、C++11、`-O2`。再ビルドは行わず、既存の `/tmp/matrix_peak_stack_audit/` の実行ファイルから `objdump -d -M intel --no-show-raw-insn --disassemble=<symbol>` で取り出した。完全な条件は[元のmetadata](../../performance_recheck_2026_11_17/memory/metadata.json)、入力・測定手順は[測定方法](../../../memory/peak_stack_method.md)を参照。

| 指標 | 意味 |
| --- | --- |
| solver / contextの`sizeof` | solverおよび入力・出力・solverを含む固定保存領域。contextは今回の測定stackの外に置いた |
| 関数frame | GCC `.su` が報告する各関数のstack使用量。以下の対象ではprologueの予約・保存命令でも確認 |
| 最大既知frame列 | `.ci` から作った呼出し列のframe合計。入力で実際に通る経路とは限らず、未解決外部callの内部量も含まない |
| 観測書込み水位 | この入力でstackの最深どこまで書いたか。単なるstack予約や、全入力での最悪値を測る指標ではない |

下表の最大既知frame列は、元の表と同様にred zone分の保守的な128 byte加算前の値。単位はbyte。元データは[peak_stack_cases.csv](../../performance_recheck_2026_11_17/memory/peak_stack_cases.csv)。

| 対象・指標 | 879 | ab31 | b599 / 現HEADと同じ本体 |
| --- | ---: | ---: | ---: |
| GMRES full double8: solver / context | 1208 / 1912 | 1208 / 1912 | 1208 / 1912 |
| GMRES full double8: `_solve` frame | 480 | 560 | 560 |
| GMRES full double8: 最大既知frame列 | 688 | 776 | 776 |
| GMRES full double8: 観測書込み水位 | 616 | 688 | 688 |
| Eigen complex double16: solver / context | 8576 / 10624 | 8576 / 10624 | 8576 / 10624 |
| Eigen complex double16: `solve_eigen_vectors` frame | 13872 | 14064 | 14064 |
| Eigen complex double16: 最大既知frame列 | 14192 | 14328 | 14328 |
| Eigen complex double16: 観測書込み水位 | 13928 | 14328 | 14328 |

2例の増加は固定値導入前のab31ですでに発生している。`BASE_MATRIX_USE_FOR_LOOP_OPERATION_` を定義した構成では、2例ともsolver/context、観測書込み水位、最大既知frame列は3版で同じだった。この比較はテンプレート展開に伴う生成コードの変化を裏付けるが、879→ab31に含まれる変更を1件ずつ切り離した実験ではない。

## 2. GMRES double8: `_solve` の予約領域が80 byte増えた

旧[GMRES `_solve`](last_879_gmres_double8_frame1.asm)と現[GMRES `_solve`](current_b599_gmres_double8_frame1.asm)のprologueには次の違いがある。

```asm
; 879: r15, r14, r13, r12, rbp, rbx の6個をpushした後
841f: sub rsp,0x1a8       ; 424 byte

; b599: 同じ6個をpushした後
833b: sub rsp,0x1f8       ; 504 byte
```

pushが48 byte、戻りアドレスが8 byteなので、`.su` の480→560 byteと一致する。増えたのは保存レジスタの個数ではなく、関数内の予約領域80 byteである。

呼出し構造も変化した。旧 `_solve` 内には `Numerical::norm<double,8>` へのcallがなく、処理が内部へ組み込まれている。新 `_solve` ではこの[norm関数](current_b599_double8_norm.asm)へのcallが複数残る。静的なcall命令の箇所数は12→35になった。これは実行時の反復回数や呼出し回数の集計ではない。80 byteの各領域を全ソース変数へ対応付けてはいないため、増分すべてをspillと断定しない。

既知frame列は旧 `入口112 + _solve480 + apply_operator96 = 688`、新 `入口112 + _solve560 + apply_operator96 + helper8 = 776`。したがってframe列の増分は88 byteで、観測水位の72 byteとは別の値である。

旧[apply_operator](last_879_gmres_double8_frame2.asm)は `sub rsp,0x58` で88 byteを予約するが、`active_size == 8` では、一時入力vectorの初期化を分岐で飛ばす。この経路でstackに書くcanaryは `[rsp+0x48]` であり、呼出し前のSPから24 byte下にある。新normには独自の `sub rsp` がなく、内部でrsqrtを呼ぶ。これらの命令列は旧 `112+480+24=616`、新 `112+560+8+8=688` という書込み水位の説明と一致する。ただし、全命令の動的な経路追跡は行っていない。

## 3. Eigen complex double16: frame増加と実際の計算途中値の保存

旧[solve_eigen_vectors](last_879_eigen_complex_double16_frame1.asm)と現[solve_eigen_vectors](current_b599_eigen_complex_double16_frame1.asm)は、どちらも6個のレジスタをpushする。その後、stack probeを伴う `sub rsp,0x1000` が3回あり、最後の予約が変わっている。

```asm
; 879: 4096 byteずつ3回予約した後
sub rsp,0x5f8

; b599: 同じ3回の予約後
sub rsp,0x6b8
```

総frameは `48+3*4096+0x5f8+8=13872` から `48+3*4096+0x6b8+8=14064` へ192 byte増加した。内部GMRESの `_solve` はどちらもこの関数へ組み込まれており、「新たにGMRESオブジェクトを丸ごともう1個追加した」という変化ではない。

現版では、別関数になった[DenseMultiply::compute](current_b599_eigen_complex_double16_frame2.asm)が `sub rsp,0xd8` で216 byteを予約する。戻りアドレスを含めframeは224 byte。その内部には、出力の途中の和をstackへ保存し、後の列の計算で再利用する命令が実際にある。

```asm
; 出力row 5の実部を更新したあと、その値をstackにも保存
1bfc6: addsd xmm12,QWORD PTR [rdx+0x50]
1bfd7: movsd QWORD PTR [rdx+0x50],xmm12
1bfe8: movsd QWORD PTR [rsp],xmm12

; 次の列の積へ、保存しておいた和を加算
1c6e7: addsd xmm2,QWORD PTR [rsp]
1c6f6: movsd QWORD PTR [rdx+0x50],xmm2
1c6fb: movsd QWORD PTR [rsp],xmm2
```

ソースの `DenseMultiply::Row` / `Column` は行列要素と入出力vectorを参照して加算するだけで、このstack保存用の配列を明示していない。ここには、コンパイラが作った計算途中値のstack保存と再読込みが確認できる。さらに[Row helper](current_b599_eigen_complex_double16_frame3.asm)へのcallが一部残り、戻りアドレスの8 byteが加わる。`inline` を付けた関数でも、すべての呼出しが必ずinline化されるわけではない。[GCCのinline説明](https://gcc.gnu.org/onlinedocs/gcc/Inline.html)

旧[apply_operator](last_879_eigen_complex_double16_frame2.asm)のframeは288 byteだが、`active_size == 16` では一時入力vector256 byteの初期化を飛ばす。stackへのcanary保存は `[rsp+0x108]` にあり、call前のSPから24 byte下になる。この違いを踏まえると、観測水位の内訳は命令列と次のように一致する。

| 経路 | 観測水位と整合する内訳 |
| --- | ---: |
| 旧 | 入口32 + 固有ベクトル関数13872 + apply_operatorの最深書込み24 = 13928 |
| 新 | 入口32 + 固有ベクトル関数14064 + DenseMultiply224 + Row helper8 = 14328 |

つまり400 byte差は、外側frameの192 byte増加と、末端の最深書込みが24→232 byteになった208 byte差と整合する。**400 byteすべてを新たなspill領域の大きさとは解釈しない。** 最大既知frame列の差は `14328-(32+13872+288)=136` byteである。ここでも書込み水位の内訳は静的な命令列と観測値の照合であり、動的な全命令トレースによる確定ではない。

## 4. なぜ計算式の展開だけで変わり得るのか

テンプレート展開後の処理をコンパイラがまとめて最適化すると、途中の値を同時に保持する範囲や、別関数として残す境界が変わり得る。GCCにはレジスタ割当・stackのspill領域の共有・inline化によるstack増大に関する制御が存在する。したがって、ソースに新しい配列がなくても、コンパイラが生成する作業領域まで不変とは限らない。今回のEigenの保存・再読込みは、その可能性が実際の命令として現れた例である。[GCC最適化オプション: レジスタ割当、spill領域、stackとinline化](https://gcc.gnu.org/onlinedocs/gcc-15.1.0/gcc/Optimize-Options.html)

これによって「TMPは必ずRAMを増やす」「すべての増分はレジスタ不足による」とまでは言えない。同じGCCでもN・型・inline化の判断で変わり、今回のfor構成ではこの2例の増分が消える。TMPでインデックスが確定することと、最終的な機械語の効率が必ず向上することは別々に確認する必要がある。

## 5. 組み込みマイコンへの適用範囲

今回確認したのはPC上のx86-64命令であり、Arm向けのコンパイル・実機測定は行っていない。Arm AAPCS32では引数・一時値に使うレジスタ、呼出し先が保存するレジスタ、stackの整列条件が規定される。例えばr0〜r3を引数等に使い、公開関数境界のstack整列は8 byteである。今回のx86-64のframeのbyte数をそのままArmへ移すことはできない。[Arm AAPCS32](https://github.com/ARM-software/abi-aa/blob/main/aapcs32/aapcs32.rst)

マイコンでも、コンパイラが途中の値をstackへ保存する可能性はある。ただし増減の向きと大きさは、対象CPU、FPUの有無・浮動小数点ABI、コンパイラと最適化設定に依存する。今回の結果から「マイコンなら増えない」「マイコンでは400 byte増える」のどちらも確定できない。対象toolchainで `.su`・呼出し関係・逆アセンブルを確認し、実機の割込み等を含めたstack設計へ反映するのが必要な検証となる。GCCの `.su` と `-fcallgraph-info=su` はそのための関数単位・呼出し関係の情報を出力する。[GCC開発者向けオプション](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html)

## 6. 保存した証拠と限界

- [assembly_notes.json](assembly_notes.json): 数値、完全commit、実行ファイルhash、現HEADとの本体一致確認、参照元。
- [assembly_extraction.json](assembly_extraction.json): 3版・2例の最大既知frame列に含まれる関数のsymbol、`.su`、ASM hash、stack調整命令とcall箇所。
- `*.asm`: 同じ実行ファイルから抜き出した実際の命令列。アドレスはこのビルド固有。

前回の既知call graphには `__stack_chk_fail` などの未解決外部呼出しが残る。今回もそれらのframeをゼロと仮定して全入力のstack上限を保証してはいない。ライブラリ本体の修正、新しい速度測定、マイコンでの性能測定はこの監査には含めていない。
