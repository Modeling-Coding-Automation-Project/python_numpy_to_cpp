# TMPを維持したEigen収束許容誤差の調整

**許容誤差の調整で改善する例はあるが、今回の型共通候補を一律適用すると未使用次数で新しい失敗が生じた。** 探索集合では品質・収束が78/92件から92/92件へ改善した。一方、候補を固定してから測った検証集合では34/48件から30/48件へ低下した。本体、TMPの展開方法、β、shift、分母保護、反復上限は変更していない。

## 固定条件と評価

対象は `6b1d627edf2de6e3e56deca59b83b37e942a97aa`。行列ヘッダは前回比較の `b599e84` と同じ。g++ 15.2.0、C++11、`-O2 -Wall -Wextra -Werror`、既定数学設定で検証した。`BASE_MATRIX_USE_FOR_LOOP_OPERATION_` が定義された場合はprobeのコンパイルをエラーにし、TMP構成を維持した。

| 固定項目 | 値 |
| --- | --- |
| GMRES breakdown係数 β | `8e-12` |
| 相対シフト | `3e-4` |
| division_min | `1e-20` |
| 絶対許容誤差 | 0 |
| 一般固有値のQR上限 | N3は128、その他は64 |
| 逆反復上限 | 30 |
| 内側GMRES | K=N、cycles=1、直交化1回 |
| Jacobi上限 | 10 |

変更するのは `relative_tolerance`、`eigenvector_tolerance`、正の `gmres_k_decay_rate` だけ。Jacobiではrtolだけが適用される。全て構築後に設定した。

品質は設定した許容誤差とは独立に、固有値誤差と固有ベクトル残差の**両方が `2e-6` 以下**かを判定する。固有値は格納された入力を `%a` で出力し、NumPy binary64を参照にした。対応付けは最大誤差を最小化する二部マッチング。固有値誤差は `||A||F`、vector残差は `||A||F ||v||2` で正規化する。`usable` はこの品質に加え、固有値と一般solverの固有ベクトルの収束フラグが成立した件数である。

探索集合はfloat/double各46件。既存N3の22件に、N4/8/16の対称三重対角、対角相似変換、回転した非正規行列、複素共役ブロックを加えた。一般real/complexが37件、Jacobiが9件/型。rtol単独13設定と追加10設定、計2,116件を測定した。全出力は有限だった。

## 探索結果と固定した候補

初期Aは `(rtol, vtol, inner)=(8e-9, 3.2e-6, 5e-3)`。型共通候補は **`(1e-6, 4.8e-6, 1e-3)`** とした。

候補はN5/7/12の検証結果を見る前に [selection記録](eigen_tolerance_selection.json) に固定した。探索集合で全件が品質・収束を満たし、その中でQR sweepが少ないrtolを選択した。inner=`1e-4`も全件通過したが、同じ合格数でより緩い `1e-3` を選んだ。時間は候補選択後に測定しており、検証結果を使った再調整はしていない。

| 設定 rtol / vtol / inner | 品質 float / 46 | 品質 double / 46 | usable float / 46 | usable double / 46 |
| --- | ---: | ---: | ---: | ---: |
| A: `8e-9 / 3.2e-6 / 5e-3` | 42 | 42 | 36 | 42 |
| `1e-5 / 3.2e-6 / 5e-3` | 40 | 40 | 40 | 40 |
| `1e-6 / 3.2e-6 / 5e-3` | 44 | 44 | 40 | 44 |
| `3e-7 / 3.2e-6 / 5e-3` | 44 | 44 | 40 | 44 |
| `1e-6 / 4.8e-6 / 5e-3` | 44 | 44 | 44 | 44 |
| `1e-6 / 3.2e-6 / 1e-3` | 46 | 46 | 42 | 46 |
| **候補: `1e-6 / 4.8e-6 / 1e-3`** | **46** | **46** | **46** | **46** |
| `1e-6 / 6.4e-6 / 5e-3` | 42 | 42 | 42 | 42 |
| `1e-6 / 4.8e-6 / 1e-6` | 46 | 46 | 44 | 43 |

rtolを`1e-6`付近へ変えると、以前失敗したN8対称三重対角の固有ベクトルが改善する。rtol単独では残ったN8非正規の残差約0.077は、innerを`1e-3`へ変えると約`1e-6`へ改善した。vtol=`4.8e-6`は、外部品質が十分でも内部基準の`3.2e-6`を満たさなかった一部float N16の停止条件を調整する。

全指標が一様に小さくなるわけではない。探索集合で候補の残差がAより増えたケースは33件あるが、全て固定品質ゲート内だった。候補の最大固有値誤差/最大vector残差はfloat `1.16004e-6 / 1.47540e-6`、double `6.61462e-7 / 1.47367e-6`。Aで合格したケースが候補で品質不合格になる例は探索集合にはなかった。

一方、次の副作用も確認した。

- rtol=`1e-5`ではN3非正規の複素solverの誤差が約`5e-6`になり、N16 Jacobiの残差も約`3.7e-6`に増える。
- vtol=`6.4e-6`ではN16非正規の残差がfloat `2.06979e-6`、double `2.01084e-6`となる。フラグは成功でも固定品質ゲートを超える。
- inner=`1e-6`まで厳しくすると、N3近接固有値・巡回行列・複素ブロックでinner失敗が出る。残差が小さいだけで成功へ読み替えていない。
- rtol=`1e-11`以下はfloatのN3近接固有値でQR上限に到達し、改善しない。

## 未使用次数による検証

候補固定後、N5/7/12で同じ4種類の大きめ入力を測定した。float/double各24件、計48件であり、候補選択には使っていない。

| 集合 | A usable | 候補 usable | 改善 | 新しい悪化 |
| --- | ---: | ---: | ---: | ---: |
| 探索92件 | 78 | 92 | 14 | 0 |
| 未使用次数48件 | 34 | 30 | 4 | 8 |

未使用次数の品質合格数も34→30である。改善はN7相似変換の両型real/complex 4件。新しい品質失敗はN5非正規とN7対称三重対角、それぞれ両型real/complexの計8件だった。両方で失敗する10件も残る。

| float realの例 | A残差 | 候補残差 | A→候補の状態 |
| --- | ---: | ---: | --- |
| 探索 N8 対称三重対角 | `0.0784536` | `1.01789e-6` | 未収束→成功 |
| 探索 N8 非正規 | `0.0772947` | `1.01652e-6` | 未収束→成功 |
| 検証 N7 相似変換 | `0.182568` | `9.91472e-7` | 未収束→成功 |
| 検証 N5 非正規 | `1.08902e-6` | `2.31163e-6` | 両方フラグ成功、候補は品質不合格 |
| 検証 N7 対称三重対角 | `7.73855e-7` | `0.121015` | 成功→未収束 |

N5非正規では候補の内部残差約`4.29e-6`が緩和したvtol=`4.8e-6`に入る一方、独立残差が`2e-6`を超える。許容誤差を緩めたフラグだけでは改善と判断できない例である。

### 固有値順と初期vectorの相互作用

N7対称三重対角floatの生出力を解析すると、Aの固有値順は厳密sin固有モード番号で `[7,6,5,1,4,2,3]`、候補は `[7,6,5,4,1,2,3]`。候補では0始まりslot 3へmode 4が移動する。

実装の初期vectorは `ones + e_slot`。mode 4の成分は `sin((i+1)4π/8)` であり、全成分和とslot 3の成分がともに0なので、この初期vectorとの厳密射影は0になる。生出力から求めた正規化射影は約`3.5e-17`で、該当vectorの残差だけが`0.121015`となった。A側の対応射影には0がない。

以前のN8を直すrtol変更が、N7では別の零射影配置を生む。これは収束許容誤差を変更するだけでは、この初期値・固有値順依存を普遍的に解消できないことを示す。有限精度の内部反復を全追跡したものではないため、全失敗の原因をこの機構だけに限定しない。候補を検証集合へ合わせて選び直すことはしていない。

## 計算時間

候補固定後、同一current binaryへA/候補を実行時引数で渡し、探索92＋検証48の140行を計測した。CPU 0固定、7 roundでプロセス順を交替。各行はwarmup後3 trialの平均時間の中央値。constructor・設定・固有値・固有ベクトル求解を含め、品質計算とprintfは計時外。入力と全solver状態へmemory barrierを入れている。

旧879版や前回レポートの時間とは測定機会が違うため比率を計算しない。以下は今回同時に測ったAとの比較である。時間比はroundごとの候補/Aの中央値。

「明瞭な高速化」は比中央値<0.90かつ全7比<0.95、「明瞭な遅延」は比中央値>1.10かつ全7比>1.05という記述的基準であり、統計的信頼区間ではない。

| 分類 | 件数 | 比中央値<1 | 明瞭な高速化 | 明瞭な遅延 |
| --- | ---: | ---: | ---: | ---: |
| 両方が固定品質ゲートを通過 | 110 | 65 | 10 | 0 |
| 両方が品質＋収束を通過 | 104 | 59 | 8 | 0 |
| 新しく品質＋収束を通過 | 18 | 18 | 14 | 0 |
| 新しく品質不合格 | 8 | 4 | 0 | 4 |
| 両方が品質＋収束を満たさない | 10 | 2 | 0 | 3 |

両方が品質＋収束を満たす104件のうち、8件の明瞭な高速化は全てJacobiだった。この分類の時間比中央値は約0.998であり、全体的な大幅高速化を示す結果ではない。新しく品質不合格になった8件のうち4件の時間中央値は短いが、改善には数えない。

| float realの例 | A→候補 μs | 時間比 | 品質・収束 |
| --- | ---: | ---: | --- |
| 探索 N8 対称三重対角 | 44.027→29.505 | 0.667 | 不合格→合格 |
| 探索 N8 非正規 | 59.190→36.825 | 0.617 | 不合格→合格 |
| 探索 N16 対称三重対角 | 373.217→278.130 | 0.735 | 独立品質は両方合格、未収束→収束 |
| 検証 N7 相似変換 | 40.733→21.808 | 0.537 | 不合格→合格 |
| 検証 N5 非正規 | 10.605→10.120 | 0.938 | 合格→品質不合格 |
| 検証 N7 対称三重対角 | 22.273→39.885 | 1.790 | 合格→不合格 |

探索集合のQR sweep合計はfloat419→369、double414→369。検証集合も309/308→273へ減ったが、N7対称の全求解時間は増えた。QR sweepだけから内側GMRES・逆反復を含む時間を推定しない。

## RAMと実装の範囲

A/候補は同一binaryの実行時パラメータだけが異なる。全140行でsolver `sizeof` が一致し、コンパイル時に確保する関数ごとのフレームも同一。solverの`sizeof`の例はreal float N8=640 B、N16=2208 B、real double N16=4352 Bである。[同一binaryの `.su`](timing/eigen_tolerance_benchmark.su) とhashを保存した。

ピークスタックの実測はこの調整試験では行っていない。入力・診断用コピーを含むbenchmarkフレームを、ライブラリ単独のピークRAMとして扱わない。動的メモリ確保の方式、テンプレート容量、関数・クラス構造の変更もしていない。

## 再現性

- [候補固定JSON](eigen_tolerance_selected_profiles.json)、[固定時点の記録](eigen_tolerance_selection.json)
- [rtol単独集計](eigen_tolerance_rtol_summary.csv)、[追加設定集計](eigen_tolerance_joint_summary.csv)、[未使用次数集計](eigen_tolerance_holdout_summary.csv)
- [時間と独立品質を結合した集計](timing/eigen_tolerance_timing_summary.csv)、[計時環境と同一binary hash](timing/eigen_tolerance_timing_metadata.json)
- [archive内容・runner更新の検証記録](eigen_tolerance_integrity_verification.json)

品質source `eigen_tolerance_probe.cpp` は探索・検証を通じて不変。計時は別の `eigen_tolerance_benchmark.cpp` を作り、入力生成部分の一致をrunnerで確認する。時間行のstatus・sweep・sizeofが事前の品質結果と一致することも全行検査した。

初回品質計測後、runnerにarchiveの全ファイルを固定Git blob・ファイル集合へ照合する検査を追加した。既存3フェーズのarchiveも照合成功。元のmetadataに記録した旧runner hashを保持し、現在のhashと検査結果を別JSONへ追記している。計時runnerはこの検査を経てコンパイルする。

NumPy対応Pythonを使う再現例（作業ツリーの本体は編集しない）：

```sh
python3 test_vs/algorithm_improvements/eigen_tolerance_sweep.py --work /tmp/eigen_tolerance_reproduce
python3 test_vs/algorithm_improvements/eigen_tolerance_sweep.py --work /tmp/eigen_tolerance_reproduce --reuse-build --label joint --profiles test_vs/algorithm_improvements/results/convergence_tolerance_tuning/eigen/eigen_tolerance_joint_profiles.json
python3 test_vs/algorithm_improvements/eigen_tolerance_sweep.py --work /tmp/eigen_tolerance_reproduce --holdout --label holdout --profiles test_vs/algorithm_improvements/results/convergence_tolerance_tuning/eigen/eigen_tolerance_selected_profiles.json
python3 test_vs/algorithm_improvements/eigen_tolerance_timing.py --work /tmp/eigen_tolerance_reproduce
```
