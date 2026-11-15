# QR実装・評価記録

対象: QR-1 / QR-2。比較元は `b7faabf3bbac107d1be1462d139f4be0fa25a6b7`。測定は x86_64 Linux / g++ 15.2.0、`-std=c++11 -O2 -Wall -Wextra -Werror`、固定配列・既定mathマクロ。MCU実測・WCETの証明ではない。

以下の数値と試験結果は2026-11-10の記録。2026-11-11以降は `Base::Math::sqrt` / `Base::Math::rsqrt` に委譲し、行列クラスの平方根精度設定を廃止した。旧精度設定の結果を現行APIの測定値として扱わない。

## 採否

| 案 | 採否・理由 |
|---|---|
| QR-1 compact Householder、Q遅延生成、Qᵀ直接適用 | 採用。`QRDecompositionHouseholder<T,M,N>`。当初追加した `SqrtIterations` は2026-11-11に廃止。16×16/32×32では因子のみ・Q出力あり双方で速度改善、因子保持RAMも削減。3×3では既存Givensの方が速いため既存APIを一律置換しない。 |
| QR-2 in-place Givens | 採用。値渡し・tuple経由の因子コピーをコアから除去。既存の値返却ヘルパは互換ラッパとして残す。 |
| QR-2 再solve・疎fill-in修正 | 採用。solveごとにQ初期化、CSRは初期scatterにだけ使用し現在のRの下三角を全走査。消去位置を明示的に0にする。 |
| QR-2 構造既知Hessenberg専用経路 | 条件付き採用。`solve_hessenberg(A)`。利用側が `row > column + 1` の全要素0を保証する。一般入力には `solve` を用いる。主に16/32次元で走査省略の改善を確認。 |
| 構造検査付きHessenberg専用経路 | 不採用。検査の全走査が省略効果を消す。試作をbenchmarkの `CheckedHessenbergQR` にのみ保存。任意帯域の構造推測・元CSRだけの消去は導入しない。 |
| sqrt精度予算0/1/2/4 | 当時は試験後に既定2を採用したが、2026-11-11の方針変更でクラス設定を廃止。現在の平方根精度は数学ライブラリ側の構成に従う。 |
| 大きい除去閾値による近似 | 試験済み、既定閾値の増大は不採用。既存の `division_min` による微小成分省略は明示的な近似設定として維持。階数落ち・設定値補正をgetterで通知する。 |

## 時間・保持量・精度

時間は3回の測定組の最小値。各組は3×3で12000回、8×8で3000回、16列以下で700回、32列で100回。毎回新しいsolverを構築してsolveする。性能測定中に誤差検証は含めず、compiler barrierで因子全体の書込みを観測可能にする。入力は決定的な非対称密行列。float/doubleで3/8/16/32次元、16×5、Hessenberg、対角近傍を評価。生データは `qr_baseline.csv` / `qr_improved.csv`。

再構成誤差は `max|QR-A| / max|A|`、直交性は `max|QᵀQ-I|`。下三角の最大値も別に検査する。当時の共通の受入れ精度をfloat `3e-5`、double `2e-12`とし、当時の既定設定の回帰テストはこれを満たした。各ケースの丸め誤差が常に単調減少するとの主張はしない。

| 型・寸法 | 方式 | ns/分解 | solver sizeof B | 再構成相対最大誤差 |
|---|---|---:|---:|---:|
| float 3×3 | 旧Givens | 108.8 | 76 | 1.19e-07 |
| float 3×3 | 新Givens | 69.5 | 80 | 1.19e-07 |
| float 3×3 | Householder 因子 | 94.1 | 56 | 1.19e-07 |
| float 3×3 | Householder＋Q | 109.8 | 56 | 1.19e-07 |
| double 16×16 | 旧Givens | 16714.6 | 4104 | 2.01e-07 |
| double 16×16 | 新Givens | 5150.5 | 4112 | 1.9e-14 |
| double 16×16 | Householder 因子 | 1562.5 | 2192 | 1.28e-14 |
| double 16×16 | Householder＋Q | 2566.8 | 2192 | 1.28e-14 |
| float 32×32 | 旧Givens | 356286.4 | 8196 | 6.68e-07 |
| float 32×32 | 新Givens | 27281.1 | 8200 | 8.9e-07 |
| float 32×32 | Householder 因子 | 7846.2 | 4232 | 2.23e-07 |
| float 32×32 | Householder＋Q | 11674.3 | 4232 | 2.23e-07 |
| double 16×5 | 旧Givens | 6897.4 | 2696 | 2.76e-07 |
| double 16×5 | 新Givens | 2518.0 | 2704 | 2.08e-14 |
| double 16×5 | Householder 因子 | 398.7 | 696 | 9.86e-15 |
| double 16×5 | Householder＋Q | 1037.1 | 696 | 9.86e-15 |

`sizeof` はsolverの保持量であり、全RAMではない。Householder＋Qには別途 `M*M*sizeof(T)` の出力が必要で、`get_R` でRを展開するとその出力も必要。`apply_Q_transpose_in_place` は既存のベクトルを直接更新できる。

float16×16の `-fstack-usage` を `qr_stack.cpp`（solverと入力は呼出し側所有）で確認した。旧Givens呼出しラッパ64 B＋内部回転4144 B＋rsqrt8 Bに対し、新Givensラッパ80 B＋内部回転8 B＋rsqrt8 B。Householder因子コア80 B＋ラッパ8 B＋rsqrt8 B。Q出力コア56 B（呼出し側の1024 B出力領域は別）。各関数のフレームであり割込み・呼出し側・ABI等を含む最終MCU最大RAMではない。生データは `qr_stack_baseline.txt` / `qr_stack_improved.txt`。

## 追加候補の比較

| double16×16 | ns | 再構成誤差 | 直交性誤差 |
|---|---:|---:|---:|
| householder_p0 | 1441.1 | 0.000318 | 0.000524 |
| householder_p1 | 1431.0 | 1.07e-07 | 1.85e-07 |
| householder_p2 | 1562.5 | 1.28e-14 | 2.35e-14 |
| householder_p4 | 1668.3 | 3.45e-16 | 7.22e-16 |
| householder_apply_qt | 1666.2 | 1.28e-14 | 2.35e-14 |

全体の実行時間差にはノイズがあり、この一測定で微小な差の優位性は主張しない。0回は精度低下が大きく、1回と比べ速度が逆転する測定もある。当時の既定2回は旧Givensより高精度・高速だった。現在は2回・4回などを行列クラスから指定するAPIは提供しない。

| double32×32 Hessenberg | ns | 再構成誤差 |
|---|---:|---:|
| givens | 2395.8 | 6.32e-15 |
| givens_hessenberg | 2241.6 | 6.32e-15 |
| givens_checked_trial | 2537.9 | 6.32e-15 |

対角近傍double16×16で `division_min=1e-2` とした試験は 216.8 ns、再構成誤差 0.000428、R下三角0。旧実装は下三角を残すため再構成だけなら誤差0になるが、下三角の相対最大値 0.000428 が残り、厳密な上三角因子にならない。近似の採否は再構成と三角性を併せて判定する。

## 安全性・API上の注意

- 既存Givensの符号規約を維持。Householderの符号は異なり得るがQ/R再構成で検証する。
- Householder・Givensとも、零・負の `division_min` を既定正値に補正し `get_threshold_corrected()` で通知。数値的階数落ちは `get_rank_deficient()`。
- Givensは大きい非零成分で比を作り、sqrtへ[1,2]の値を渡す。Householderは列最大絶対値でスケールする。正規化分母の非零性を先行判定で保証し、単位の異なる量に同じ閾値を再適用しない。
- 回帰テスト `qr_regression.cpp` はfloat/double、1×1、零、矩形、繰返しsolve、疎fill-in、Qᵀ適用、設定値補正、1e±200のdouble尺度を検証。0 failures。
- 新APIを含む `constraints_regression.cpp` は、固定配列構成で0 allocations。公開集約ヘッダを `<cmath>` / `<algorithm>` のpoison headerと `-fno-exceptions` 付きでコンパイルして実行成功。ライブラリ内の非有限値チェックは追加していない。
- `<algorithm>`/`<cmath>`/while/例外/動的ワークは追加していない。現行のテンプレートパラメータは固定容量を表し、平方根の反復数は指定しない。
