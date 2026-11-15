# LU・Cholesky 実装と比較の記録

比較基準は `b7faabf3bbac107d1be1462d139f4be0fa25a6b7`。以下はホスト g++、`-std=c++11 -O2 -Wall -Wextra -Werror -fstack-usage`、固定配列構成の測定例。1測定は500回の中央値（5組）、さらに独立3プロセスの中央値を取った。CPU負荷・周波数により時間は変動する。MCUのサイクル数・WCET測定ではない。

再現ソースは `lu_chol_benchmark.cpp` と `lu_chol_regression.cpp`。共通 `run_validation.py` から実行できる。旧ソースには `ALGORITHM_IMPROVEMENTS` を定義せず、新ソースにだけ定義する。ホスト検証用の `<cmath>` はこれらの試験ファイルだけで使用する。

以下の数値と試験結果は2026-11-10の記録。2026-11-11以降は平方根・逆平方根を `Base::Math::sqrt` / `Base::Math::rsqrt` に委譲し、`PackedCholesky` の `ROOT_LOOPS` / `MULTIPLY_ROOT` を廃止した。旧設定の測定値を現行APIの性能として扱わない。

## 採否

| 提案 | 判断と実装 |
|---|---|
| LU-1 | 採用。全列の決定的な部分ピボット、`P A = L U`、置換符号、特異状態。PythonNumpy側にも `get_P()` / `get_pivot_indices()` を追加。対角が非零でも極小な例の精度が大幅に改善した。 |
| LU-1 スケール付き選択 | `solve_scaled(A)` に限定して採用。比較用の行尺度を使い、A/bは変更しない。doubleのtiny-pivot例で誤差 `1.46e-16 → 1.98e-17`、Hilbertで `2.43e-11 → 1.62e-11`。float Hilbertでは悪化したため既定化は不採用。追加の行尺度はO(n)の局所領域。旧実装が通常行列でピボット探索を省略する経路も比較したが、その方式はtiny-pivot例で不安定なため新しい省略APIは採用しない。 |
| LU-2 | 採用。1配列の因子保存、ピボット逆数の共有、右辺のみの `solve_rhs`。逆対角キャッシュは `LUDecomposition<T,N,true>` で任意選択とし、既定は省RAMのfalse。subnormalピボットでは逆数化をせず、既知の非零ピボットで直接除算する。 |
| LU-3 | 採用、既定は補正なし。`refine<ResidualT>(A,b,x,R)` / `solve_refined<ResidualT>(A,b,R)`。R=0/1/2と同精度/double残差を比較。float Hilbertに対するdouble残差は有効。同精度の補正を常に実行する方針は不採用。 |
| CHOL-1 | ワーク共有・疎入力の直接散布・packed保存を採用。密なright-lookingへの一律置換は不採用：同等精度・同じdense出力では一貫した改善がなかった。既存APIにはin-place内積型を用い、right-lookingはpacked経路で採用。`chol_right_trial` にdense案の試験を残した。 |
| CHOL-2 | 採用。全pivotで精度/入力下限を統一、固定失敗枠、fallback維持、対角の零は厳密な0。現行APIは `PackedCholesky<T,N>`。平方根の精度と計算方式のクラス設定は廃止し、`Base::Math::rsqrt` の逆数から対角を作る。旧精度設定と `d * rsqrt(d)` の測定記録は下表に残す。 |
| CHOL-3 | 採用。`SPDLDLT<T,N>` はpackedのL/D保存、平方根なし、正のpivotの確認、`solve_rhs` を提供。SPD専用であり、不定値入力は失敗状態を返す。 |

これらの選択肢が全ての行列で優れることは保証しない。特に正しい部分ピボットでも、特定の行列の丸め誤差が旧版より大きい例がある。既定経路のRAM減少・数値安全性、用途別の精度/時間設定を分けて評価する。

## LU：誤差・時間・保持量

解の参照値は、実際にTで格納されたA/bに対するlong double部分ピボット解。`max|x-x_ref| / max|x_ref|` を報告する。regularは決定的な密行列に対角Nを加えたもの、tiny-pivotは先頭対角を `1e-7` へ置換、Hilbertは `A(i,j)=1/(i+j+1)`。同じ右辺を両リビジョンで使う。

| ケース | 旧版 ns / 相対誤差 | 新版 ns / 相対誤差 |
|---|---:|---:|
| float, N=16, regular, 分解+解 | 739 / 4.12e-7 | 887 / 4.12e-7 |
| double, N=16, regular, 分解+解 | 762 / 3.59e-16 | 859 / 3.59e-16 |
| float, N=6, tiny-pivot | 84 / 1.64e-1 | 118 / 2.09e-6 |
| double, N=6, tiny-pivot | 93 / 9.01e-10 | 約130 / 1.46e-16 |
| float, N=6, Hilbert | 83 / 2.33e-2 | 約120 / 2.20e-3 |
| double, N=6, Hilbert | 92 / 7.34e-12 | 約140 / 2.43e-11 |

N=16のLUオブジェクトはfloat **2184→1168 byte**、double **4232→2192 byte**。部分ピボット探索の時間増加があるため「常に高速化」とはしない。右辺だけの解法はfloat180 ns、double185 ns。キャッシュを有効にするとそれぞれ153 ns、148 ns、オブジェクトは1232/2320 byte。キャッシュは丸め順序が変わるため誤差も完全一致ではない。

float Hilbert, N=6の補正比較：

| 残差の計算型 | R=0 | R=1 | R=2 |
|---|---:|---:|---:|
| float | 2.20e-3 | 1.04e-2 | 1.50e-2 |
| double | 2.20e-3 | 1.65e-5 | 1.09e-7 |

同精度の反復改善はこの例では逆効果であり、自動適用しない。float regular, N=16ではdouble残差1回で `4.12e-7 → 2.45e-8`、時間は887→1173 ns。double regularでは1回の効果が小さく、2回で改善しない例もある。double演算がソフトウェアになるMCUでは時間を再評価する。

## Cholesky / LDLᵀ

SPDは対角 `2+(i%5+1)/7`、下三角の小さい要素を持つLから `A=L Lᵀ` をTで構築した。同じ格納Aに対し、long doubleで `max|YᵀY-A|/max|A|` または `max|LDLᵀ-A|/max|A|` を計算する。二進数で厳密な平方根だけになる行列は、精度設定の評価には使用しない。

| N=16 | 時間 ns | 相対再構成誤差 | 保持する出力/因子 byte |
|---|---:|---:|---:|
| float旧dense | 514 | 1.88e-7 | dense出力1024 |
| float新dense互換API | 523 | 1.87e-7 | dense出力1024 |
| float新dense出力参照 | 406 | 1.87e-7 | dense出力1024 |
| float旧sparse | 536 | 1.88e-7 | dense出力1024 |
| float新sparse互換API | 489 | 1.87e-7 | dense出力1024 |
| float新sparse出力参照 | 398 | 1.87e-7 | dense出力1024 |
| float packed, root=1 | 636 | 1.87e-7 | オブジェクト556 |
| float packed, root=3 | 749 | 4.17e-8 | オブジェクト556 |
| float LDLᵀ | 495 | 1.17e-7 | オブジェクト556 |
| double旧dense | 519 | 1.35e-7 | dense出力2048 |
| double新dense互換API | 590 | 1.35e-7 | dense出力2048 |
| double新sparse出力参照 | 444 | 1.35e-7 | dense出力2048 |
| double packed, root=1 | 675 | 1.35e-7 | オブジェクト1112 |
| double packed, root=2 | 745 | 1.82e-14 | オブジェクト1112 |
| double packed, root=3 | 781 | 3.17e-16 | オブジェクト1112 |
| double packed, root=2, multiply=true | 706 | 1.80e-14 | オブジェクト1112 |
| double LDLᵀ | 505 | 2.39e-16 | オブジェクト1112 |

root=0も試験した。N=16の相対誤差は約3.68e-4で、packed root=1に対する時間短縮は約50〜60ns。当時は粗い許容誤差向けの候補として比較し、既定化は不採用とした。現在は行列クラスでroot回数を指定しない。floatは丸めによりroot=2がroot=1より改善しないケースもあり、回数と精度の単調性は保証しない。

`MULTIPLY_ROOT=true` は対角と非対角の近似誤差の関係を変える。double/root=2の小さい改善を確認したが、float/root=2は誤差・時間とも悪化した試行があるため、その構成の既定化は見送った。現在は `MULTIPLY_ROOT` 自体を廃止している。

### RAMの読み方

上表のsizeofだけを最大RAMと解釈しない。`-fstack-usage` によるN=16呼出しフレーム（g++の今回のインライン判断を含む）は次のとおり。

| 呼出し | float 旧→新 byte | double 旧→新 byte |
|---|---:|---:|
| dense互換関数を呼ぶbenchmark wrapper | 3184→2128 | 6256→4160 |
| sparse互換関数を呼ぶbenchmark wrapper | 2096→2128 | 4144→4160 |
| 新dense出力参照wrapper | 40 | 32 |
| 新sparse出力参照wrapper | 40 | 24 |

出力参照wrapperには呼出し側のdense出力1024/2048 byte、入力、呼出し先フレームを別途加算する。packedオブジェクトも呼出し先の局所領域や必要ならdense getter出力を別途加算する。ソース上で疎入力用のdenseコピーを消しても、既存tuple APIではコンパイラの一時領域再利用が変わり、最大RAMがそのままn²個分減るとは限らなかった。そこで出力参照APIを追加した。互換APIは従来どおりfallbackとboolを返す。出力参照APIは失敗時に部分因子とtrueを返し、fallbackを自動保持しない。

## API・安全性・回帰試験

- LUは特異・未分解の因子から解を生成せず、零ベクトルを返す。`is_valid()` / `is_singular()` を先に確認する。置換は `P(i,pivot_indices[i])=1`。
- `division_min<=0` は既定の正値へ補正して状態に記録する。Cholesky旧APIは追加状態を返せないため、その場合は既存の失敗フラグとfallbackを返す。
- LU/LDLᵀでsubnormal pivotの逆数がoverflowし得る場合は直接除算に戻す。`A=[[4e-310,1e-310],[1e-310,3e-310]]`、`division_min=1e-320` でLU、キャッシュLU、LDLᵀの有限な正しい解を確認した。
- Choleskyから呼ぶ `Base::Math::rsqrt` の既定Newton実装は初期値にfloat変換を使う。pivotは正のnormal float範囲内であることを契約とし、doubleの全指数範囲を保証しない。`1e30` のpivotに対するrsqrt戻り値を入力のdivision_minでクランプしないことも確認した。
- 一般Cholesky/LDLᵀはSPDを前提とし、相対pivot下限はpacked APIで選べる。対角版は従来の半正定値許容を保ち、0→0、小さい正値の平方根も別の固定下限で持ち上げない。
- 全ループは有限for、失敗後のpivot枠はフラグで演算を省く。while、動的ワーク、例外、非有限値検出を追加していない。入力・出力の非有限値検出は利用側。
- float/doubleでPA=LU、行列式の符号、コピー後の解、再利用、キャッシュ、固定補正、零行列、invalid threshold、1×1、SPD再構成、負pivot、fallback、CSRの下三角散布、疎パターン変更後の初期化、出力参照のin-place alias、対角零、小さい対角、相対閾値を検証した。
- `lu_chol_regression.cpp` は通常実行と `address,undefined,float-divide-by-zero` sanitizerで失敗0。既存全体試験も最初の統合時に通過（最終の全構成結果は共通レポートを参照）。
