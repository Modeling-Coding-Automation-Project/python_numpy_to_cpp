# 行列計算アルゴリズムの実装・性能評価結果

評価日: 2026-11-10

比較元: `b7faabf3bbac107d1be1462d139f4be0fa25a6b7`。サブモジュールは `base_utility_cpp=6d4807d`、`python_math_to_cpp=2a2c19b`。改善後は本レポートと同時に変更した作業ツリー。

[改善案](06_matrix_algorithm_improvement_proposals.md)のQR-1〜2、EIG-1〜4、LU-1〜3、CHOL-1〜3、GMRES-1〜5を実装・試験した。既存APIの不整合を修正し、省RAM・精度・時間を選ぶAPIを追加した。効果のなかった設定の既定化や一律置換は採用していない。特に、密Choleskyのright-lookingへの一律変更、構造検査付きHessenberg QR、矩形GMRESの常時matrix-free化は見送った。

**全ケースで実行時間や誤差が一律に減る、という結果ではない。** 用途ごとの時間・RAM・精度の選択肢が改善したことを確認した。例えばLUの部分ピボットは時間を増やすが、因子保持量を減らし、微小な非零ピボットを持つ例で真値への誤差を大きく改善する。設定によって誤差が悪化した例も以下に記録する。

## 1. 評価方法と制約

- C++11、g++ 15.2.0、x86_64 Linux。性能測定は `-O2 -Wall -Wextra -Werror -fstack-usage`。`-ffast-math`は使用しない。入力とコンパイラ条件を改善前後で揃えた。基準ソースは作業ツリー外へ固定して保存した。
- 組み込み構成は固定配列、標準数学関数マクロ未定義。ターゲット実装に `<cmath>`・`<algorithm>`、`while`、例外、動的ワーク領域を追加しない。テンプレートは容量・精度・任意機能の選択に用い、ループのメタプログラミング展開は追加していない。
- API前後の非有限値チェックは利用側の責任を維持した。ホスト試験の `<cmath>`、NumPy、メモリ監視はターゲット実装へ持ち込んでいない。
- `division_min`の零・負値を補正または失敗状態で通知する。零ノルム、数値的階数落ち、特異な三角因子、非正定値を単なる正常終了にしない。入力と逆数に同じ閾値を無条件に適用しない。
- 因子の検証は再構成・直交性・三角性、線形方程式は既知解または独立な高精度参照、固有値は既知スペクトルとNumPy、固有ベクトルは `Av−λv` を用いた。小さい残差だけで真値に近いと判断していない。
- `sizeof`、呼出し先を含めたスタックフレーム、密な出力・入力を区別する。ホストのスタック結果はMCUの最大RAMではない。対象MCUの命令時間、割込みを含む最大スタック、WCETは未測定である。
- 反復予算は呼出し前に決める。有限のforと状態フラグで演算を制御し、収束しない場合を通知する。これは分岐を含めたサイクル単位の定時間実行を意味しない。

## 2. 提案ごとの採否

| 提案 | 採用した内容 | 一律採用しなかった内容・条件 |
|---|---|---|
| QR-1 | compact Householder、Qの遅延生成、Qᵀの直接適用、固定sqrt精度 | 小行列ではGivensが速い場合があり、既存クラスを一律置換しない。粗いsqrt精度・大きい省略閾値を既定にしない |
| QR-2 | Givensの参照更新、再solve時のQ初期化、疎fill-inの消去、構造既知Hessenberg経路 | 構造の全検査を伴う専用経路は利益が消えたため不採用。専用APIの構造保証は利用側 |
| EIG-1 | Householderをベクトルとして左右から直接適用し、反射行列・汎用行列積を除去 | 複素solverのHessenberg保存は最終的な複素配列を直接使い、追加の実数密行列を作らない |
| EIG-2 | active blockの暗黙的QR、末尾2×2シフト、収束に基づくdeflation、継続、状態通知 | 実数クラスで複素対を実数2値として成功扱いにしない。複素対は既存のComplexクラスで処理。Francis二重シフトとの同時実装は選択していない |
| EIG-3 | 実対称専用のpacked cyclic Jacobi。値のみなら固有ベクトル領域なし | 一般非対称行列への適用、対称三重対角QRの同時導入は行わない |
| EIG-4 | 固有値・逆反復・内側GMRESの予算分離、実固有残差、尺度に応じた摂動、失敗伝播 | 逆反復で小さいKや2-passを無条件に使う案は不採用。一般非対称固有ベクトルへ直交性を強制しない |
| LU-1 | 毎列の部分ピボット、`PA=LU`、置換の公開、行列式の符号、特異状態 | scaled pivotは任意選択。ピボット省略を新しい既定にしない |
| LU-2 | compact LU、逆数共有、因子再利用、任意の逆対角キャッシュ | キャッシュはRAMを増やすため既定は無効。逆数が表現できない極小pivotは直接除算 |
| LU-3 | R回の反復改善、残差のみdoubleで計算する選択肢 | 同精度の反復改善は悪化例があり、自動適用しない。既定R=0 |
| CHOL-1 | 上三角ワーク共有、CSR下三角からの直接散布、出力参照API、packed保存 | 密出力のright-looking一律置換は不採用。既存APIはin-place内積型、packedはright-looking |
| CHOL-2 | 全pivotの精度・閾値方針統一、失敗後の更新停止、fallback、対角0→0、固定sqrt予算 | `d*rsqrt(d)`は一部double条件で有効な任意選択。既定は `1/rsqrt(d)`。入力閾値によるrsqrt戻り値のクランプはしない |
| CHOL-3 | SPD専用packed LDLᵀ、平方根なしの因子と右辺解法 | 不定値用の分解としては使わない。通常のCholesky出力へ黙って置換しない |
| GMRES-1 | Krylov基底nK、packed R、Hessenbergは1列、逆行列のRHS/work再利用、入力const参照 | 密な返却行列の必要領域は残る |
| GMRES-2 | 固定容量K、固定cycle数C、真の残差による再始動 | 小Kを一律の既定にしない。停滞する行列がある |
| GMRES-3 | MGSの1/2-pass選択、零残差・breakdown処理、最終実残差、共有逆ノルム | 2-passが真値への誤差を常に改善するとはしない。表現不能な逆ノルムは直接除算 |
| GMRES-4 | スケーリング付き実数/複素Givens、実数cと位相を含む複素s | 複素rhoの旧二乗値を維持する案は不採用。実残差ノルムに統一 |
| GMRES-5 | 任意の固定右対角前処理、任意のmatrix-free矩形経路 | 前処理の既定化、常時matrix-free化は不採用。既存矩形APIは明示Gramを維持 |

試作した不採用方式は比較用benchmarkに残し、製品APIの既定処理としては使わない。条件付きで採用した機能には有効条件と測定結果を併記する。

## 3. QR

決定的な密行列、3/8/16/32次元、16×5の矩形、Hessenberg、対角近傍をfloat/doubleで比較した。時間は各測定組の反復平均を3組計測した最小値。以下は成分別の評価記録の値で、実行環境の負荷による変動がある。

| double 16×16 | 時間 ns | solver保持量 byte | 相対再構成誤差 |
|---|---:|---:|---:|
| 改善前Givens | 16,715 | 4,104 | 2.01e-7 |
| 改善後Givens | 5,151 | 4,112 | 1.90e-14 |
| Householder・因子のみ | 1,563 | 2,192 | 1.28e-14 |
| Householder・Q生成込み | 2,567 | 2,192＋Q出力 | 1.28e-14 |

因子だけの経路にはQの密配列を保持しない。Qを返す場合は `M*M*sizeof(T)`、Rを展開する場合は `M*N*sizeof(T)` の出力領域を別に数える。`apply_Q_transpose_in_place` はQも追加ベクトルも作らない。

float3×3では新Givens約70ns、Householder因子約94ns。したがってHouseholderは新しい選択肢として追加した。sqrt回数0/1/2/4を比較し、既定2、より精度が必要なdouble向けに4を選べるようにした。再solve不整合と疎fill-inは回帰試験で修正を確認した。

[QRの全測定・追加候補・スタック内訳](../test_vs/algorithm_improvements/qr_results.md)

## 4. 固有値・固有ベクトル

明示反射行列を使ったHessenberg化を直接更新へ変更し、QR sweepも暗黙的な相似変換にした。active block外との結合部分も更新する。未収束のままブロックを確定しない。周期行列でシフト0が続くケースには、事前に決めた周期の補助シフトを使う。

double16の実数QR固有値では、旧約925µs・規格化固有値誤差0.497に対し、新約31µs・誤差4.4e-15。保持オブジェクトは8,504→4,352byteになった。旧版はこの例で固有値自体を誤るため、「同精度で約30倍」とは解釈しない。double8の対称Jacobiは、J=8で値のみ約12µs・336byte、ベクトルあり約13µs・848byteだった。

独立なNumPy照合は、3/4/6次元の乱数・近接固有値・重複固有値・巡回置換を含む45例。複素solver、S=512の予算で全例が収束し、最大の尺度付き固有値誤差は約1.9e-15だった。これは任意の行列に対する固定予算内収束の保証ではない。

実対称専用 `EigenSolverSymmetric<T,N,ComputeVectors>` を追加した。固有値だけの設定はpacked三角領域だけを保持し、固有ベクトルを求める設定は回転を蓄積する。一般非対称solverの逆反復はGMRESを維持した。

`iteration_max` は**1呼出し全体のsweep枠数**へ変更した。既定は `10*N`。旧版の「ブロックごと10回」と同じ意味ではない。逆反復回数と内側のK/C/Pは独立に指定する。`eigen_values_converged()` / `eigen_vectors_converged()` / `get_status()` を確認し、未収束時の対角近似値を確定固有値として扱わない。

[固有値の全測定・予算比較・スタック内訳](../test_vs/algorithm_improvements/eigen_results.md)

## 5. LU

通常行列では部分ピボットの探索分だけ遅くなる場合があるが、保持量と不安定なpivotに対する精度は改善した。L/Uを別々に保持せず、getter時に展開する。

| ケース | 改善前 | 改善後 |
|---|---:|---:|
| float16・solver保持量 | 2,184 byte | 1,168 byte |
| double16・solver保持量 | 4,232 byte | 2,192 byte |
| float16・通常行列の分解＋解 | 739ns | 887ns |
| float6・微小非零pivotの真値相対誤差 | 1.64e-1 | 2.09e-6 |

同じ因子から右辺だけを解くfloat16の処理は約180ns。逆対角キャッシュを使うと約153nsで、保持量は64byte増える。

float6のHilbert例では、同精度の補正R=0/1/2で誤差が `2.20e-3 → 1.04e-2 → 1.50e-2` と悪化した。一方、残差だけdoubleにすると `2.20e-3 → 1.65e-5 → 1.09e-7` と改善した。同精度の反復改善を既定にしない理由である。doubleのHilbertでは部分ピボット後の丸め誤差が旧版より大きい例も記録しており、全入力での誤差の単調改善を主張しない。

置換の向きは `P(i,pivot_indices[i])=1`、`PA=LU`。既存テストの固定L/U期待値はこの再構成とLの単位対角へ変更した。PythonNumpyにも `get_P()` と置換添字を公開した。

## 6. Cholesky / LDLᵀ

SPD行列を同じTで生成し、因子からの再構成をlong doubleで評価した。密なright-looking案に一律の利益はなく、既存dense/sparse APIはワークを共有する内積型を採用した。packed経路はright-lookingでRAMを減らす。

| float16 | 時間 ns | 相対再構成誤差 | 出力または保持量 |
|---|---:|---:|---:|
| 改善前dense | 514 | 1.88e-7 | dense出力1,024 byte |
| 改善後dense互換API | 523 | 1.87e-7 | dense出力1,024 byte |
| dense出力参照API | 406 | 1.87e-7 | dense出力1,024 byte |
| packed、root=1 | 636 | 1.87e-7 | solver556 byte |
| packed、root=3 | 749 | 4.17e-8 | solver556 byte |
| SPD LDLᵀ | 495 | 1.17e-7 | solver556 byte |

packed double16ではroot=1/2/3の再構成誤差が `1.35e-7 / 1.82e-14 / 3.17e-16`、時間は約675/745/781ns。LDLᵀは約505ns、誤差2.39e-16だった。通常の上三角Cholesky因子が必要な場合、LDLᵀからの変換に平方根と出力領域が再び必要になる。

`d*rsqrt(d)` はdouble/root=2で約745→706ns、誤差は同等であり、任意選択として残した。float/root=2では悪化例があり既定にしない。

旧tuple APIはfallbackを維持する。`cholesky_decomposition_to` / `cholesky_decomposition_sparse_to` は呼出し側の出力を直接更新し、失敗時は部分因子と失敗boolを返す。fallbackを自動で保持するAPIとは契約が異なる。疎入力の密コピー削除だけではコンパイラの一時領域配置によりスタックが減らない場合があり、この出力参照経路を追加した。

CholeskyのNewton初期値は既存mathのfloat変換を利用するため、正のpivotがnormal float範囲内という契約を維持する。double全指数範囲の保証はしない。LU/LDLᵀでは逆数が表現できないsubnormal pivotに対して直接除算を使う。

[LU・Choleskyの全測定・不採用案・スタック内訳](../test_vs/algorithm_improvements/lu_chol_results.md)

## 7. GMRES

解法はGMRESのまま維持した。packed RとHessenbergの1列ワークを共通コアへまとめ、実数・複素数・dense・CSRで共用する。最終Arnoldi段でも末尾のノルムと回転を計算し、K<Nで推定残差が誤って0になる問題を避ける。

`GmresSolver<T,N,K>` でKを固定容量として選び、`cycles`、`orthogonalization_passes`、絶対/相対許容値、`division_min`を設定できる。既存関数はK=N、C=1の互換入口である。初期残差零なら解を保持する。特異な小三角系では巨大な補正を生成せず、既知の解を保持してbreakdownを通知する。

複素回転は実数cと複素sを用いるユニタリ回転へ修正した。文書の再現例 `A=[[i,0],[1,1]], b=[1,0]` は、旧版が `[0,i]` とrho=0を返していたのに対し、新版は正解 `[-i,i]`、実残差0を返す。

矩形版は両方式を試し、既存APIでは明示Gramを維持した。任意の `gmres_k_rect_matrix_free` / `sparse_gmres_k_rect_matrix_free` はGram保存を省くが、長さMの一時ベクトルと毎回の2つの積が必要になる。MがN²より大きい場合はRAMまで増えるため一律採用しない。正規方程式の条件数二乗の問題は両方式に残る。返すrhoは元のA/b/xから直接計算した `norm(Aᵀ(b−Ax))` であり、不整合系の `norm(b−Ax)` が0になることは要求しない。

**返却値の意味の変更:** rhoは実数・複素数とも真の残差ノルム。`decay_rate`は右辺ノルムに対する相対許容値。旧版の複素二乗ノルムや初期残差に対する比を前提にした設定は見直す必要がある。状態や独立した絶対許容値が必要なら `GmresSolver` を使う。

時間は1000回の平均を独立3プロセスで計測した中央値。最終測定はホストの同じ論理CPUへ固定し、改善前後を交互に実行した。新版のクラス行は構築済みworkspaceを再利用し、対角前処理の準備時間も除く。旧関数は呼出しごとに内部workspaceを構築する。1回だけの呼出しと再利用時の時間を混同しない。

解誤差は、C++から `%a` で取り出した**実際のbinary32/binary64のA/b**をDecimal80桁で解いた参照値に対する `||x−x_ref||₂/||x_ref||₂`。120桁での再計算との一致も確認した。性能用CSVの `rms_target_error` は生成目標onesへの距離であり、特にHilbertでは数学的真値への誤差と異なる。

| 通常行列16×16 | K / C / P | 時間 ns | 真の解への相対誤差 | 新workspaceのsizeof byte |
|---|---|---:|---:|---:|
| float改善前 | 16 / 1 / 1 | 2,887 | 1.89e-4 | — |
| float改善後 | 16 / 1 / 1 | 5,501 | 5.10e-7 | 1,952 |
| float改善後 | 8 / 1 / 1 | 2,773 | 2.25e-7 | 880 |
| double改善前 | 16 / 1 / 1 | 3,481 | 1.39e-6 | — |
| double改善後 | 16 / 1 / 1 | 6,581 | 4.91e-16 | 3,864 |
| double改善後 | 8 / 1 / 1 | 2,890 | 1.91e-10 | 1,720 |

full Kの新コアは、スケーリング・実残差の再計算などにより遅くなる場合がある。K=8はこの通常行列では少ない時間・RAMで旧版より小さい解誤差を得た。ただし同じKを全行列に推奨する結果ではない。

float16の大きな配列だけでは旧3,200byteから新K=16で1,568byte、K=8で656byteへ削減した。上表のsizeofはさらに内部のベクトル・設定・状態を含むが、入力・出力・solve局所変数は含まない。`-fstack-usage` の旧benchmark関数フレームは5,152byte、新しい互換呼出し側は3,520byteで、別フレームの新solve本体480byteなどが加わる。フレームを区別した生データを保存しており、sizeof単体を最大RAMとはしていない。

尺度差のあるfloat16、K=8/C=4/P=1では、前処理なしの真の解誤差0.384が、固定右対角前処理で1.08e-6になった。追加保持量は64byte。極小対角で逆数が表現できない場合は、その成分を1として前処理を無効にする。

**改善しなかった例:** float Hilbert8では、試した12設定のうち最良の新版でも真の解誤差0.973で、旧版の0.746より悪かった。新版は未収束・breakdownを通知し、成功とはしない。これは精度改善の採用根拠に数えていない。両版とも正確な解を得られておらず、この用途に今回のfloat設定を推奨しない。MGSの2-passや前処理を一律に有効にしない理由でもある。double Hilbert8ではP=1からP=2で解誤差が2.11e-7→1.03e-7に改善するなど、条件付きの利益は確認した。

matrix-freeの比較では、double32×8の明示Gramが約2,253ns、matrix-freeが約2,961nsで、Gram512byteを中間ベクトル256byteへ置き換える交換関係だった。double128×4では明示版約1,353nsに対してmatrix-free約2,469ns、対応領域も128→1,024byteへ増えた。そのためmatrix-freeを既定とする案は不採用で、省RAMになる寸法・疎性・右辺数を選ぶ任意経路だけを採用した。

[丸め後の入力に対する独立した真値検証・悪化例](../test_vs/algorithm_improvements/gmres_oracle_results.md)と[最終測定の生データ](../test_vs/algorithm_improvements/results/)を参照。

## 8. 検証と再現

各アルゴリズムの回帰試験、既存のBase/PythonNumpy試験、ヘッダ依存とヒープ監視を同じ作業ツリーで実行する。再現用ソースとrunnerは [test_vs/algorithm_improvements](../test_vs/algorithm_improvements/) にまとめた。

```sh
python3 test_vs/algorithm_improvements/run_validation.py \
  --output /tmp/matrix_algorithm_results \
  --legacy-configs --sanitize --repeats 3
python3 test_vs/algorithm_improvements/eigen_numpy_oracle.py
```

runnerは指定commitとそのgitlinkのサブモジュールを作業ツリー外へarchiveし、改善前後を別々にビルドする。Decimal oracleも実行し、NumPy oracleは上記の別コマンドで実行する。既存CIの8マクロ構成は `-O0`、新しい回帰試験とbenchmarkは `-O2`、sanitizerは `-O1`。NumPyは独立oracleだけに必要である。

| 検証 | 結果 |
|---|---|
| 既存CIと同じ3マクロの8構成、Base/PythonNumpyのfloat/double | 全構成成功 |
| QR / eigen / LU・Cholesky / GMRESの追加回帰試験 | 全試験成功 |
| 追加回帰・確保監視にAddressSanitizer、UndefinedBehaviorSanitizer、float-divide-by-zero | 全試験成功 |
| 公開集約ヘッダ、`<cmath>`・`<algorithm>`をエラーにするヘッダ、`-fno-exceptions` | 成功 |
| 固定配列構成で新APIの構築・solve・getterを監視 | 動的確保0回 |
| utility既存試験・禁止ヘッダ/確保監視、4マクロ構成 | 全構成成功、確保0回 |
| NumPy独立oracle、45固有値問題 | 全例収束、最大規格化誤差1.89e-15 |
| Decimal80/120桁の独立GMRES oracle、18入力・252設定 | 参照解一致最大相対差1.40e-64。改善例と悪化・未収束例を記録 |
| 対象7ヘッダと依存ソースの禁止構文・標準copy/fillの検索 | 該当なし。cmathはmath側の標準数学マクロ有効時だけ |

受入れ試験は、正しい結果が求まるケースと、失敗・未収束を安全に返すケースを区別する。「全試験成功」は、上記252設定の全てが高精度解を得たという意味ではない。

依存監査では、実際のチェックアウトの `base_utility_cpp` にstd::copyと条件付きalgorithm includeが残っていた。提案書の置換済みという記述だけに依存せず、6箇所を等価な前方forへ置換した。既定の再帰コピーとmemcpy経路、旧 `BASE_UTILITY_USE_STD_COPY_` のマクロ名・コピー個数を維持した。**サブモジュール内の変更も今回の変更に含む。**

[results/README.md](../test_vs/algorithm_improvements/results/README.md) に最終測定のCSV、benchmarkのスタック情報、試験ログ、環境、ソースのハッシュをまとめた。巨大な既存テスト全体のスタック出力や実行バイナリはリポジトリへ格納せず、runnerの出力先に生成する。成分別レポートの時間は個別評価時の測定値であり、最終統合測定とは負荷や集計方法が異なる。僅差をMCUでの速度差として断定しない。
