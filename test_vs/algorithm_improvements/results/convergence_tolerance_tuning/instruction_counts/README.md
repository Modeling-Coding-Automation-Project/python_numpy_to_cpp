# TMP版とfor版の動的な四則演算命令数

調査日: 2026-11-17。

**機械語の四則演算命令の実行回数は、今回の20ケース中16ケースで同じだった。複素行列積の4ケースではTMP版の方が多かった。** for版がpacked SIMDを使い、1命令で複数成分を計算するためである。一方、命令の全laneへ換算した算術数は、この4ケースではTMP版の方が少なかった。機械語の命令数、SIMDの成分数、結果に必要な数学的演算数は区別する必要がある。

これは静的な逆アセンブルの行数比較ではない。Linuxの`PTRACE_SINGLESTEP`で実際の実行を1命令ずつ進め、各停止位置の命令を逆アセンブルと照合した。繰り返し同じ命令位置へ来た回数も加算している。20ケース×TMP/forの40呼出しを追跡した。[Linux ptraceの単一命令実行](https://man7.org/linux/man-pages/man2/ptrace.2.html)

## 比較条件

- 固定した本体: `6b1d627edf2de6e3e56deca59b83b37e942a97aa`。前回RAM測定のb599と同じ`base_matrix`内容。Gitから隔離展開し、作業ツリーの変更は参照していない。
- Linux x86-64、g++ 15.2.0、`-std=c++11 -O2 -Wall -Wextra -Werror`、既定数学ライブラリ。for版だけに`BASE_MATRIX_USE_FOR_LOOP_OPERATION_`を追加した。
- 同じhost専用probeを2回ビルドした。公開の`GmresOperation::multiply`、`Numerical::norm`、`GmresSolver::solve`を呼び、製品ヘッダには計測処理を追加していない。
- N=8/16、float/double、実数/複素数の行列ベクトル積とnorm、加えて実数GMRESの4設定。すべて`active_size=N`。
- GMRESは相対・絶対許容誤差0、breakdown係数8e-12、`division_min=1e-25`、`K=N, cycles=1, passes=1`。**新しい既定許容誤差の採用前に条件を固定した比較であり、新既定値で停止を早めたGMRESの命令数ではない。**
- 1回warmup後の1呼出しを、host用の`noinline/noipa`境界間で追跡した。呼出し先の数学処理も含め、入力生成・出力処理は除く。境界のcall/return等は総step数には含むが、FP四則命令には含まれない。
- 40呼出しすべてで、通常実行と追跡実行の出力が一致した。20ケースすべてで、TMP/for間の格納入力・出力・GMRESの状態がbit単位で一致した。

## 何を数えたか

| 指標 | 定義 |
| --- | --- |
| FP四則命令実行数 | `addss/addsd/addps/addpd`等の浮動小数点加減乗除命令を実行した回数。packed命令でも1実行を1回とする |
| lane換算四則数 | scalarは1成分、128bit packedはfloat4成分・double2成分として加算。結果へ使わない成分や重複成分への計算も含む |
| 実行されたFP命令位置数 | 追跡で訪れた異なるFP四則命令アドレスの個数。命令の実行回数ではない |
| 整数・アドレス計算 | 整数の加減乗除、shift、bit演算、`lea`等。ループ添字だけでなく、アドレス生成・stack調整・数学処理のbit操作も含む |
| 比較・分岐・call/return | FP四則命令と分離して記録 |

Intelの命令定義でも、例えば`ADDSD`は下位のdouble1成分、`ADDPS`の128bit形式はfloat4成分に作用する。したがって命令数と成分数を同じものとして比較しない。[Intel命令セットリファレンス](https://cdrdv2-public.intel.com/812383/253666-sdm-vol-2a.pdf)

FMAは「命令1回、乗算と加減算を各laneで1回ずつ」として別集計する実装だが、今回の40呼出しではFMA実行は0だった。比較、min/max、符号を反転するbit演算、load/store、平方根等の専用命令も四則命令へ混ぜていない。normが呼ぶ既定数学ライブラリ内の加減乗除は含めている。

総step数はPMUのretired-instruction数やCPU内部のmicro-op数ではない。GMRES double16には`REP`文字列命令があり、単一stepでは同じ命令位置に反復ごとの停止が発生した。そのstepも保存し、`repeat_string_step_events`へ別記した。**この点はFP四則命令数には影響しない。**

## 動的な実行回数

同じ数値のfloat/doubleはまとめて表示した。全20ケースの型別・加減乗除別の値は[dynamic_counts.csv](dynamic_counts.csv)、前後差は[comparison.csv](comparison.csv)にある。

| 処理 | 型 | N | FP四則命令 TMP / for | lane換算 TMP / for |
| --- | --- | ---: | ---: | ---: |
| 実数行列積 | float/double | 8 | 128 / 128 | 128 / 128 |
| 実数行列積 | float/double | 16 | 512 / 512 | 512 / 512 |
| 複素行列積 | float | 8 | 512 / 320 | 512 / 1280 |
| 複素行列積 | float | 16 | 2039 / 1280 | 2066 / 5120 |
| 複素行列積 | double | 8 | 512 / 320 | 512 / 640 |
| 複素行列積 | double | 16 | 2039 / 1280 | 2048 / 2560 |
| 実数norm | float/double | 8 | 41 / 41 | 41 / 41 |
| 実数norm | float/double | 16 | 65 / 65 | 65 / 65 |
| 複素norm | float/double | 8 | 65 / 65 | 65 / 65 |
| 複素norm | float/double | 16 | 113 / 113 | 113 / 113 |
| full GMRES | float | 8 | 3689 / 3689 | 4565 / 4565 |
| full GMRES | float | 16 | 20573 / 20573 | 27125 / 27125 |
| full GMRES | double | 8 | 3981 / 3981 | 4565 / 4565 |
| full GMRES | double | 16 | 22757 / 22757 | 27125 / 27125 |

GMRESは両版とも8次が8反復、16次が16反復だった。許容誤差0なので収束フラグはfalse、breakdownもfalseで、反復数・報告残差・出力すべてが一致した。ここでは同じ仕事量で命令数を比較しており、このprobe自体を真値精度の検証とは扱わない。

### 静的な命令位置数との違い

実数double8行列積のFP四則命令位置は、TMPでは128箇所、forでは2箇所だった。一方、**実行回数は両方128回**である。for版の乗算・加算の各1箇所が64回ずつ実行されるため、128対2という静的な差をそのまま計算回数の差にはできない。

原記録: [TMPの実行位置](template_multiply_real_double_8.executed.csv)、[forの実行位置](for_multiply_real_double_8.executed.csv)。各行に命令位置・命令・operand・実際の訪問回数を保存した。

### 複素行列積で命令数とlane数が逆転した理由

for版のdouble8では、1要素の複素積と加算に次の5命令が実行され、各位置への訪問回数は64回だった。

```asm
431e: mulpd xmm0,xmm2
432b: mulpd xmm1,xmm2
4333: subpd xmm2,xmm1
4337: addpd xmm0,xmm1
433f: addpd xmm0,xmm3
```

この間の並べ替え命令等は抜粋から省略した。2成分同時の加算と減算で、必要な実部・虚部に加えて最終結果へ使わない成分も計算し、その後に必要な成分を選んでいる。よって5命令×64=320回、2lane×320=640成分分となる。TMP版の同ケースはscalarの512命令で512成分分だった。[for版の全実行位置](for_multiply_complex_double_8.executed.csv)、[TMP版の全実行位置](template_multiply_complex_double_8.executed.csv)

floatのpacked命令は4laneなので、未利用laneも含む換算値がさらに大きくなる。これは必要な数学的演算が増えたという意味ではない。TMPのcomplex float16にある2066という値も、9回のpacked加算が結果に使わない上位laneまで計算することを含む。double16では同じ9回が2laneなので2048となる。

## RAMの2例との関係

前回RAM増を調べたGMRES double8と同じ行列・RHS・予算にした今回のGMRES比較では、四則命令3981回、lane換算4565回は同じだった。整数・アドレス計算はTMP1610回 / for2881回、整数比較1047回 / 2167回、分岐1464回 / 2478回だった。FP計算の回数と、ループ制御・途中値の保持に使う命令は別々に変わっている。

Eigen complex double16のRAM説明に現れる行列積についても、同じ型・次元のkernelを今回測定した。ただし、今回のkernel入力は固定の非零虚部を含む行列・ベクトルであり、固有ベクトル計算中の全中間入力を追跡したものではない。Eigen全体の四則命令数を測ったという主張はしない。

今回のバイナリは専用host wrapperで再ビルドしたため、前回RAM計測バイナリと関数の配置・inline化が同一とは限らない。**同じ四則演算回数でもstack予約量が同じとは限らない**という説明と今回の結果は矛盾しないが、今回の命令数から前回の400 byte差を再算出したわけではない。前回の実バイナリで確認した予約・保存命令は[assembly_notes.md](../codegen/assembly_notes.md)を参照。

命令1回あたりの時間、依存関係、並列実行、load/store、キャッシュ等も時間に影響するため、命令数だけから実時間の比やRAMの増減を推定しない。今回の追跡中の経過時間は性能測定値として使用していない。MCUのコンパイル・実機命令追跡も行っていない。

## 保存物と再実行

- [instruction_probe.cpp](instruction_probe.cpp): 同じ入力と設定で呼び出すhost専用probe。
- [count_instructions.py](count_instructions.py): 固定commit展開、2構成のビルド、ptrace追跡、命令分類。
- [summarize_counts.py](summarize_counts.py): 保存済み命令位置別データを再集計し、比較表を作る。
- [metadata.json](metadata.json): commit・全ヘッダhash・compiler・実行ファイルhash・ビルドコマンド。
- [trace_reports.json](trace_reports.json): ケースごとの動的mnemonic内訳・集計・通常実行との一致。
- [validation.json](validation.json)、[summary.json](summary.json): 入出力・状態の一致、有限性、原記録からの再集計一致。
- `template.asm` / `for.asm`: 同じ実行ファイルの静的逆アセンブル。
- `*.executed.csv`: 実際に訪れた命令位置と訪問回数。`*.output.txt`: 格納入力と出力の16進表現。

```sh
# Linux x86-64、g++、binutils、Python 3が必要。
# 自分の子プロセスに対するptraceが許可された環境で実行する。
python3 test_vs/algorithm_improvements/results/convergence_tolerance_tuning/instruction_counts/count_instructions.py
python3 test_vs/algorithm_improvements/results/convergence_tolerance_tuning/instruction_counts/summarize_counts.py
```

再集計のみなら2つ目のコマンドでよく、コンパイル・実行・追跡は不要。今回の追跡は別の実時間測定を停止してから行った。
