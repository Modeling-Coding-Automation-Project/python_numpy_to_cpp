# 収束許容誤差の調整：測定データ

調査日：2026-11-17。ここは本体や既定値を変更しなかった初回調査の記録である。その後の追加依頼で既定値を選定・採用した結果は、[本文レポート](../../../../document/12_convergence_tolerance_tuning.md)と[採用後検証](../balanced_tolerance_selection/README.md)を参照。

| 保存先 | 内容 |
| --- | --- |
| [gmres](gmres/README.md) | 相対許容誤差10値、2442状態の真値誤差・実残差・状態、候補の直列計時 |
| [eigen](eigen/README.md) | 92ケースでの探索、事前固定候補の別サイズ48ケース検証、同一実行ファイルでの時間比較 |
| [codegen](codegen/assembly_notes.md) | 前回の実行ファイルを逆アセンブルし、固定領域・フレーム予約・書込み水位を区別したRAM調査 |
| [integrity_validation.json](integrity_validation.json) | 全データの行数・候補合格数・probe hash・本体無変更の確認 |

現行の比較元は`6b1d627edf2de6e3e56deca59b83b37e942a97aa`。本体は前回の`b599e846`と同じ内容。GMRESの履歴比較は`879a31981d4cb99173939bd4a12df96bf8ccdf1a`を同じ新probeで再測定している。codegenだけは前回測定したバイナリの静的解析であり、新しい時間・スタック測定ではない。

計算結果を探索用許容誤差とは独立した固定基準で評価した。GMRESの真値は実際の格納A・bをDecimal80/120桁で解き、Eigenは格納AのNumPy固有値と独立固有対残差を使う。収束フラグだけの改善を精度改善として数えない。

## 再実行

リポジトリのルートで実行する。ホスト側にg++、Git、Python 3.12以降、tasksetが必要。Eigenの評価にはNumPyも必要で、今回の環境は2.5.4。各runnerは指定commitを一時領域に展開する。時間測定は他のコンパイル・品質探索と同時に実行しない。

```sh
# GMRESの品質探索と、候補だけの7回計時。
python3 test_vs/algorithm_improvements/gmres_tolerance_sweep.py --phase explore
python3 test_vs/algorithm_improvements/gmres_tolerance_sweep.py \
  --phase timing --taus 0 1e-12 1e-10 1e-7 1e-6 --repeats 7

# 固有値rtolの単独探索。
python3 test_vs/algorithm_improvements/eigen_tolerance_sweep.py \
  --work /tmp/eigen_tolerance_reproduction --label rtol

# vtol・内部GMRES許容誤差の追加探索。
python3 test_vs/algorithm_improvements/eigen_tolerance_sweep.py \
  --work /tmp/eigen_tolerance_reproduction --reuse-build --label joint \
  --profiles test_vs/algorithm_improvements/results/convergence_tolerance_tuning/eigen/eigen_tolerance_joint_profiles.json

# 保存済み候補を、選定には使わなかった5/7/12次で確認。
python3 test_vs/algorithm_improvements/eigen_tolerance_sweep.py \
  --work /tmp/eigen_tolerance_reproduction --holdout --label holdout \
  --profiles test_vs/algorithm_improvements/results/convergence_tolerance_tuning/eigen/eigen_tolerance_selected_profiles.json
```

保存データを上書きしたくない場合、各runnerの`--output`で別ディレクトリを指定する。Eigenの計時コマンドと詳細は[eigen/eigen_tolerance_notes.md](eigen/eigen_tolerance_notes.md)を参照。試験ソースにある標準数学・動的なホスト処理・計時機能は組み込み本体の依存ではない。

固有値の候補決定は別サイズ検証の前に記録した。別サイズでの悪化もそのまま保存し、探索集合の成功を全入力への保証として扱っていない。RAMに関して今回直接確認したのは固定保存量の不変と過去の生成コードであり、許容誤差候補ごとのピークスタック再測定はしていない。
