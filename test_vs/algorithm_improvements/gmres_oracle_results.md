# GMRES：丸め後の入力に対する独立した真値検証

以下は2026-11-10の実装に対する検証記録。2026-11-11以降は共有ノルムなどの平方根を数学ライブラリの公開APIへ委譲しており、この記録は変更後の実装の精度を示すものではない。

`gmres_oracle.py` は標準PythonのDecimal80桁で、C++が実際に保持するA/bに対する参照解を計算する。`gmres_oracle_probe.cpp` はA/b/xを `%a` で出力し、binary32/binary64の値を失わずPythonへ渡す。RHSは性能測定と同じ `A * ones` に1000回の交互摂動を適用した後の値であり、理想化したHilbertや理想化した右辺へ置き換えない。

18種類の入力（float/double、N=4/8/16、fixture=0/1/2）について252件の旧/新設定を比較した。80桁解は120桁の独立した再計算と相対 `1.40e-64` 以内で一致した。参照解の相対残差は最大 `6.90e-80`。生データは `gmres_oracle_results.csv`、再現時の出力には正確なhex入力・出力とメタデータも保存する。

```sh
python3 test_vs/algorithm_improvements/gmres_oracle.py \
  --baseline /tmp/matrix_algorithm_baseline --output /tmp/gmres_oracle_results
```

比較指標は `||x-x_ref||₂ / ||x_ref||₂` と、Decimalで再評価した `||b-Ax||₂ / ||b||₂`。CSVの `manufactured_target_rms_error` は元のonesに対する誤差であり、前者と区別する。`converged` / `breakdown` の-1は、旧形式のtuple APIから状態を取得できないことを意味する。

## 真値との比較が支持する改善

| 入力 | 旧版の相対解誤差 | 新版の設定と相対解誤差 |
|---|---:|---:|
| float regular N=8 | 7.18e-4 | full K, P1: 1.58e-7 / P2: 1.04e-7 |
| float regular N=16 | 1.89e-4 | half K, C1, P1: 2.25e-7 |
| double regular N=16 | 1.39e-6 | half K, C1, P1: 1.91e-10 |
| double scaled N=16 | 2.07e-5 | full K, P1: 4.42e-15 |
| float scaled N=16 | 2.61e-5 | half K, C4, P1, diagonal: 1.08e-6 |
| double Hilbert N=8 | 8.20e-1 | full K, P1: 2.11e-7 / P2: 1.03e-7 |

小さい行列のfloatでは、一部の設定で旧版より誤差が大きい。例としてregular N=4は旧2.00e-7、新full P1が2.13e-7、P2は1.38e-7。全入力・全設定で精度が改善するとはしない。

## Hilbertでonesを真値と見なせない例と失敗例

| 入力 | 丸め後の参照解とonesのRMS距離 | 旧版の相対解誤差 | 新full P1 | 新full P2 |
|---|---:|---:|---:|---:|
| float N=8 | 14.1222 | 0.7460 | 0.9730 | 1.5455 |
| float N=16 | 12.7841 | 1.6575 | 0.9997 | 9.8529 |
| double N=16 | 2.7043 | 12403.75 | 0.9318 | 2.1841 |

**float Hilbert N=8では、調べた12種類の新版設定に真の解誤差の改善がない。** 最良でもfull K=8/C1/P1/前処理なしの0.9730で、旧0.7460より悪い。全候補はfull/half K、P1/P2、half KのC1/C4、各設定の対角前処理あり/なしを含む。旧版・新版とも高精度の解は得られていない。

この設定の真の相対残差は旧 `3.14e-7`、新 `1.65e-7` と小さくなるが、解誤差は改善しない。新版は `converged=0` / `breakdown=1` を返し、成功扱いにはしない。残差の小ささだけを精度改善の根拠にすることも避ける。

float Hilbert N=16ではfull K/P1/対角前処理の相対解誤差0.6768が最良だったが、これも `converged=0` / `breakdown=1`。double Hilbert N=16のfull P1もbreakdownを報告する。これらは実用的な高精度解の実証ではない。

P2を常に使う方針や対角前処理の一律適用は、この検証からも採用できない。用途ごとの設定選択は支持する一方、極端な悪条件を含めた「真値への精度が全例で同等以上」という主張は支持しない。精度限界と安全な未収束通知を、RAM・時間の改善とは分けて報告する。
