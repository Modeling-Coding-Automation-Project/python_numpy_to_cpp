# バランスを考慮した既定値の選定・採用後検証

更新日：2026-11-17。[本文レポート](../../../../document/12_convergence_tolerance_tuning.md)を参照。

前回の「全入力に一律の改善はない」という調査に対し、今回は改善と悪化の両方を含めて既定値を選定し、本体に反映した。収束許容誤差4定数だけを変更し、β・シフト・反復上限・演算方式は維持している。

| 定数 | 採用値 |
| --- | ---: |
| 単独GMRESの相対許容誤差 | `1e-12` |
| 固有値の相対許容誤差 | `1e-6` |
| 固有ベクトルの相対許容誤差 | `4e-6` |
| 内部GMRESの相対許容誤差 | `1e-3` |

| 保存先 | 内容 |
| --- | --- |
| [Eigen](eigen/README.md) | 固定値30組＋旧ε由来2設定、140件の選定集合、新次数48件、7組の時間測定 |
| [GMRES](../convergence_tolerance_tuning/balanced_defaults/gmres/balanced_gmres_decision.md) | 前回の全データを同じ精度基準で再集計し、保守的な`1e-12`を選定。旧API17箇所も監査 |
| [機械語命令の比較](../convergence_tolerance_tuning/instruction_counts/README.md) | TMP/forの20ケース×2構成を追跡し、四則演算命令・SIMD lane・ループ等の制御命令を区別 |
| [RAM](memory/README.md) | 変更前と採用後を再ビルドし、全112比較で固定保存量・観測stack・既知フレーム列が同一 |
| [採用後検証](validation/validation.json) | 24実行すべて成功。各実行の`.txt`と、本体・試験ソースのSHA-256を保存 |

## 採用後の検証を再実行

```sh
python3 test_vs/algorithm_improvements/run_balanced_parameter_validation.py \
  --output /tmp/balanced_parameter_validation --legacy-configs --sanitize
```

このrunnerは作業ツリーの本体を検証する。既定数学/ホスト標準数学とTMP/forのGMRES・演算helper・Eigenを12実行、禁止ヘッダ/例外なしを2実行、サニタイザを2実行、既存CI構成を8実行する。新しい8次三重対角の試験は、実際の既定パラメータ・既定予算を用い、解析的な固有値と独立残差を確認する。既存assertは緩めていない。

数値探索・時間・命令・RAMの個別再現手順は各保存先を参照。時間測定は他のコンパイル・追跡と同時実行していない。採用後の本体はvalidationおよびmemoryのヘッダhashで特定できる。過去の測定データは保存時の条件を表すため、今回の既定値とは区別する。
