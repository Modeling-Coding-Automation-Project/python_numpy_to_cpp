# Eigen収束許容誤差の調整結果

[調査ノート](eigen_tolerance_notes.md)に、固定条件、探索、候補固定後の検証、計算時間とRAMの範囲をまとめた。本体とTMP構成は変更していない。

型共通候補 `(rtol,vtol,inner)=(1e-6,4.8e-6,1e-3)` は探索92件で品質・収束を満たしたが、未使用次数では合格がAの34/48から30/48へ低下した。一律に性能・品質が改善する設定ではない。

| 保存物 | 内容 |
| --- | --- |
| `eigen_tolerance_rtol_*` | rtol単独13設定、1,196件 |
| `eigen_tolerance_joint_*` | vtol/inner追加10設定、920件 |
| `eigen_tolerance_selected_profiles.json` | Aと固定した型共通候補 |
| `eigen_tolerance_selection.json` | 未使用次数を測る前の候補固定記録 |
| `eigen_tolerance_holdout_*` | N5/7/12、A/候補96件 |
| `timing/` | 同一binary・CPU固定・7組の140行比較、raw・品質結合集計・`.su` |
| `eigen_tolerance_integrity_verification.json` | archive全ファイルのGit照合、旧/new runner hashの区別 |

各フェーズの `_raw/` は入力・出力のhex表現、`_metrics.csv` は独立誤差とフラグ、`_summary.csv` は集計、`_metadata.json` は固定コミット・依存先・source hash・コンパイル条件を記録している。
