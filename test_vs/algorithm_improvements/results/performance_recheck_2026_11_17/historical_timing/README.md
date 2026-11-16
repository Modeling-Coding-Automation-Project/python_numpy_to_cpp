# 前回の時間測定の保存データ

2026-11-12の測定時に作成された `/tmp/matrix_root_policy_timing_4et38633` から、2026-11-17に時間CSV40件と `environment.json` を回収した。QR、固有値、LU・Cholesky、GMRESの各benchmarkについて、before/afterを交互に5プロセスずつ測定した記録である。

比較元は `f50060e726c2fb03fa2a1bc44bc91c538966de96`、比較後は `879a31981d4cb99173939bd4a12df96bf8ccdf1a`。g++ 15.2.0、C++11、`-O2 -Wall -Wextra -Werror -DALGORITHM_IMPROVEMENTS`、論理CPU 0で取得した。詳細は [environment.json](environment.json) を参照。

回収前に、GMRESの10件すべてが既存の [2026-11-12_gmres_iterations](../../2026-11-12_gmres_iterations/README.md) のCSVとバイト単位で一致することを確認した。環境JSONも同ディレクトリの `timing_environment.json` と完全一致した。回収した全ファイルのSHA-256は [baseline_audit.json](../baseline_audit.json) に記録した。CSVの値や集計は変更していない。

このbefore/afterは平方根の公開APIへの委譲前後の比較であり、2026-11-17のコードとの比較ではない。現在の性能は、同じホスト条件で版を交互に再測定して比較する。

当時のこの時間測定ディレクトリにはスタック使用量の `.su` はない。以前保存された `.su` は2026-11-10の別のソース状態による関数単体のフレーム記録であり、呼出し全体のピークRAMを測ったものではない。旧ソースとの一致範囲と再現の限界は [baseline_notes.md](../baseline_notes.md) を参照。
