# 採用した固定しきい値のRAM確認

**測定した全112比較で、保持量・観測stack水位・最大既知callchainに差はなかった。** テンプレート構成56例、for構成56例を、変更前と採用後の両方で同じソース・同じflagsから再ビルドして比較した。過去に保存した数値を今回の変更前の値として転用していない。

## 比較対象

- 変更前: `6b1d627edf2de6e3e56deca59b83b37e942a97aa`。
- 採用後: 同commitをHEADとするworking treeの`base_matrix`を隔離ディレクトリへコピー。
- 採用後の定数: 独立GMRES tau=`1e-12`、Eigen rtol=`1e-6`、vtol=`4e-6`、内部GMRES tolerance=`1e-3`。GMRES breakdown=`8e-12`、逆反復shift=`3e-4`は維持。
- 依存先: 両方とも6b1d627が参照するsubmoduleを使用。変更されたheaderは`base_matrix_eigen_solver.hpp`と`base_matrix_gmres.hpp`の2個だけ。

コピー前後とビルド完了時にworking headerのhash一致を検査した。`header_manifest.json`には両snapshotの依存先を含む37個すべてのheader hashを保存している。採用後の2headerは次のhashで識別する。

| header | SHA-256 |
| --- | --- |
| `base_matrix_eigen_solver.hpp` | `1c155ca8cf52924234bbe0a61a06cbfccccb760bf1816a2646f32bd4cc7cb5e8` |
| `base_matrix_gmres.hpp` | `a34b819c4cd80ecf5c7e7264602f34d2312d854a10feb34543de7b04eb4e07a9` |

## 測定と結果

既存の`peak_stack_probe.cpp`、`peak_stack_worker.cpp`、worker headerをそのまま使った。3ソースが6b1d627の内容と一致することも確認した。g++ 15.2.0、C++11、`-O2 -Wall -Wextra -Werror -fstack-usage -fcallgraph-info=su -pthread`、既定数学、固定配列。workerだけに`-mno-red-zone -fno-shrink-wrap -fno-shrink-wrap-separate`を付け、ライブラリの演算関数へ計測hookを挿入していない。

各型float/double、N=8/16で、GMRES full/restarted、Eigen real/complexの値のみ／ベクトル込み、Jacobi両形式、Givens QR、Householder因子／Q生成、LU、dense/packed Choleskyを測定した。

| 指標 | テンプレート構成: 差分0 / 比較数 | for構成: 差分0 / 比較数 |
| --- | ---: | ---: |
| solverの`sizeof` | 56 / 56 | 56 / 56 |
| 入力・出力・solverを含むcontextの`sizeof` | 56 / 56 | 56 / 56 |
| 呼出し先を含む最大stack書込み深さ | 56 / 56 | 56 / 56 |
| GCC call graphの最大既知frame合計 | 56 / 56 | 56 / 56 |
| context＋観測stack | 56 / 56 | 56 / 56 |
| 未解析外部callの集合 | 56 / 56 | 56 / 56 |

主構成であるテンプレート版の代表値。単位はbyteで、すべて変更前と採用後で同じ。solverはcontextにすでに含まれており、合計へ重複加算しない。

| 処理 | solver | context | 観測stack | context＋観測stack |
| --- | ---: | ---: | ---: | ---: |
| GMRES full float16 | 1952 | 3168 | 600 | 3768 |
| GMRES full double16 | 3864 | 6296 | 992 | 7288 |
| Eigen real double16、ベクトル込み | 4352 | 6400 | 7488 | 13888 |
| Eigen complex double16、ベクトル込み | 8576 | 10624 | 14328 | 24952 |
| Jacobi double16、ベクトル保存 | 3184 | 5232 | 120 | 5352 |
| Householder double16、因子のみ | 2192 | 4240 | 104 | 4344 |
| LU double16、分解＋解法 | 2192 | 4496 | 240 | 4736 |
| Packed Cholesky double16 | 1112 | 3160 | 16 | 3176 |

4バイナリのcalibrationは空関数8 byte、外側256 byte＋内側4096 byteの入れ子呼出し4,416 byteで一致した。全232測定行（56処理＋校正2種、2版×2構成）を保存。2種類のpaint値×3回で最小・最大が違ったのは、テンプレート構成のcomplex固有値float16の変更前後2行だけで、223～224 byteの1 byte差。両方とも最大224 byteで比較している。

## 指標の限界

stack paintは書込み位置を観測するため、未書込みの予約領域や全入力の最悪経路は保証しない。同じバイナリのGCC call graphも調べ、既知frame列と未解決の`__stack_chk_fail`、`memmove`、`memset`を区別した。既知frameへleaf red zoneの128 byteを加えた補助値もCSVへ保存しているが、外部callが未解析の場合に完全な上限とは扱わない。

結論は**このホスト・コンパイラ・代表入力で、今回の定数変更によるRAM増加は観測されなかった**というもの。組み込み実機のABI、割込みstack、全入力に対する最悪RAMが等しいことまで断定しない。詳細な方法は[peak_stack_method.md](../../../memory/peak_stack_method.md)を参照。

## 保存物と再実行

- `metadata.json`: 共通flags、compiler、probe hash、両版のビルドコマンド。
- `header_manifest.json`: 全header hash、変更header一覧、隔離snapshotの説明。
- `*.raw.csv`: 校正・書込み水位の原記録。
- `*.su` / `*.ci` / `call_chains.json`: 同じ演算objectの静的frame情報とcall graph。
- `peak_stack_cases.csv` / `peak_stack_comparison.csv`: 保持量・観測値・既知frameの統合値と差分。
- `validation.json`: 件数、全差分0、既存probeとの一致、校正結果。
- `worker_disassembly.txt` / `worker.su`: 計測workerの監査情報。

```sh
python3 test_vs/algorithm_improvements/memory/peak_stack_balanced.py --stage all
```

既定ではtemplate/forの両方を測る。`--loop-modes template`で主構成だけに絞れる。時間測定と重ならないよう`--stage compile`、`--stage measure`、`--stage analyze`へ分けることもできる。working treeのheaderを対象にするため、再実行時に変更されたheaderは新しいhashで記録される。
