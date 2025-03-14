# リアルタイム制御向けC++行列ライブラリ

## 特徴

- 密行列、対角行列、スパース行列を使い分けることで計算時間、メモリ消費を削減可能
- 反復計算の回数調整が可能
- ゼロ割、マイナス平方根などを回避する機能を実装
- Python NumPyの行列計算コードから簡単に置き換えられる
  - 生成AIを用いて変換を自動化できるように

### 対応言語バージョン

C++11

## 実装済み機能

- 行列の和、差、積、結合
- GMRES(k)法による左除算、逆行列
- コレスキー分解
- LU分解、行列式
- QR分解
- 固有値、固有ベクトル

## 使い方

「sample」ディレクトリに使い方の例を示しています。

また、詳細については記事をご参照ください。

https://note.com/claude_a/n/n131a654a4c41

## サポート

新規にissueを作成して、詳細をお知らせください。

## 貢献

コミュニティからのプルリクエストを歓迎します。もし大幅な変更を考えているのであれば、提案する修正についての議論を始めるために、issueを開くことから始めてください。

また、プルリクエストを提出する際には、関連するテストが必要に応じて更新または追加されていることを確認してください。

## ライセンス

[MIT License](./LICENSE.txt)
