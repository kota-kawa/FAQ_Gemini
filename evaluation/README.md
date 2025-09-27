# Reranker Evaluation Assets

このディレクトリには日本語 FAQ リランカーの評価結果と設定ファイルを配置します。

## known_questions.jsonl

- 既知質問セットを JSON Lines 形式で保存します。
- 各行は以下のキーを持つ JSON オブジェクトです。
  - `question`: ユーザーからの既知の質問文。
  - `relevant_sources`: 正解とみなすドキュメントの `source` 値 (文字列) の配列。
- 例は `known_questions.sample.jsonl` を参照してください。
- 本ファイルが存在しない場合、`ai_engine_gemini.py` の自動評価処理はスキップされます。

## latest_rerank_metrics.json

- 自動評価の結果 (recall@k, MRR など) を JSON 形式で出力します。
- `get_answer` 実行時の再ランキング精度を継続的に監視する用途を想定しています。

## query_pipeline_stats.json

- Gemini によるクエリ拡張の呼び出し回数やレイテンシーなど、コスト関連の統計を記録します。
- 値は `ai_engine_gemini.py` の `QueryCostMonitor` により自動更新されます。
