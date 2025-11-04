# マルチエージェントシステム実装サマリー

## 概要

このPRでは、Multi-Agent-Platform、IoT-Agent、web_agent02を参考に、FAQ_Geminiにマルチエージェントシステムのサポートを追加しました。

## 実装内容

### 1. エージェントレジストリ (`agent_registry.py`)

全エージェントの情報を一元管理するモジュールを実装：

- **AgentInfo**: エージェントの詳細情報（役割、機能、接続先）
- **AgentCapability**: エージェントが提供する機能の定義
- **AgentEndpoint**: APIエンドポイントの情報
- **AgentRegistry**: 全エージェントの管理とクエリ機能

#### 主な機能

```python
from agent_registry import get_agent_info, get_optimal_agent_for_task

# エージェント情報の取得
agent = get_agent_info("iot")

# タスクに最適なエージェントの選択
optimal = get_optimal_agent_for_task("部屋の温度を下げて")
```

### 2. エージェント情報

#### FAQエージェント（本リポジトリ）
- **役割**: 家庭内の知識ベースエージェント
- **接続先**: `http://localhost:5000`
- **主な機能**: RAG回答、会話分析、他エージェントサポート

#### IoTエージェント
- **リポジトリ**: https://github.com/kota-kawa/IoT-Agent
- **役割**: IoTデバイスの制御と状態確認
- **接続先**: `https://iot-agent.project-kk.com`
- **主な機能**: デバイス制御、センサーデータ取得、会話評価

#### ブラウザエージェント
- **リポジトリ**: https://github.com/kota-kawa/web_agent02
- **役割**: Webブラウザ自動操作
- **接続先**: `http://localhost:5005`
- **主な機能**: Web検索、ブラウザ操作、情報抽出

### 3. 新規APIエンドポイント (`app.py`)

#### `/agents` (GET)
利用可能な全エージェントの情報を返す

```bash
curl http://localhost:5000/agents
```

#### `/agents/{agent_type}` (GET)
特定エージェントの詳細情報を返す

```bash
curl http://localhost:5000/agents/iot
```

#### `/agents/suggest` (POST)
タスクから最適なエージェントを提案

```bash
curl -X POST http://localhost:5000/agents/suggest \
  -H "Content-Type: application/json" \
  -d '{"task": "部屋の温度を下げたい"}'
```

### 4. ドキュメント

- **AGENTS_INFO.md**: 各エージェントの詳細仕様とエンドポイント
- **USAGE_EXAMPLES.md**: 使用例とエージェント間通信のパターン
- **test_agent_registry.py**: テストスイート

## エージェント間通信の例

### 例1: FAQエージェント → IoTエージェント

```python
# ユーザー: 「部屋の温度を25度に設定して」
# 1. FAQエージェントが最適なエージェントを判定
optimal = get_optimal_agent_for_task("部屋の温度を25度に設定して")
# → IoTエージェント

# 2. IoTエージェントに転送
response = requests.post(
    f"{optimal.base_urls[0]}/api/chat",
    json={"messages": [{"role": "user", "content": "部屋の温度を25度に設定して"}]}
)
```

### 例2: IoTエージェント → FAQエージェント

```python
# ユーザー: 「エアコンのエラーコードE01の意味は？」
# IoTエージェントが知識をFAQエージェントに照会
response = requests.post(
    "http://localhost:5000/agent_rag_answer",
    json={"question": "エアコンのエラーコードE01の意味"}
)
```

## テスト結果

全てのテストがパス：

```bash
$ python test_agent_registry.py
============================================================
FAQ_Gemini エージェントレジストリテスト
============================================================

✓ モジュールインポート成功
✓ エージェント情報取得
✓ 別名によるエージェント取得
✓ タスクからのエージェント選択
✓ 全エージェント一覧取得
✓ 能力によるエージェント検索
✓ エンドポイント情報取得

全てのテストが成功しました! ✓
```

## 参考リポジトリ

1. **Multi-Agent-Platform**: https://github.com/kota-kawa/Multi-Agent-Platform
   - マルチエージェントオーケストレーターの実装参考
   - エージェントの役割定義とエイリアスの構造

2. **IoT-Agent**: https://github.com/kota-kawa/IoT-Agent  
   - エンドポイント構造とデバイス管理API
   - 会話履歴評価とIoT操作の実装

3. **web_agent02**: https://github.com/kota-kawa/web_agent02
   - ブラウザ自動化とストリーミングAPI
   - エージェント間リレー機能

## ファイル構成

```
FAQ_Gemini/
├── agent_registry.py           # エージェントレジストリ（新規）
├── app.py                       # Flask app（更新 - エージェント発見API追加）
├── AGENTS_INFO.md              # エージェント仕様書（新規）
├── USAGE_EXAMPLES.md           # 使用例とパターン（新規）
├── IMPLEMENTATION_SUMMARY.md   # 実装サマリー（新規）
└── test_agent_registry.py      # テストスイート（新規）
```

## 変更点まとめ

### 追加ファイル
- `agent_registry.py`: 529行 - エージェント情報管理
- `AGENTS_INFO.md`: 各エージェントの詳細ドキュメント
- `USAGE_EXAMPLES.md`: 実装パターンと使用例
- `test_agent_registry.py`: テストスイート
- `IMPLEMENTATION_SUMMARY.md`: このファイル

### 変更ファイル
- `app.py`: 
  - ヘッダーコメントにエージェントの役割説明を追加
  - agent_registryのインポート追加
  - 3つの新規エンドポイント追加（/agents, /agents/{type}, /agents/suggest）

## 使い方

### Pythonコード内での使用

```python
from agent_registry import get_agent_info, get_optimal_agent_for_task

# エージェント情報取得
agent = get_agent_info("iot")
print(agent.display_name)  # → "IoT エージェント"

# 最適なエージェント選択
optimal = get_optimal_agent_for_task("Webで天気を調べて")
print(optimal.agent_type.value)  # → "browser"
```

### REST API経由での使用

```bash
# 全エージェント一覧
curl http://localhost:5000/agents

# IoTエージェントの詳細
curl http://localhost:5000/agents/iot

# エージェント提案
curl -X POST http://localhost:5000/agents/suggest \
  -H "Content-Type: application/json" \
  -d '{"task": "部屋の温度を確認したい"}'
```

## 今後の拡張可能性

1. **LLMベースのエージェント選択**: 現在はキーワードベースだが、LLMで判断することも可能
2. **エージェント能力の動的取得**: 各エージェントから能力情報を動的に取得
3. **エージェントヘルスチェック**: 各エージェントの稼働状態を監視
4. **負荷分散**: 複数インスタンス間でのリクエスト分散
5. **会話履歴の共有**: エージェント間での会話コンテキスト共有

## まとめ

Multi-Agent-Platformのアーキテクチャを参考に、FAQ_Geminiが他のエージェントと協調動作できるよう拡張しました。これにより：

✅ 各エージェントの役割が明確に定義された  
✅ エージェント情報をプログラムで取得・利用可能  
✅ タスクに応じた最適なエージェントの自動選択が可能  
✅ エージェント間通信の基盤が整備された  
✅ Multi-Agent-Platformとの統合準備が完了  

すべての変更は後方互換性を保ちつつ、新機能として追加されています。
