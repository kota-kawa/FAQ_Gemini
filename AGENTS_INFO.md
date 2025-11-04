# エージェント情報とエンドポイント

このドキュメントは、FAQ_Geminiが連携する各エージェントの役割、機能、接続先エンドポイントを説明します。

## マルチエージェントシステムの概要

このシステムは3つの専門エージェントで構成されています：

1. **FAQエージェント（家庭内エージェント）** - 本リポジトリ
2. **IoTエージェント** - IoTデバイス制御
3. **ブラウザエージェント** - Webブラウザ操作

各エージェントは独立して動作し、必要に応じて他のエージェントに助けを求めることができます。

## 1. FAQエージェント（家庭内エージェント）

### 役割
家庭内の出来事や家電に関する専門知識を持つナレッジベースエージェント。FAQデータベース、取扱説明書、家庭内デバイスの知識に基づいて質問に回答します。

### 主な機能
- RAGベースの質問応答（ベクトルデータベース検索）
- 他エージェントからの問い合わせ対応  
- 会話履歴の分析と支援メッセージの生成

### 接続先
```
デフォルト: http://localhost:5000
Docker: http://faq_gemini:5000
環境変数: FAQ_GEMINI_API_BASE
```

### エンドポイント

#### `/rag_answer` (POST)
ユーザーからの質問にRAGで回答
```json
// リクエスト
{
  "question": "エアコンの設定温度は何度がおすすめですか？"
}

// レスポンス
{
  "answer": "一般的に26-28度が推奨されます...",
  "sources": ["doc1.pdf", "doc2.pdf"]
}
```

#### `/agent_rag_answer` (POST)
他エージェントからの問い合わせに応答（会話履歴に保存しない）
```json
// リクエスト
{
  "question": "照明の色温度設定について教えて"
}

// レスポンス
{
  "answer": "色温度は...",
  "sources": [...]
}
```

#### `/analyze_conversation` (POST)
会話履歴を分析し、VDBの知識で解決できる問題があれば支援
```json
// リクエスト
{
  "conversation_history": [
    {"role": "User", "message": "洗濯機が動かない"},
    {"role": "AI", "message": "確認してみます"}
  ]
}

// レスポンス
{
  "analyzed": true,
  "needs_help": true,
  "problem": "洗濯機のトラブル",
  "support_message": "エラーコードを確認してください...",
  "sources": [...]
}
```

#### `/conversation_history` (GET)
会話履歴を取得

#### `/conversation_summary` (GET)
会話の要約を取得

#### `/reset_history` (POST)
会話履歴をリセット

#### `/agents` (GET)
利用可能なエージェント一覧を取得
```json
// レスポンス
{
  "agents": [
    {
      "type": "faq",
      "display_name": "家庭内エージェント",
      "description": "...",
      "base_urls": ["http://localhost:5000"],
      "endpoints": [...]
    },
    ...
  ],
  "total": 3
}
```

#### `/agents/{agent_type}` (GET)
特定エージェントの詳細情報を取得

#### `/agents/suggest` (POST)
タスク内容から最適なエージェントを提案
```json
// リクエスト
{
  "task": "部屋の温度を25度に設定したい"
}

// レスポンス
{
  "suggested_agent": "iot",
  "display_name": "IoT エージェント",
  "reason": "温度制御はIoTエージェントが担当します",
  "base_url": "https://iot-agent.project-kk.com",
  "endpoint": "/api/chat"
}
```

---

## 2. IoTエージェント

### 役割
IoTデバイスの状態確認と操作を行うエージェント。Raspberry Piなどのエッジデバイスと連携し、センサーデータの取得、アクチュエーターの制御、デバイスの管理を担当します。

### 主な機能
- 自然言語によるIoTデバイス制御
- センサーデータの取得と監視
- デバイスの登録と管理
- 他エージェントからの問い合わせ対応
- 会話履歴の評価とIoT操作の実行

### 接続先
```
本番環境: https://iot-agent.project-kk.com
ローカル: http://localhost:5006
環境変数: IOT_AGENT_API_BASE
```

### リポジトリ
https://github.com/kota-kawa/IoT-Agent

### エンドポイント

#### `/api/chat` (POST)
自然言語でIoTデバイスと対話
```json
// リクエスト
{
  "messages": [
    {"role": "user", "content": "リビングの照明をつけて"}
  ]
}

// レスポンス
{
  "reply": "リビングの照明をオンにしました"
}
```

#### `/api/agents/respond` (POST)
他エージェントからのスポット問い合わせに応答
```json
// リクエスト
{
  "request": "現在の室温は？",
  "metadata": {...},
  "include_device_snapshot": true
}

// レスポンス
{
  "reply": "現在の室温は23.5度です",
  "device_snapshot": [...]
}
```

#### `/api/conversations/review` (POST)
会話履歴を評価し、必要ならIoT操作を実行
```json
// リクエスト
{
  "history": [
    {"role": "user", "content": "部屋が暑い"},
    {"role": "assistant", "content": "エアコンをつけましょうか？"}
  ]
}

// レスポンス
{
  "analysis": {
    "action_required": true,
    "reason": "室温調整が必要",
    "suggested_device_commands": [...]
  },
  "action_taken": true,
  "execution_reply": "エアコンを26度に設定しました"
}
```

#### `/api/devices/register` (POST)
新しいデバイスを登録

#### `/api/devices` (GET)
登録済みデバイス一覧を取得

#### `/api/devices/<device_id>` (GET)
特定デバイスの詳細を取得

#### `/api/devices/<device_id>/jobs` (POST)
デバイスにジョブを送信

---

## 3. ブラウザエージェント

### 役割
Webブラウザを自動操作するエージェント。ウェブページの閲覧、情報検索、フォーム入力、ボタンクリックなどのブラウザ操作を自然言語の指示で実行します。

### 主な機能
- 自然言語によるブラウザ操作
- Web検索と情報抽出
- フォーム操作とデータ入力
- リアルタイム操作進捗のストリーミング
- 他エージェントからの会話履歴チェック

### 接続先
```
デフォルト: http://localhost:5005
Docker: http://browser-agent:5005
環境変数: BROWSER_AGENT_API_BASE
```

### リポジトリ
https://github.com/kota-kawa/web_agent02

### エンドポイント

#### `/api/chat` (POST)
自然言語でブラウザ操作を実行
```json
// リクエスト
{
  "prompt": "Googleで今日の天気を検索して",
  "new_task": true
}

// レスポンス
{
  "run_summary": "天気予報を検索しました。今日は晴れで最高気温25度です。",
  "messages": [...]
}
```

#### `/api/check-conversation-history` (POST)
会話履歴をチェックしてブラウザ操作が必要か判断
```json
// リクエスト
{
  "conversation_history": [
    {"role": "user", "content": "今日の天気は？"},
    {"role": "assistant", "content": "調べてみます"}
  ]
}

// レスポンス
{
  "needs_browser_action": true,
  "suggested_task": "天気予報サイトで今日の天気を確認"
}
```

#### `/api/agent-relay` (POST)
他エージェントからのリレー要求を処理

#### `/api/history` (GET)
ブラウザエージェントの操作履歴を取得

#### `/api/stream` (GET)
リアルタイムで操作進捗を取得（Server-Sent Events）

#### `/api/reset` (POST)
ブラウザセッションをリセット

#### `/api/pause` (POST)
ブラウザ操作を一時停止

#### `/api/resume` (POST)
ブラウザ操作を再開

---

## エージェント間通信の例

### 例1: FAQエージェントがIoTエージェントに助けを求める

1. ユーザーが「部屋の温度を下げて」と依頼
2. FAQエージェントが `/agents/suggest` で最適なエージェントを確認
3. 返答: IoTエージェントが推奨される
4. FAQエージェントがIoTエージェントの `/api/chat` にリクエスト
5. IoTエージェントがエアコンを制御して結果を返す

### 例2: IoTエージェントがFAQエージェントに助けを求める

1. ユーザーが「エアコンのエラーコードE01の意味は？」と質問
2. IoTエージェントが `/agent_rag_answer` で知識を問い合わせ
3. FAQエージェントがナレッジベースから回答を生成
4. IoTエージェントがユーザーに回答を返す

### 例3: ブラウザエージェントが会話履歴を分析

1. Multi-Agent-Platformが会話履歴をブラウザエージェントに送信
2. ブラウザエージェントが `/api/check-conversation-history` で評価
3. Web検索が必要と判断した場合、自動で検索を実行
4. 検索結果を含む応答を返す

---

## プログラムでのエージェント情報の使用

### agent_registry.py の使用例

```python
from agent_registry import (
    get_agent_info,
    get_optimal_agent_for_task,
    list_all_agents,
    AgentType
)

# エージェント情報を取得
faq_agent = get_agent_info("faq")
print(f"名前: {faq_agent.display_name}")
print(f"接続先: {faq_agent.base_urls[0]}")

# タスクから最適なエージェントを選択
optimal = get_optimal_agent_for_task("部屋の温度を25度にして")
print(f"推奨エージェント: {optimal.display_name}")

# 全エージェントのリスト
print(list_all_agents())
```

### APIエンドポイントでの使用例

```python
import requests

# 利用可能なエージェント一覧を取得
response = requests.get("http://localhost:5000/agents")
agents = response.json()["agents"]

# 最適なエージェントを提案してもらう
response = requests.post(
    "http://localhost:5000/agents/suggest",
    json={"task": "Webで情報を検索したい"}
)
suggestion = response.json()
print(f"推奨: {suggestion['suggested_agent']}")
print(f"理由: {suggestion['reason']}")
print(f"接続先: {suggestion['base_url']}{suggestion['endpoint']}")
```

---

## 環境変数

各エージェントの接続先は環境変数で設定可能：

```bash
# FAQエージェント
export FAQ_GEMINI_API_BASE="http://localhost:5000"

# IoTエージェント
export IOT_AGENT_API_BASE="https://iot-agent.project-kk.com"

# ブラウザエージェント  
export BROWSER_AGENT_API_BASE="http://localhost:5005"
```

---

## 参考リンク

- [Multi-Agent-Platform](https://github.com/kota-kawa/Multi-Agent-Platform) - マルチエージェントオーケストレーター
- [IoT-Agent](https://github.com/kota-kawa/IoT-Agent) - IoTデバイス制御エージェント
- [web_agent02](https://github.com/kota-kawa/web_agent02) - ブラウザ自動化エージェント
