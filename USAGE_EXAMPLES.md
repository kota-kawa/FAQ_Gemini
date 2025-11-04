# エージェント間通信の使用例

このドキュメントでは、新しく追加されたエージェントレジストリとエージェント間通信機能の使用方法を具体例とともに説明します。

## 概要

FAQ_Geminiに以下の機能が追加されました：

1. **エージェントレジストリ** (`agent_registry.py`) - 全エージェントの情報を一元管理
2. **エージェント発見API** - 利用可能なエージェントとその機能を照会
3. **エージェント提案API** - タスクに最適なエージェントを自動選択
4. **詳細なエージェント情報** - 各エージェントの役割、エンドポイント、機能

## 使用例

### 1. Pythonコードでの使用

#### 基本的な使い方

```python
from agent_registry import (
    get_agent_info,
    get_optimal_agent_for_task,
    list_all_agents,
    registry
)

# エージェント情報を取得
faq_agent = get_agent_info("faq")
print(f"名前: {faq_agent.display_name}")
print(f"説明: {faq_agent.description}")
print(f"接続先: {faq_agent.base_urls[0]}")

# タスクから最適なエージェントを選択
task = "部屋の温度を25度に設定したい"
optimal = get_optimal_agent_for_task(task)
print(f"推奨エージェント: {optimal.display_name}")

# 全エージェントのリスト表示
print(list_all_agents())
```

#### エンドポイント情報の取得

```python
from agent_registry import registry, AgentType

# IoTエージェントのエンドポイント情報を取得
iot_info = registry.get_agent_endpoints_summary(AgentType.IOT)

print(f"エージェント名: {iot_info['display_name']}")
print(f"接続先: {iot_info['base_urls'][0]}")
print("\nエンドポイント:")
for endpoint in iot_info['endpoints']:
    print(f"  {endpoint['capability']}: {endpoint['url']}")
    print(f"    {endpoint['description']}")
```

#### 能力による検索

```python
from agent_registry import registry

# chatエンドポイントを持つエージェントを検索
chat_agents = registry.get_agent_by_capability("chat")
for agent in chat_agents:
    print(f"{agent.display_name} が chat をサポート")
```

### 2. REST APIでの使用

#### 全エージェント一覧の取得

```bash
curl http://localhost:5000/agents
```

レスポンス例：
```json
{
  "agents": [
    {
      "type": "faq",
      "display_name": "家庭内エージェント",
      "description": "家庭内の出来事や家電に関する専門知識...",
      "base_urls": ["http://localhost:5000"],
      "endpoints": [
        {
          "capability": "rag_answer",
          "description": "ベクトルデータベースから関連情報を検索し、質問に回答",
          "url": "/rag_answer",
          "method": "POST"
        },
        ...
      ]
    },
    ...
  ],
  "total": 3
}
```

#### 特定エージェントの詳細取得

```bash
curl http://localhost:5000/agents/iot
```

#### 最適なエージェントの提案

```bash
curl -X POST http://localhost:5000/agents/suggest \
  -H "Content-Type: application/json" \
  -d '{"task": "Webで今日の天気を調べたい"}'
```

レスポンス例：
```json
{
  "suggested_agent": "browser",
  "display_name": "ブラウザエージェント",
  "reason": "Web検索やブラウザ操作はブラウザエージェントが担当します",
  "base_url": "http://localhost:5005",
  "endpoint": "/api/chat"
}
```

### 3. エージェント間通信の実装例

#### 例1: FAQエージェントからIoTエージェントへの問い合わせ

```python
import requests
from agent_registry import get_optimal_agent_for_task

def handle_user_request(user_message: str):
    """ユーザーリクエストを処理し、必要なら他のエージェントに転送"""
    
    # 1. 最適なエージェントを判定
    optimal_agent = get_optimal_agent_for_task(user_message)
    
    # 2. 自分（FAQエージェント）で処理すべきか判定
    if optimal_agent.agent_type.value == "faq":
        # FAQエージェント自身で処理
        return process_locally(user_message)
    
    # 3. 他のエージェントに転送
    base_url = optimal_agent.base_urls[0]
    
    if optimal_agent.agent_type.value == "iot":
        # IoTエージェントに問い合わせ
        response = requests.post(
            f"{base_url}/api/chat",
            json={"messages": [{"role": "user", "content": user_message}]}
        )
        return response.json()["reply"]
    
    elif optimal_agent.agent_type.value == "browser":
        # ブラウザエージェントに問い合わせ
        response = requests.post(
            f"{base_url}/api/chat",
            json={"prompt": user_message, "new_task": True}
        )
        return response.json()["run_summary"]
```

#### 例2: IoTエージェントからFAQエージェントへの知識照会

```python
import requests

def get_device_manual_info(device_name: str, question: str):
    """デバイスのマニュアル情報をFAQエージェントに問い合わせ"""
    
    faq_url = "http://localhost:5000"
    
    # agent_rag_answerエンドポイントを使用（会話履歴に残さない）
    response = requests.post(
        f"{faq_url}/agent_rag_answer",
        json={"question": f"{device_name}の{question}"}
    )
    
    result = response.json()
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }

# 使用例
info = get_device_manual_info("エアコン", "フィルター掃除の方法")
print(info["answer"])
```

#### 例3: ブラウザエージェントへの会話履歴送信

```python
import requests

def check_if_browser_needed(conversation_history):
    """会話履歴からブラウザ操作が必要か確認"""
    
    browser_url = "http://localhost:5005"
    
    response = requests.post(
        f"{browser_url}/api/check-conversation-history",
        json={"conversation_history": conversation_history}
    )
    
    result = response.json()
    return result.get("needs_browser_action", False)

# 使用例
history = [
    {"role": "user", "content": "今日の天気は？"},
    {"role": "assistant", "content": "調べてみます"}
]

if check_if_browser_needed(history):
    print("ブラウザエージェントに検索を依頼します")
```

### 4. Multi-Agent-Platformとの統合

Multi-Agent-Platformからこのエージェントを利用する場合：

```python
# Multi-Agent-Platformのコード例
import requests

def call_faq_agent(question: str):
    """FAQエージェントに質問"""
    response = requests.post(
        "http://faq_gemini:5000/rag_answer",
        json={"question": question}
    )
    return response.json()["answer"]

def analyze_conversation_with_faq(conversation_history):
    """会話履歴をFAQエージェントで分析"""
    response = requests.post(
        "http://faq_gemini:5000/analyze_conversation",
        json={"conversation_history": conversation_history}
    )
    
    result = response.json()
    if result["needs_help"]:
        return {
            "support_needed": True,
            "problem": result["problem"],
            "solution": result["support_message"]
        }
    return {"support_needed": False}

def discover_available_agents():
    """利用可能なエージェントを発見"""
    response = requests.get("http://faq_gemini:5000/agents")
    return response.json()["agents"]
```

## 実装パターン

### パターン1: ルーティング型

エージェントがユーザーリクエストを受け取り、適切なエージェントにルーティング：

```python
def route_request(user_input: str):
    # 1. agent/suggest APIで最適なエージェントを取得
    suggestion = requests.post(
        "http://localhost:5000/agents/suggest",
        json={"task": user_input}
    ).json()
    
    # 2. 提案されたエージェントにリクエスト送信
    agent_url = suggestion["base_url"] + suggestion["endpoint"]
    response = requests.post(agent_url, json={"prompt": user_input})
    
    return response.json()
```

### パターン2: コンサルテーション型

エージェントが必要に応じて他のエージェントに助けを求める：

```python
def process_with_consultation(task: str):
    # まず自分で処理を試みる
    result = try_process_locally(task)
    
    if result.needs_external_knowledge:
        # FAQエージェントに知識を照会
        faq_response = requests.post(
            "http://localhost:5000/agent_rag_answer",
            json={"question": result.knowledge_query}
        ).json()
        
        # 取得した知識を使って処理を完了
        return complete_with_knowledge(result, faq_response)
    
    return result
```

### パターン3: 協調型

複数のエージェントが協力してタスクを完了：

```python
def collaborative_task(user_request: str):
    # 1. FAQエージェントで背景知識を取得
    knowledge = requests.post(
        "http://localhost:5000/agent_rag_answer",
        json={"question": user_request}
    ).json()
    
    # 2. ブラウザエージェントで最新情報を検索
    web_info = requests.post(
        "http://localhost:5005/api/chat",
        json={"prompt": f"{user_request} の最新情報", "new_task": True}
    ).json()
    
    # 3. IoTエージェントで実際の操作を実行
    action = requests.post(
        "https://iot-agent.project-kk.com/api/chat",
        json={"messages": [
            {"role": "user", "content": user_request}
        ]}
    ).json()
    
    # 4. 結果を統合
    return {
        "knowledge": knowledge["answer"],
        "current_info": web_info["run_summary"],
        "action_result": action["reply"]
    }
```

## テスト

実装されたテストを実行：

```bash
cd /home/runner/work/FAQ_Gemini/FAQ_Gemini
python test_agent_registry.py
```

期待される出力：
```
============================================================
FAQ_Gemini エージェントレジストリテスト
============================================================

=== モジュールインポートのテスト ===
✓ agent_registry.py のインポート成功
✓ AgentInfo dataclass 確認
...

全てのテストが成功しました! ✓
============================================================
```

## トラブルシューティング

### エージェントに接続できない

```python
from agent_registry import get_agent_info

agent = get_agent_info("iot")
for url in agent.base_urls:
    try:
        response = requests.get(f"{url}/api/devices/ping", timeout=5)
        if response.ok:
            print(f"接続成功: {url}")
            break
    except Exception as e:
        print(f"接続失敗 {url}: {e}")
```

### 環境変数で接続先を変更

```bash
# .env ファイルに追加
FAQ_GEMINI_API_BASE=http://your-faq-server:5000
IOT_AGENT_API_BASE=http://your-iot-server:5006
BROWSER_AGENT_API_BASE=http://your-browser-server:5005
```

## まとめ

この実装により、以下が可能になりました：

1. ✅ 各エージェントの役割と能力を明確に定義
2. ✅ エージェント情報をプログラムで取得・利用
3. ✅ タスクに応じた最適なエージェントの自動選択
4. ✅ エージェント間の協調動作の実装基盤
5. ✅ Multi-Agent-Platformとのシームレスな統合

詳細な仕様は `AGENTS_INFO.md` を参照してください。
