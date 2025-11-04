# flask_app.py
"""
FAQ_Gemini - 家庭内エージェント (FAQ Agent)

【エージェントの役割】
このエージェントは家庭内の出来事や家電に関する専門知識を持つナレッジベースエージェントです。
FAQデータベース、取扱説明書、家庭内デバイスの知識に基づいて質問に回答します。

【主な機能】
- RAGベースの質問応答（ベクトルデータベース検索）
- 他エージェントからの問い合わせ対応
- 会話履歴の分析と支援メッセージの生成

【連携エージェント】
- IoT Agent: IoTデバイスの制御と状態確認
- Browser Agent: Webブラウザ操作と情報検索

【エンドポイント】
- /rag_answer: ユーザーからの質問に回答
- /agent_rag_answer: 他エージェントからの問い合わせに応答
- /analyze_conversation: 会話履歴を分析して支援が必要か判断

参考: https://github.com/kota-kawa/Multi-Agent-Platform
"""

from flask import Flask, request, jsonify, render_template

from flask_cors import CORS

import os
import logging
from dotenv import load_dotenv

#import ai_engine  # AI/RAG ロジックをまとめた別モジュール
import ai_engine_faiss as ai_engine

# エージェント情報のインポート
from agent_registry import (
    registry as agent_registry,
    get_agent_info,
    get_optimal_agent_for_task,
    list_all_agents,
    AgentType
)

# ── 環境変数 / Flask 初期化 ──
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
logging.basicConfig(level=logging.DEBUG)



@app.after_request
def add_cors_headers(response):
    """Allow all domains to access the API without altering existing logic."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    request_headers = request.headers.get("Access-Control-Request-Headers")
    if request_headers:
        response.headers["Access-Control-Allow-Headers"] = request_headers
    else:
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    return response



@app.route("/rag_answer", methods=["POST"])
def rag_answer():
    """POST: { "question": "..." } → RAG で回答"""
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "質問を入力してください"}), 400
    try:
        answer, sources = ai_engine.get_answer(question)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        app.logger.exception("Error during query processing:")
        return jsonify({"error": str(e)}), 500


@app.route("/agent_rag_answer", methods=["POST"])
def agent_rag_answer():
    """他エージェントからの問い合わせに応答するが、会話履歴には保存しない"""
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "質問を入力してください"}), 400
    try:
        answer, sources = ai_engine.get_answer(question, persist_history=False)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        app.logger.exception("Error during external agent query processing:")
        return jsonify({"error": str(e)}), 500


@app.route("/reset_history", methods=["POST"])
def reset_history():
    """会話履歴をリセット"""
    ai_engine.reset_history()
    return jsonify({"status": "Conversation history reset."})


@app.route("/conversation_history", methods=["GET"])
def conversation_history():
    history = ai_engine.get_conversation_history()
    return jsonify({"conversation_history": history})


@app.route("/conversation_summary", methods=["GET"])
def conversation_summary():
    try:
        summary = ai_engine.get_conversation_summary()
        return jsonify({"summary": summary})
    except Exception as e:
        app.logger.exception("Error during conversation summarization:")
        return jsonify({"error": "会話の要約中にエラーが発生しました"}), 500


@app.route("/analyze_conversation", methods=["POST"])
def analyze_conversation():
    """
    外部エージェントから会話履歴を受け取り、VDBの知識で解決できる問題があれば支援メッセージを返す。
    
    リクエスト形式:
    {
        "conversation_history": [
            {"role": "User", "message": "..."},
            {"role": "AI", "message": "..."}
        ]
    }
    
    レスポンス形式:
    {
        "analyzed": true,
        "needs_help": true/false,
        "problem": "特定された問題" (needs_helpがtrueの場合),
        "support_message": "支援メッセージ" (needs_helpがtrueの場合),
        "sources": [...] (needs_helpがtrueの場合)
    }
    """
    try:
        data = request.get_json()
    except Exception as e:
        app.logger.exception("Error parsing JSON:")
        return jsonify({"error": "無効なJSON形式です"}), 400
    
    if data is None:
        return jsonify({"error": "リクエストボディが必要です"}), 400
    
    conversation_history = data.get("conversation_history", [])
    
    if not conversation_history:
        return jsonify({"error": "会話履歴が空です"}), 400
    
    if not isinstance(conversation_history, list):
        return jsonify({"error": "conversation_historyはリスト形式で送信してください"}), 400
    
    try:
        # 会話履歴を分析
        analysis = ai_engine.analyze_external_conversation(conversation_history)
        
        response = {
            "analyzed": True,
            "needs_help": analysis.get("needs_help", False)
        }
        
        # エラーがあればログに記録するが、レスポンスには含めない（セキュリティのため）
        if "error" in analysis:
            app.logger.warning(f"Analysis error: {analysis['error']}")
        
        # 支援が必要な場合、VDBから回答を取得
        if analysis.get("needs_help"):
            # LLMからの出力を安全に取得（例外情報は含まれない）
            problem = analysis.get("problem", "")
            question = analysis.get("question", "")
            
            # 安全性のため、problemとquestionが文字列であることを確認
            if not isinstance(problem, str):
                problem = ""
            if not isinstance(question, str):
                question = ""
            
            response["problem"] = problem
            
            if question:
                # 既存のRAGロジックを使用して回答を生成
                try:
                    answer, sources = ai_engine.get_answer(question)
                    response["support_message"] = answer
                    response["sources"] = sources
                except Exception:
                    app.logger.exception("Error getting answer from VDB:")
                    response["support_message"] = "回答の取得中にエラーが発生しました。"
                    response["sources"] = []
            else:
                response["support_message"] = "問題は特定されましたが、具体的な質問が生成されませんでした。"
                response["sources"] = []
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.exception("Error during conversation analysis:")
        return jsonify({"error": "会話の分析中にエラーが発生しました"}), 500


@app.route("/")
def index():
    """トップページ（テンプレートは従来どおり）"""
    return render_template("index.html")


@app.route("/agents", methods=["GET"])
def list_agents():
    """
    利用可能なエージェント一覧を返す
    
    他のエージェントがこのエンドポイントを使用して、
    システム内で利用可能なエージェントとその機能を確認できます。
    
    レスポンス形式:
    {
        "agents": [
            {
                "type": "faq",
                "display_name": "家庭内エージェント",
                "description": "...",
                "base_urls": [...],
                "capabilities": [...]
            },
            ...
        ]
    }
    """
    agents_info = []
    for agent_type in [AgentType.FAQ, AgentType.IOT, AgentType.BROWSER]:
        summary = agent_registry.get_agent_endpoints_summary(agent_type)
        if summary:
            agents_info.append(summary)
    
    return jsonify({
        "agents": agents_info,
        "total": len(agents_info)
    })


@app.route("/agents/<agent_type>", methods=["GET"])
def get_agent_details(agent_type: str):
    """
    特定のエージェントの詳細情報を取得
    
    Args:
        agent_type: エージェントタイプ (faq, iot, browser)
        
    レスポンス形式:
    {
        "agent_type": "iot",
        "display_name": "IoT エージェント",
        "description": "...",
        "base_urls": [...],
        "endpoints": [...]
    }
    """
    agent_info = get_agent_info(agent_type)
    if not agent_info:
        return jsonify({"error": f"エージェント '{agent_type}' が見つかりません"}), 404
    
    # AgentTypeからsummaryを取得
    try:
        agent_enum = AgentType(agent_type)
        summary = agent_registry.get_agent_endpoints_summary(agent_enum)
        return jsonify(summary)
    except ValueError:
        return jsonify({"error": f"無効なエージェントタイプ: {agent_type}"}), 400


@app.route("/agents/suggest", methods=["POST"])
def suggest_agent():
    """
    タスク内容から最適なエージェントを提案
    
    他のエージェントが「どのエージェントに助けを求めるべきか」を判断する際に使用。
    
    リクエスト形式:
    {
        "task": "部屋の温度を25度に設定したい"
    }
    
    レスポンス形式:
    {
        "suggested_agent": "iot",
        "display_name": "IoT エージェント",
        "reason": "温度制御はIoTエージェントが担当します",
        "base_url": "https://iot-agent.project-kk.com",
        "endpoint": "/api/chat"
    }
    """
    data = request.get_json() or {}
    task = data.get("task", "").strip()
    
    if not task:
        return jsonify({"error": "タスク内容を指定してください"}), 400
    
    optimal_agent = get_optimal_agent_for_task(task)
    if not optimal_agent:
        return jsonify({"error": "適切なエージェントが見つかりませんでした"}), 404
    
    # 推奨理由を生成
    reason_map = {
        AgentType.FAQ: "家電の使い方や知識ベース検索はFAQエージェントが担当します",
        AgentType.IOT: "IoTデバイスの制御や状態確認はIoTエージェントが担当します",
        AgentType.BROWSER: "Web検索やブラウザ操作はブラウザエージェントが担当します"
    }
    
    # chatエンドポイントを探す
    chat_endpoint = None
    for cap in optimal_agent.capabilities:
        if cap.name == "chat" and cap.endpoint:
            chat_endpoint = cap.endpoint.url
            break
    
    return jsonify({
        "suggested_agent": optimal_agent.agent_type.value,
        "display_name": optimal_agent.display_name,
        "reason": reason_map.get(optimal_agent.agent_type, "このエージェントが適切です"),
        "base_url": optimal_agent.base_urls[0],
        "endpoint": chat_endpoint or "/api/chat"
    })


if __name__ == "__main__":
    # インデックス読み込みは ai_engine 側で一度だけ行われる
    app.run(host="0.0.0.0", port=5000, debug=True)
