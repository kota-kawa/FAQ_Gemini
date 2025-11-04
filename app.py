# flask_app.py
from flask import Flask, request, jsonify, render_template

from flask_cors import CORS

import os
import logging
from dotenv import load_dotenv

from external_agents import AgentDelegator, AgentDispatchError

#import ai_engine  # AI/RAG ロジックをまとめた別モジュール
import ai_engine_faiss as ai_engine

# ── 環境変数 / Flask 初期化 ──
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
logging.basicConfig(level=logging.DEBUG)


agent_delegator = AgentDelegator()



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


@app.route("/delegate_agent_help", methods=["POST"])
def delegate_agent_help():
    """Forward a help request to the most suitable external agent."""

    data = request.get_json(silent=True) or {}

    request_text = ""
    for key in ("question", "prompt", "request", "message", "instruction"):
        raw_value = data.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            request_text = raw_value.strip()
            break

    if not request_text:
        return jsonify({"error": "問い合わせ内容を入力してください。"}), 400

    hint_raw = data.get("agent_hint") or data.get("agent") or data.get("target_agent")
    hint = hint_raw.strip() if isinstance(hint_raw, str) else None

    metadata_raw = data.get("metadata")
    if metadata_raw is not None and not isinstance(metadata_raw, dict):
        return jsonify({"error": "metadata はオブジェクト形式で指定してください。"}), 400

    supplemental_context = data.get("context") or data.get("supplemental_context")
    if supplemental_context is not None and not isinstance(supplemental_context, str):
        return jsonify({"error": "context は文字列で指定してください。"}), 400

    try:
        delegation = agent_delegator.delegate_help_request(
            request_text,
            hint=hint,
            metadata=metadata_raw if isinstance(metadata_raw, dict) else None,
            supplemental_context=supplemental_context,
        )
    except AgentDispatchError as exc:
        response = {"error": str(exc)}
        if exc.details:
            response["details"] = exc.details
        return jsonify(response), 502
    except Exception:
        app.logger.exception("Unexpected error during external agent delegation:")
        return (
            jsonify({"error": "外部エージェントへの問い合わせ中にエラーが発生しました。"}),
            500,
        )

    return jsonify(delegation)


@app.route("/delegate_agent_conversation", methods=["POST"])
def delegate_agent_conversation():
    """Send a conversation log to an external agent's review endpoint."""

    data = request.get_json(silent=True) or {}
    history = data.get("conversation_history") or data.get("history")

    if history is None:
        return jsonify({"error": "conversation_history を指定してください。"}), 400

    if not isinstance(history, list):
        return jsonify({"error": "conversation_history はリスト形式で指定してください。"}), 400

    agent_hint = data.get("agent") or data.get("agent_hint")
    hint = agent_hint.strip() if isinstance(agent_hint, str) else None

    try:
        delegation = agent_delegator.forward_conversation(history, agent=hint)
    except AgentDispatchError as exc:
        response = {"error": str(exc)}
        if exc.details:
            response["details"] = exc.details
        return jsonify(response), 502
    except Exception:
        app.logger.exception("Unexpected error during conversation delegation:")
        return (
            jsonify({"error": "外部エージェントへの会話転送中にエラーが発生しました。"}),
            500,
        )

    return jsonify(delegation)


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

                # 専門エージェントにも同じ質問を委譲する
                try:
                    delegation = agent_delegator.delegate_help_request(
                        question,
                        supplemental_context=problem or None,
                    )
                    response["agent_delegation"] = delegation
                except AgentDispatchError as exc:
                    app.logger.warning("Agent delegation failed: %s", exc)
                    response["agent_delegation_error"] = str(exc)
                    if exc.details:
                        response["agent_delegation_details"] = exc.details
                except Exception:
                    app.logger.exception("Unexpected error during agent delegation:")
                    response["agent_delegation_error"] = "外部エージェントへの委譲に失敗しました。"
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


if __name__ == "__main__":
    # インデックス読み込みは ai_engine 側で一度だけ行われる
    app.run(host="0.0.0.0", port=5000, debug=True)
