# flask_app.py
from flask import Flask, request, jsonify, render_template

from flask_cors import CORS

import os
import logging
from dotenv import load_dotenv

#import ai_engine  # AI/RAG ロジックをまとめた別モジュール
import ai_engine_faiss as ai_engine

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
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    """トップページ（テンプレートは従来どおり）"""
    return render_template("index.html")


if __name__ == "__main__":
    # インデックス読み込みは ai_engine 側で一度だけ行われる
    app.run(host="0.0.0.0", port=5000, debug=True)
