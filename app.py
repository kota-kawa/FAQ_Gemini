# flask_app.py
from flask import Flask, request, jsonify, render_template
import os
import logging
from dotenv import load_dotenv

#import ai_engine  # AI/RAG ロジックをまとめた別モジュール
import ai_engine_gemini as ai_engine

# ── 環境変数 / Flask 初期化 ──
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
logging.basicConfig(level=logging.DEBUG)


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


@app.route("/")
def index():
    """トップページ（テンプレートは従来どおり）"""
    return render_template("index.html")


if __name__ == "__main__":
    # インデックス読み込みは ai_engine 側で一度だけ行われる
    app.run(host="0.0.0.0", port=5000, debug=True)
