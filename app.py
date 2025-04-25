# app.py

from flask import Flask, request, jsonify, render_template
import os
import json
import logging
from dotenv import load_dotenv

# llama_index 関連
from llama_index.core import (
    ComposableGraph,
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
    PromptHelper
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_google_genai import GoogleGenerativeAI

# ── 環境変数 / Flask 初期化 ──
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
logging.basicConfig(level=logging.DEBUG)

# ── チャンク設定（FAQ 例を想定） ──
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

prompt_helper = PromptHelper(
    context_window=4096,                     # ← max_input_size の代替 :contentReference[oaicite:2]{index=2}
    num_output=CHUNK_SIZE,
    chunk_overlap_ratio=CHUNK_OVERLAP / CHUNK_SIZE,
    # chunk_size_limit=CHUNK_SIZE,          # 必要なら設定
)

# ── LLM / 埋め込みモデル設定 ──
llm = GoogleGenerativeAI(model="gemini-2.0-flash")
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
Settings.llm = llm
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper

# ── インデックス設定 ──
INDEX_DB_DIR = "./static/vector_db_llamaindex"
HISTORY_FILE = "conversation_history.json"

def load_all_indices():
    if not os.path.exists(INDEX_DB_DIR):
        raise RuntimeError(f"Directory not found: {INDEX_DB_DIR}")
    subdirs = [d for d in os.listdir(INDEX_DB_DIR)
               if os.path.isdir(os.path.join(INDEX_DB_DIR, d))]
    indices, summaries = [], []
    for subdir in subdirs:
        persist = os.path.join(INDEX_DB_DIR, subdir, "persist")
        if not os.path.exists(persist):
            app.logger.warning(f"Persist directory not found in {subdir}, skipping...")
            continue
        ctx = StorageContext.from_defaults(persist_dir=persist)
        idx = load_index_from_storage(ctx)
        indices.append(idx)
        summaries.append(f"ファイル: {subdir}")
    if not indices:
        raise RuntimeError("Failed to load any index.")
    return indices, summaries

# 一度だけロード
try:
    indices, index_summaries = load_all_indices()
    NUM_INDICES = len(indices)
    if NUM_INDICES == 1:
        graph_or_index = indices[0]
    else:
        graph_or_index = ComposableGraph.from_indices(
            VectorStoreIndex,
            indices,
            index_summaries=index_summaries
        )
except Exception:
    app.logger.exception("Indexの初期化に失敗しました。")
    graph_or_index = None
    NUM_INDICES = 0

def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("conversation_history", [])
        except Exception as e:
            app.logger.exception(f"履歴の読み込みエラー: {e}")
    return []

def save_conversation_history(hist):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversation_history": hist}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        app.logger.exception(f"履歴の保存エラー: {e}")

COMBINE_PROMPT = """あなたは、資料を基にユーザーの問いに対してサポートするためのアシスタントです。
…（省略／必要に応じて追記）…"""

@app.route("/rag_answer", methods=["POST"])
def rag_answer():
    if graph_or_index is None:
        return jsonify({"error": "インデックスが初期化されていません。"}), 500

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "質問を入力してください"}), 400

    history = load_conversation_history()
    history.append({"role": "User", "message": question})
    query_text = "\n".join(f"{e['role']}: {e['message']}" for e in history)

    try:
        query_engine = graph_or_index.as_query_engine(
            prompt_template=COMBINE_PROMPT,
            graph_query_kwargs={"top_k": NUM_INDICES},
            child_query_kwargs={
                "similarity_top_k": 5,
                "similarity_threshold": 0.2
            },
            response_mode="tree_summarize",
            # response_mode="simple_summarize",
            # response_mode="rank_based",
        )
        response = query_engine.query(query_text)
        answer = response.response

        # 上位2ファイル抽出
        nodes = getattr(response, "source_nodes", [])
        sorted_nodes = sorted(nodes, key=lambda n: getattr(n, "score", 0), reverse=True)
        top_srcs = []
        for n in sorted_nodes:
            meta = getattr(n, "extra_info", {}) or {}
            src = meta.get("source") or n.metadata.get("source")
            if src and src != "不明ファイル" and src not in top_srcs:
                top_srcs.append(src)
            if len(top_srcs) == 2:
                break

        ref_dict = {s: set() for s in top_srcs}
        for n in nodes:
            meta = getattr(n, "extra_info", {}) or {}
            s = meta.get("source") or n.metadata.get("source")
            if s in ref_dict:
                pg = meta.get("page") or n.metadata.get("page") or "不明"
                ref_dict[s].add(str(pg))

        if ref_dict:
            refs = ", ".join(f"{s} (page: {', '.join(sorted(pgs))})" for s, pgs in ref_dict.items())
            final = answer + "\n\n【使用したファイル】\n" + refs
        else:
            final = answer

        history.append({"role": "AI", "message": final})
        save_conversation_history(history)
        return jsonify({"answer": final, "sources": list(ref_dict.keys())})
    except Exception as e:
        app.logger.exception("Error during query processing:")
        return jsonify({"error": str(e)}), 500

@app.route("/reset_history", methods=["POST"])
def reset_history():
    save_conversation_history([])
    return jsonify({"status": "Conversation history reset."})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
