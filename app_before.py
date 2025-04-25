# app.py
from flask import Flask, request, jsonify, render_template
import os
import json
import logging
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込み
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ログレベル設定（デバッグ時はDEBUGにしておくと詳細が出ます）
logging.basicConfig(level=logging.DEBUG)

# --- llama_index 関連 ---
# コアコンポーネントのインポート
from llama_index.core import (
    Document, PromptHelper, ComposableGraph,
    VectorStoreIndex, load_index_from_storage, StorageContext
)
# Settings は内部モジュールからのインポート（最新版 0.12.30 ではこちらのパスになります）
from llama_index.core.settings import Settings

# LLM として GoogleGenerativeAI を使用（例: gemini-2.0-flash）
from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# プロンプト設定
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
prompt_helper = PromptHelper(4096, CHUNK_SIZE, CHUNK_OVERLAP / CHUNK_SIZE)

# 埋め込みモデルとして HuggingFaceEmbedding を使用
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")

# グローバル設定（Settings は全モジュールに影響します）
Settings.llm = llm
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper

# 定数：インデックスのディレクトリと会話履歴ファイル
INDEX_DB_DIR = "./static/vector_db_llamaindex"
HISTORY_FILE = "conversation_history.json"

def load_all_indices():
    if not os.path.exists(INDEX_DB_DIR):
        raise RuntimeError(f"Directory not found: {INDEX_DB_DIR}")
    subdirs = [d for d in os.listdir(INDEX_DB_DIR) if os.path.isdir(os.path.join(INDEX_DB_DIR, d))]
    if not subdirs:
        raise RuntimeError(f"No index subdirectories found in {INDEX_DB_DIR}")
    
    indices = []
    index_summaries = []
    for subdir in subdirs:
        persist_dir = os.path.join(INDEX_DB_DIR, subdir, "persist")
        if not os.path.exists(persist_dir):
            app.logger.warning(f"Persist directory not found in {subdir}, skipping...")
            continue
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            # Settings により llm 等はグローバル設定済みなので引数は不要
            idx = load_index_from_storage(storage_context)
            indices.append(idx)
            index_summaries.append(f"ファイル: {subdir}")
        except Exception as e:
            app.logger.exception(f"Failed to load index from {persist_dir}: {str(e)}")
    if not indices:
        raise RuntimeError("Failed to load any index.")
    return indices, index_summaries

def create_composable_graph():
    #indices, index_summaries = load_all_indices()
    indices, index_summaries = load_all_indices()
    # ── ここで「上位7件」のみを抽出 ──
    indices = indices[:7]
    index_summaries = index_summaries[:7]

    if len(indices) == 1:
        return indices[0]
    # グローバル設定済みのため、追加の引数は不要
    graph = ComposableGraph.from_indices(VectorStoreIndex, indices, index_summaries=index_summaries)
    return graph

try:
    graph_or_index = create_composable_graph()
except Exception as e:
    app.logger.exception("Indexの初期化に失敗しました。")
    graph_or_index = None  # 後続のリクエストではエラーを返す仕組みにしています

# --- 会話履歴管理 ---
def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("conversation_history", [])
        except Exception as e:
            app.logger.exception(f"履歴の読み込みエラー: {str(e)}")
            return []
    else:
        return []

def save_conversation_history(conversation_history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversation_history": conversation_history}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        app.logger.exception(f"履歴の保存エラー: {str(e)}")

# カスタム結合プロンプト（変更せずフロントエンドと連携）
COMBINE_PROMPT = """あなたは、資料を基にユーザーの問いに対してサポートするためのアシスタントです。

以下の回答候補を統合して、最終的な回答を作成してください。  
【回答候補】  
{summaries}

【統合ルール】  
- もし用意されたドキュメント内の情報が十分でない場合には、情報不足であることを明示し、その上であなたの知識で回答を生成してください。  
- 可能な限り、既に行われた会話内容からも補足情報を取り入れて、有用な回答を提供してください。  
- 各候補の根拠（参照ファイル情報）がある場合、その情報を保持してください。  
- 重複する参照は１つにまとめてください。 
- 回答が十分な情報を含むよう、可能な範囲で詳細に記述してください。  
- 重要！　必ず日本語で回答すること！

【回答例】
    【資料に答えがある場合】
    (質問例-スイッチが入らない時にはどうすればいい？)
    - そうですね、まずは電源ケーブルがしっかりと接続されているか確認してください。
        次に、バッテリーが充電されているか確認してください。
        もしそれでもスイッチが入らない場合は、取扱説明書のトラブルシューティングのページを参照するか、カスタマーサポートにご連絡ください。

    【資料に答えがない場合】
    (質問例-この製品の最新のファームウェアのリリース日はいつですか？)
    - 最新のファームウェアのリリース日については、現在用意されている資料には記載がありません。
        しかし、一般的には、製品のウェブサイトのサポートセクションや、メーカーからのメールマガジンなどで告知されることが多いです。
        そちらをご確認いただくか、直接メーカーにお問い合わせいただくことをお勧めします。
"""

@app.route("/rag_answer", methods=["POST"])
def rag_answer():
    if graph_or_index is None:
        return jsonify({"error": "インデックスが初期化されていません。サーバーログを確認してください。"}), 500

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "質問を入力してください"}), 400

    conversation_history = load_conversation_history()
    conversation_history.append({"role": "User", "message": question})
    query_with_history = "\n".join([f"{entry['role']}: {entry['message']}" for entry in conversation_history])
    
    try:
        # as_query_engine() の段階で prompt_template を設定（変更済み）
        if hasattr(graph_or_index, "query"):
            query_engine = graph_or_index.as_query_engine(prompt_template=COMBINE_PROMPT)
        else:
            query_engine = graph_or_index.as_query_engine(prompt_template=COMBINE_PROMPT)
        
        # include_source を指定せずに query() を呼び出す
        response = query_engine.query(query_with_history)
        answer = response.response

        # ソース情報が response.source_nodes に入っている場合（バージョンに応じて確認）-
        source_nodes = getattr(response, "source_nodes", [])
        ref_dict = {}
        for node in source_nodes:
            # extra_info と metadata の両方からファイル名とページ番号を取得する
            meta = getattr(node, "extra_info", {}) or {}
            source = meta.get("source")
            page = meta.get("page")
            # extra_info に情報がない場合、metadata から取得する
            if not source and hasattr(node, "metadata"):
                source = node.metadata.get("source")
            if not page and hasattr(node, "metadata"):
                page = node.metadata.get("page")
            
            # ファイル名が取得できない、または「不明ファイル」の場合はスキップする
            if not source or source == "不明ファイル":
                continue
                
            # ページ番号が取得できなければデフォルト値を設定
            page = page or "不明"
            
            # 重複しないように、ファイルごとにセットでページ番号を管理
            ref_dict.setdefault(source, set()).add(str(page))

        # ここで、複数回参照されているファイルのみを残す
        if ref_dict:
            # 例として、2回以上参照されているファイルのみを抽出
            filtered_ref_dict = {src: pages for src, pages in ref_dict.items() if len(pages) >= 1}
            # フィルタ結果が空の場合は、フィルタ前の結果を利用する（任意のフォールバック）
            if filtered_ref_dict:
                ref_dict = filtered_ref_dict

        # 参照表示用の文字列を生成
        if ref_dict:
            references = ", ".join(
                [f"{src} (page: {', '.join(sorted(pages, key=lambda x: int(x) if x.isdigit() else x))})"
                for src, pages in ref_dict.items()]
            )
            if "【参照：" not in answer:
                final_answer = answer + "\n\n【使用したファイル】\n" + references
            else:
                final_answer = answer
        else:
            # 参照対象がなければファイル一覧は表示しない
            final_answer = answer



        conversation_history.append({"role": "AI", "message": final_answer})
        save_conversation_history(conversation_history)

        return jsonify({
            "answer": final_answer,
            "sources": list(ref_dict.keys())
        })
    except Exception as e:
        app.logger.exception("Error during query processing:")
        return jsonify({"error": str(e)}), 500




@app.route("/reset_history", methods=["POST"])
def reset_history():
    try:
        save_conversation_history([])
        return jsonify({"status": "Conversation history reset."})
    except Exception as e:
        app.logger.exception("Error during history reset:")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
