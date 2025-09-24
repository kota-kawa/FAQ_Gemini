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
    PromptHelper,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_google_genai import GoogleGenerativeAI

# ── 環境変数 ──
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
logging.basicConfig(level=logging.DEBUG)

# ── チャンク設定（FAQ 例を想定） ──
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

prompt_helper = PromptHelper(
    context_window=4096,                # ← max_input_size の代替
    num_output=CHUNK_SIZE,
    chunk_overlap_ratio=CHUNK_OVERLAP / CHUNK_SIZE,
)

# ── LLM / 埋め込みモデル設定 ──
llm = GoogleGenerativeAI(model="gemini-2.0-flash")
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
#embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-mpnet-base-v2')
#embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/stsb-xlm-r-multilingual')

Settings.llm = llm
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper

# ── インデックス設定 ──
INDEX_DB_DIR = "./static/vector_db_csv"
HISTORY_FILE = "conversation_history.json"

def load_all_indices():
    """./static/vector_db_llamaindex 以下を走査し、
    サブディレクトリごとの VectorStoreIndex をロードして返す"""
    if not os.path.exists(INDEX_DB_DIR):
        raise RuntimeError(f"Directory not found: {INDEX_DB_DIR}")
    subdirs = [
        d for d in os.listdir(INDEX_DB_DIR)
        if os.path.isdir(os.path.join(INDEX_DB_DIR, d))
    ]
    indices, summaries = [], []
    for subdir in subdirs:
        persist = os.path.join(INDEX_DB_DIR, subdir, "persist")
        if not os.path.exists(persist):
            logging.warning(f"Persist directory not found in {subdir}, skipping...")
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
            index_summaries=index_summaries,
        )
except Exception:
    logging.exception("Indexの初期化に失敗しました。")
    graph_or_index = None
    NUM_INDICES = 0

# ── 会話履歴ユーティリティ ──
def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("conversation_history", [])
        except Exception as e:
            logging.exception(f"履歴の読み込みエラー: {e}")
    return []

def save_conversation_history(hist):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversation_history": hist}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.exception(f"履歴の保存エラー: {e}")

# ── プロンプト ──
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

# ── 公開 API ──
def get_answer(question: str):
    """質問文字列を受け取り、RAG 結果（answer, sources）を返す"""
    if graph_or_index is None:
        raise RuntimeError("インデックスが初期化されていません。")
    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    # 会話履歴に保存（ただし検索クエリには使わない）
    history = load_conversation_history()
    history.append({"role": "User", "message": question})

    # ── ここを変更 ──
    # 元: query_text = "\n".join(f"{e['role']}: {e['message']}" for e in history)
    query_text = question
    # ──────────────────

    query_engine = graph_or_index.as_query_engine(
        prompt_template=COMBINE_PROMPT,
        graph_query_kwargs={"top_k": NUM_INDICES},
        child_query_kwargs={
            "similarity_top_k": 5,
            "similarity_threshold": 0.2,
        },
        response_mode="tree_summarize",
    )
    response = query_engine.query(query_text)
    answer = response.response

    # 上位 2 ファイル抽出
    nodes = getattr(response, "source_nodes", [])
    sorted_nodes = sorted(nodes, key=lambda n: getattr(n, "score", 0), reverse=True)
    top_srcs = []
    for n in sorted_nodes:
        meta = getattr(n, "extra_info", {}) or {}
        src = meta.get("source") or n.metadata.get("source")
        if src and src != "不明ファイル" and src not in top_srcs:
            top_srcs.append(src)
        if len(top_srcs) == 3:
            break

    ref_dict = {s: set() for s in top_srcs}
    for n in nodes:
        meta = getattr(n, "extra_info", {}) or {}
        s = meta.get("source") or n.metadata.get("source")
        if s in ref_dict:
            pg = meta.get("page") or n.metadata.get("page") or "不明"
            ref_dict[s].add(str(pg))

    if ref_dict:
        refs = ", ".join(
            f"{s} (page: {', '.join(sorted(pgs))})" for s, pgs in ref_dict.items()
        )
        final = answer + "\n\n【使用したファイル】\n" + refs
    else:
        final = answer

    # 履歴保存
    history.append({"role": "AI", "message": final})
    save_conversation_history(history)
    return final, list(ref_dict.keys())

def reset_history():
    """conversation_history.json を空にする"""
    save_conversation_history([])
