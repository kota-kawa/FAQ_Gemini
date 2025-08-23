# ai_engine_cot.py
import os
import json
import logging
from typing import List, Dict, Any, Tuple

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
    context_window=4096,
    num_output=CHUNK_SIZE,
    chunk_overlap_ratio=CHUNK_OVERLAP / CHUNK_SIZE,
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

# =================================================
#  Chain-of-Thought 生成関数
# =================================================
def _generate_chain_of_thought(question: str) -> str:
    """
    与えられた質問に対して “思考の連鎖 (CoT)” を生成して返す。
    失敗した場合は空文字を返す。
    CoT は検索精度を高めるための内部データとしてのみ利用する。
    """
    cot_prompt = (
        "あなたは熟練したアシスタントです。以下のユーザー質問に対して、"
        "ステップバイステップで詳細に思考を列挙し（各行を「- 」で始める）、"
        "最後に一行で「結論:」から始まる形で結論のみをまとめてください。\n\n"
        f"【ユーザーの質問】\n{question}"
    )
    try:
        response = llm.invoke(cot_prompt)
        content = getattr(response, "content", None) or str(response)
        return content.strip()
    except Exception as e:
        logging.warning(f"Chain-of-thought generation failed: {e}")
        return ""

# =================================================
#  インデックス読み込み
# =================================================
def load_all_indices() -> Tuple[List[VectorStoreIndex], List[str]]:
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
    logging.exception("Index の初期化に失敗しました。")
    graph_or_index = None
    NUM_INDICES = 0

# =================================================
#  会話履歴ユーティリティ
# =================================================
def load_conversation_history() -> List[Dict[str, str]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("conversation_history", [])
        except Exception as e:
            logging.exception(f"履歴の読み込みエラー: {e}")
    return []

def save_conversation_history(hist: List[Dict[str, str]]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversation_history": hist}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.exception(f"履歴の保存エラー: {e}")

# =================================================
#  回答合成プロンプト
# =================================================
COMBINE_PROMPT = """あなたは、資料を基にユーザーの問いに対してサポートするためのアシスタントです。

以下の回答候補を統合して、思考過程を踏まえつつ最終的な回答を生成してください。  
【回答候補】  
{summaries}

【統合ルール】  
1. まず “--- chain of thought ---” 行から始め、資料や過去の会話を参照しながらステップバイステップで推論をまとめてください。  
2. 続いて “--- final answer ---” 行を書き、ユーザーに提示すべき最終回答を簡潔に記述してください。  
3. もし用意されたドキュメント内の情報が十分でない場合には、その旨を明示したうえで、あなたの知識を補完して回答してください。  
4. 参照ファイル情報が重複する場合は 1 つにまとめてください。  
5. 回答は必ず日本語で行ってください。
"""

# =================================================
#  公開 API
# =================================================
def get_answer(question: str) -> Tuple[str, List[str]]:
    """質問文字列を受け取り、RAG 結果（answer, sources）を返す"""
    if graph_or_index is None:
        raise RuntimeError("インデックスが初期化されていません。")
    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    # --- Chain of Thought を内部生成 ---
    cot = _generate_chain_of_thought(question)
    if cot:
        logging.debug(f"[CoT] {cot}")

    # --- 会話履歴の取り込み（保存用） ---
    history = load_conversation_history()
    history.append({"role": "User", "message": question})

    # --- 検索クエリ生成（最新のユーザー入力のみ） ---
    query_text_parts = [
        f"User: {question}",
        f"Assistant_internalthoughts: {cot}" if cot else "",
    ]
    query_text = "\n".join(p for p in query_text_parts if p)

    # --- Query Engine 実行 ---
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
    answer_text = response.response

    # --- 参照ファイル整理（上位 2 件） ---
    nodes = getattr(response, "source_nodes", [])
    sorted_nodes = sorted(nodes, key=lambda n: getattr(n, "score", 0), reverse=True)
    top_srcs: List[str] = []
    for n in sorted_nodes:
        meta = getattr(n, "extra_info", {}) or {}
        src = meta.get("source") or n.metadata.get("source")
        if src and src != "不明ファイル" and src not in top_srcs:
            top_srcs.append(src)
        if len(top_srcs) == 2:
            break

    ref_dict: Dict[str, set[str]] = {s: set() for s in top_srcs}
    for n in nodes:
        meta = getattr(n, "extra_info", {}) or {}
        s = meta.get("source") or n.metadata.get("source")
        if s in ref_dict:
            pg = meta.get("page") or n.metadata.get("page") or "不明"
            ref_dict[s].add(str(pg))

    # --- 参照付き回答整形（ユーザー向け） ---
    if ref_dict:
        refs = ", ".join(
            f"{s} (page: {', '.join(sorted(pgs))})" for s, pgs in ref_dict.items()
        )
        final_answer = answer_text + "\n\n【使用したファイル】\n" + refs
    else:
        final_answer = answer_text

    # --- 履歴保存（CoT は保存しない） ---
    history.append({"role": "AI", "message": final_answer})
    save_conversation_history(history)

    return final_answer, list(ref_dict.keys())

def reset_history() -> None:
    """conversation_history.json を空にする"""
    save_conversation_history([])