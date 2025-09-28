import os
import json
import logging
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAI

# ── 環境変数 ──
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
logging.basicConfig(level=logging.DEBUG)

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# ── インデックス設定 ──
INDEX_DB_DIR = "./constitution_vector_db"
HISTORY_FILE = "conversation_history.json"


def load_all_indices():
    """./constitution_vector_db 以下を走査し、BM25 用のコーパスを読み込む"""
    if not os.path.exists(INDEX_DB_DIR):
        raise RuntimeError(f"Directory not found: {INDEX_DB_DIR}")

    subdirs = [
        d for d in os.listdir(INDEX_DB_DIR)
        if os.path.isdir(os.path.join(INDEX_DB_DIR, d))
    ]

    all_documents: List[Document] = []
    for subdir in subdirs:
        persist = os.path.join(INDEX_DB_DIR, subdir, "persist")
        index_path = os.path.join(persist, "bm25_index.json")
        if not os.path.exists(index_path):
            logging.warning(f"BM25 index not found in {subdir}, skipping...")
            continue

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logging.exception(f"Failed to read BM25 index: {index_path}")
            continue

        docs_data = payload.get("documents") if isinstance(payload, dict) else None
        if docs_data is None:
            logging.warning(f"Invalid BM25 index format: {index_path}")
            continue

        for entry in docs_data:
            text = entry.get("text", "")
            if not text:
                continue
            metadata = entry.get("metadata", {}) or {}
            metadata.setdefault("source", metadata.get("source") or f"{subdir}.jsonl")
            all_documents.append(Document(page_content=text, metadata=metadata))

    if not all_documents:
        raise RuntimeError("Failed to load any BM25 corpora.")

    return all_documents


try:
    ALL_DOCUMENTS = load_all_indices()
    bm25_retriever = BM25Retriever.from_documents(ALL_DOCUMENTS, k=5)
except Exception:
    logging.exception("BM25 インデックスの初期化に失敗しました。")
    bm25_retriever = None


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

【会話履歴】
{history}

【質問】
{question}

【回答候補】
{summaries}

【統合ルール】
- もし用意されたドキュメント内の情報が十分でない場合には、情報不足であることを明示し、その上であなたの知識で回答してください。
- 可能な限り、会話履歴にある関連情報も反映させてください。
- 各候補の根拠（参照ファイル情報）がある場合、その情報を保持してください。
- 重複する参照は１つにまとめてください。
- 回答が十分な情報を含むよう、可能な範囲で詳細に記述してください。
- 重要！　必ず日本語で回答すること！
"""


# ── 公開 API ──
def _format_history_for_prompt(history: List[dict]) -> str:
    if not history:
        return "（履歴なし）"
    recent = history[-6:]
    return "\n".join(f"{entry['role']}: {entry['message']}" for entry in recent)


def get_answer(question: str):
    """質問文字列を受け取り、RAG 結果（answer, sources）を返す"""
    if bm25_retriever is None:
        raise RuntimeError("BM25 インデックスが初期化されていません。")

    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    history = load_conversation_history()
    history.append({"role": "User", "message": question})

    retrieved_docs = bm25_retriever.get_relevant_documents(question)

    context_blocks = []
    top_srcs = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "不明ファイル")
        if source not in top_srcs:
            top_srcs.append(source)
        header_parts = [f"候補{idx}: 出典={source}"]
        if doc.metadata.get("line") is not None:
            header_parts.append(f"行={doc.metadata['line']}")
        if doc.metadata.get("chunk_id") is not None:
            header_parts.append(f"チャンク={doc.metadata['chunk_id']}")
        header = " / ".join(header_parts)
        context_blocks.append(f"{header}\n{doc.page_content.strip()}")

    if context_blocks:
        summaries_text = "\n\n".join(context_blocks)
    else:
        summaries_text = "該当資料は見つかりませんでした。資料が不足する場合はその旨を伝えつつ、一般的な知識で補足してください。"

    prompt_text = COMBINE_PROMPT.format(
        history=_format_history_for_prompt(history[:-1]),
        question=question,
        summaries=summaries_text,
    )

    raw_answer = llm.invoke(prompt_text)
    if isinstance(raw_answer, str):
        answer = raw_answer
    else:
        answer = getattr(raw_answer, "content", None) or getattr(raw_answer, "text", None) or str(raw_answer)

    ref_dict = {s: set() for s in top_srcs[:3] if s and s != "不明ファイル"}
    for doc in retrieved_docs:
        src = doc.metadata.get("source")
        if src in ref_dict:
            page = doc.metadata.get("page") or doc.metadata.get("line") or doc.metadata.get("chunk_id")
            if page is not None:
                ref_dict[src].add(str(page))

    if ref_dict:
        refs = ", ".join(
            f"{s} (page: {', '.join(sorted(pgs))})" for s, pgs in ref_dict.items()
        )
        final = answer + "\n\n【使用したファイル】\n" + refs
    else:
        final = answer

    history.append({"role": "AI", "message": final})
    save_conversation_history(history)

    return final, list(ref_dict.keys())


def reset_history():
    """conversation_history.json を空にする"""
    save_conversation_history([])
