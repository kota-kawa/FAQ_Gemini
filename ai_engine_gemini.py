import os
import json
import logging
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI

# ── 環境変数 ──
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
logging.basicConfig(level=logging.DEBUG)

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# ── インデックス設定 ──
INDEX_DB_DIR = "./constitution_vector_db"
HISTORY_FILE = "conversation_history.json"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": EMBEDDING_DEVICE},
)


def load_all_indices():
    """./constitution_vector_db 以下を走査し、FAISS のインデックスを読み込む"""
    if not os.path.exists(INDEX_DB_DIR):
        raise RuntimeError(f"Directory not found: {INDEX_DB_DIR}")

    subdirs = [
        d for d in os.listdir(INDEX_DB_DIR)
        if os.path.isdir(os.path.join(INDEX_DB_DIR, d))
    ]

    combined_store: Optional[FAISS] = None
    for subdir in subdirs:
        persist = os.path.join(INDEX_DB_DIR, subdir, "persist")
        index_path = os.path.join(persist, "index.faiss")
        if not os.path.exists(index_path):
            logging.warning(f"FAISS index not found in {subdir}, skipping...")
            continue

        try:
            store = FAISS.load_local(
                persist,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            logging.exception(f"Failed to load FAISS index from {persist}")
            continue

        if combined_store is None:
            combined_store = store
        else:
            combined_store.merge_from(store)

    if combined_store is None:
        raise RuntimeError("Failed to load any FAISS indices.")

    return combined_store


try:
    VECTOR_STORE = load_all_indices()
    vector_retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 5})
except Exception:
    logging.exception("FAISS インデックスの初期化に失敗しました。")
    vector_retriever = None


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
    if vector_retriever is None:
        raise RuntimeError("FAISS インデックスが初期化されていません。")

    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    history = load_conversation_history()
    history.append({"role": "User", "message": question})

    retrieved_docs = vector_retriever.get_relevant_documents(question)

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
