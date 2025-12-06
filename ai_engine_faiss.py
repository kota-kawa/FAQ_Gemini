import os
import json
import logging
from datetime import datetime
from typing import List, Optional
from env_loader import load_secrets_env

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from model_selection import apply_model_selection, update_override

# ── 環境変数 ──
load_secrets_env()
logging.basicConfig(level=logging.DEBUG)


def refresh_llm(selection_override: dict | None = None):
    """Refresh the shared LLM instance based on selection."""
    global llm
    if selection_override:
        provider, model_name, _ = update_override(selection_override)
    else:
        provider, model_name, _ = apply_model_selection("lifestyle")

    provider = provider.lower()
    logging.info(f"Refreshing LLM: provider={provider}, model={model_name}")

    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True)
    elif provider == "claude":
        llm = ChatAnthropic(model_name=model_name)
    elif provider == "groq":
        llm = ChatGroq(model_name=model_name)
    else:  # openai
        llm = ChatOpenAI(model=model_name)


refresh_llm()

# ── インデックス設定 ──
INDEX_DB_DIR = "./home-topic-vdb"
HISTORY_FILE = "conversation_history.json"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": EMBEDDING_DEVICE},
)


def _current_datetime_line() -> str:
    """Return the timestamp string embedded in prompts."""
    return datetime.now().strftime("現在の日時ー%Y年%m月%d日%H時%M分")


def load_all_indices():
    """./home-topic-vdb 以下を走査し、FAISS のインデックスを読み込む"""
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


def get_conversation_history():
    """会話履歴を外部に提供するためのラッパー"""
    return load_conversation_history()


# ── プロンプト ──
COMBINE_PROMPT = """現在の日時ー{current_datetime}

あなたは、資料を基にユーザーの問いに対してサポートするためのアシスタントです。

【会話履歴】
{history}

【質問】
{question}

【回答候補】
{summaries}

【統合ルール】
- 出力はユーザーに見やすく分かりやすく、簡潔に綺麗にしてください。
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


def get_answer(question: str, persist_history: bool = True):
    """質問文字列を受け取り、RAG 結果（answer, sources）を返す"""
    if vector_retriever is None:
        raise RuntimeError("FAISS インデックスが初期化されていません。")

    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    history = load_conversation_history()

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
        current_datetime=_current_datetime_line(),
        history=_format_history_for_prompt(history),
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

    if persist_history:
        updated_history = history + [
            {"role": "User", "message": question},
            {"role": "AI", "message": final},
        ]
        save_conversation_history(updated_history)

    return final, list(ref_dict.keys())


def reset_history():
    """conversation_history.json を空にする"""
    save_conversation_history([])


SUMMARY_PROMPT = """あなたは会話を俯瞰し、重要なポイントを簡潔にまとめるアシスタントです。

【会話履歴】
{history}

【要約ルール】
- 重要な質問や回答、決定事項、未解決の点などを中心に2〜5行でまとめてください。
- 会話がない場合は、その旨を1行で伝えてください。
- 必ず日本語で出力してください。
"""


def _format_history_for_summary(history: List[dict], limit: int = 20) -> str:
    if not history:
        return "（会話はありません）"
    recent = history[-limit:]
    return "\n".join(f"{entry['role']}: {entry['message']}" for entry in recent)


def summarize_conversation(history: List[dict]) -> str:
    if not history:
        return "まだ会話はありません。"

    formatted_history = _format_history_for_summary(history)
    prompt_text = SUMMARY_PROMPT.format(history=formatted_history)

    summary_response = llm.invoke(prompt_text)
    if isinstance(summary_response, str):
        return summary_response.strip()
    return (
        getattr(summary_response, "content", None)
        or getattr(summary_response, "text", None)
        or str(summary_response)
    ).strip()


def get_conversation_summary() -> str:
    history = load_conversation_history()
    return summarize_conversation(history)


# ── 外部会話履歴の分析 ──
ANALYZE_CONVERSATION_PROMPT = """あなたは「家庭科・生活知識専門」のアシスタントです。

【あなたの専門分野（発言可能な範囲）】
- 料理・食事: レシピ、調理法、食材の保存、献立計画
- 洗濯・クリーニング: 洗濯方法、シミ抜き、衣類の手入れ
- 掃除・整理整頓: 掃除方法、収納、片付け術
- 栄養・健康管理: 栄養バランス、食事制限、健康的な食生活
- 育児・子育て: 離乳食、子どもの食事、生活習慣
- 家計管理: 食費節約、家庭の予算管理

【発言してはいけない場合】
- ブラウザ操作・Web検索に関する話題 → Browser Agentの専門
- IoTデバイス・スマートホーム操作 → IoT Agentの専門
- スケジュール・予定・タスク管理 → Scheduler Agentの専門
- 一般的な雑談、挨拶、天気の話題（家庭科知識が不要な場合）
- 自分の専門知識がなくても他エージェントが対応できる内容

【判断ルール】
1. 会話が自分の専門分野に「直接的に」関係する場合のみ `should_reply: true`
2. 「少し関係があるかも」程度では発言しない
3. 他エージェントへの呼びかけは、自分の専門分野の文脈でのみ行う
4. `needs_help: true` は、VDBの知識で具体的な回答ができる場合のみ

【発言する例】
- 「夕食のレシピを教えて」→ 発言する（料理の専門）
- 「洗濯機の使い方がわからない」→ 発言する（洗濯の専門）
- 「子どもの離乳食について」→ 発言する（育児・栄養の専門）

【発言しない例】
- 「今日の天気を調べて」→ 発言しない（Browser Agentの専門）
- 「エアコンをつけて」→ 発言しない（IoT Agentの専門）
- 「明日の予定を確認して」→ 発言しない（Scheduler Agentの専門）
- 「こんにちは」→ 発言しない（専門知識不要）

【会話履歴】
{conversation_history}

【出力形式（必ずJSONのみ）】
{{
  "should_reply": true/false,
  "reply": "発言内容（専門分野に限定）",
  "addressed_agents": [],
  "needs_help": true/false,
  "problem": "特定された問題（needs_helpがtrueの場合）",
  "question": "VDBに問い合わせる具体的質問（needs_helpがtrueの場合）"
}}
"""


def analyze_external_conversation(conversation_history: List[dict]) -> dict:
    """
    外部から送られてきた会話履歴を分析し、VDBの知識で解決できる問題があるかを判断する。
    
    Args:
        conversation_history: 会話履歴のリスト [{"role": "User"/"AI", "message": "..."}]
    
    Returns:
        {
            "needs_help": bool,
            "problem": str (optional),
            "question": str (optional)
        }
    """
    if not conversation_history:
        return {
            "should_reply": False,
            "reply": "",
            "addressed_agents": [],
            "needs_help": False,
        }
    
    # 会話履歴の検証とフォーマット
    validated_entries = []
    for entry in conversation_history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        message = entry.get("message") or entry.get("content")
        if not isinstance(role, str) or not isinstance(message, str):
            continue
        validated_entries.append({"role": role, "message": message})
    
    if not validated_entries:
        return {
            "should_reply": False,
            "reply": "",
            "addressed_agents": [],
            "needs_help": False,
        }
    
    formatted_history = "\n".join(
        f"{entry['role']}: {entry['message']}"
        for entry in validated_entries
    )
    
    prompt_text = ANALYZE_CONVERSATION_PROMPT.format(
        conversation_history=formatted_history
    )
    
    try:
        response = llm.invoke(prompt_text)
        if isinstance(response, str):
            response_text = response
        else:
            response_text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
        
        # JSONの抽出（```json ``` で囲まれている可能性がある）
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        if not isinstance(result, dict):
            return {
                "should_reply": False,
                "reply": "",
                "addressed_agents": [],
                "needs_help": False,
            }
        # Ensure required defaults
        result.setdefault("should_reply", False)
        result.setdefault("reply", "")
        result.setdefault("addressed_agents", [])
        result.setdefault("needs_help", False)
        return result
    except json.JSONDecodeError as e:
        logging.error("JSON解析エラー: LLMの出力がJSON形式ではありません")
        return {
            "should_reply": False,
            "reply": "",
            "addressed_agents": [],
            "needs_help": False,
            "error": "JSON解析に失敗しました",
        }
    except Exception as e:
        logging.error("会話分析エラーが発生しました")
        return {
            "should_reply": False,
            "reply": "",
            "addressed_agents": [],
            "needs_help": False,
            "error": "会話分析エラー",
        }
