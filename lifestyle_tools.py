"""Shared helpers for Life-Assistant API and MCP tools."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import ai_engine_faiss as ai_engine


def run_rag_answer(question: str, *, persist_history: bool = True) -> Dict[str, Any]:
    """Invoke the RAG pipeline and return the API-shaped payload."""

    answer, sources = ai_engine.get_answer(question, persist_history=persist_history)
    return {"answer": answer, "sources": sources}


def _normalise_conversation_history(raw_history: Any) -> List[Dict[str, str]]:
    """Validate and normalise conversation history entries."""

    if raw_history is None:
        return []
    if not isinstance(raw_history, list):
        raise ValueError("conversation_historyはリスト形式で送信してください")

    cleaned: List[Dict[str, str]] = []
    for entry in raw_history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        message = entry.get("message") or entry.get("content")
        if isinstance(role, str) and isinstance(message, str):
            cleaned.append({"role": role, "message": message})
    return cleaned


def analyze_conversation_payload(raw_history: Any) -> Dict[str, Any]:
    """
    Analyse a conversation log and, if useful, fetch a support message from the VDB.
    Mirrors the HTTP endpoint behaviour so MCP and Flask stay in sync.
    """

    conversation_history = _normalise_conversation_history(raw_history)
    if not conversation_history:
        raise ValueError("会話履歴が空です")

    analysis = ai_engine.analyze_external_conversation(conversation_history)
    response: Dict[str, Any] = {
        "analyzed": True,
        "needs_help": bool(analysis.get("needs_help", False)),
        "should_reply": bool(analysis.get("should_reply", False)),
        "reply": analysis.get("reply", "") or "",
        "addressed_agents": analysis.get("addressed_agents", []) or [],
    }

    if "error" in analysis:
        logging.warning("Analysis error detected: %s", analysis.get("error"))

    if not response["needs_help"]:
        return response

    problem = analysis.get("problem", "")
    question = analysis.get("question", "")

    response["problem"] = problem if isinstance(problem, str) else ""
    question_text = question if isinstance(question, str) else ""

    if not question_text:
        response["support_message"] = "問題は特定されましたが、具体的な質問が生成されませんでした。"
        response["sources"] = []
        if not response["reply"]:
            response["reply"] = response["support_message"]
            response["should_reply"] = True
        return response

    try:
        answer, sources = ai_engine.get_answer(question_text)
        response["support_message"] = answer
        response["sources"] = sources
        if not response["reply"]:
            response["reply"] = answer
            response["should_reply"] = True
    except Exception:  # noqa: BLE001
        logging.exception("Error getting answer from VDB during conversation analysis")
        response["support_message"] = "回答の取得中にエラーが発生しました。"
        response["sources"] = []
        if not response["reply"]:
            response["reply"] = response["support_message"]
            response["should_reply"] = True

    return response
