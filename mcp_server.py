"""MCP server exposing Life-Style tools."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from env_loader import load_secrets_env
from lifestyle_tools import analyze_conversation_payload, run_rag_answer

try:
    import mcp.types as types
    from mcp.server import Server
    from mcp.types import Tool
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for clarity
    raise RuntimeError('mcp[cli] is required. Install with `pip install "mcp[cli]"`.') from exc

load_secrets_env()
mcp_server = Server("life-style-agent")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose available Life-Style tools."""

    return [
        Tool(
            name="rag_answer",
            description="生活に関する質問に答えるツール。内部で ai_engine.get_answer を呼び出します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "ユーザーからの質問文"},
                    "persist_history": {
                        "type": "boolean",
                        "description": "True で会話履歴へ保存（デフォルト: true）",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="analyze_conversation",
            description="会話ログからサポートが必要か分析するツール。内部で ai_engine.analyze_external_conversation を呼び出します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_history": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "role と message を含む会話履歴",
                    },
                    "history": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "conversation_history のエイリアス",
                    },
                },
            },
        ),
    ]


def _text_payload(payload: Dict[str, Any]) -> types.TextContent:
    return types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> List[types.TextContent]:
    """Dispatch MCP tool calls to the underlying logic."""

    if name == "rag_answer":
        question = ""
        persist_history = True
        if isinstance(arguments, dict):
            question = str(arguments.get("question") or "").strip()
            if isinstance(arguments.get("persist_history"), bool):
                persist_history = bool(arguments.get("persist_history"))

        if not question:
            return [_text_payload({"error": "question is required"})]

        try:
            result = run_rag_answer(question, persist_history=persist_history)
            return [_text_payload(result)]
        except Exception as exc:  # noqa: BLE001
            logging.exception("MCP rag_answer failed: %s", exc)
            return [_text_payload({"error": str(exc)})]

    if name == "analyze_conversation":
        history = []
        if isinstance(arguments, dict):
            history = arguments.get("conversation_history") or arguments.get("history") or []
        try:
            result = analyze_conversation_payload(history)
            return [_text_payload(result)]
        except Exception as exc:  # noqa: BLE001
            logging.exception("MCP analyze_conversation failed: %s", exc)
            return [_text_payload({"error": str(exc)})]

    raise ValueError(f"Tool not found: {name}")


if __name__ == "__main__":
    import anyio
    import mcp.server.stdio

    async def _main() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                initialization_options=mcp_server.create_initialization_options(),
            )

    anyio.run(_main)
