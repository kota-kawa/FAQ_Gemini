# flask_app.py
import asyncio
import json
import logging
import os
import queue
import threading
import uuid

import anyio
import requests
from env_loader import load_secrets_env
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from mcp.types import JSONRPCMessage

import ai_engine_faiss as ai_engine
from lifestyle_tools import analyze_conversation_payload, run_rag_answer
from mcp_server import mcp_server
from model_selection import current_selection, update_override
import evaluation_manager

# ── 環境変数 / Flask 初期化 ──
load_secrets_env()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
logging.basicConfig(level=logging.DEBUG)
_PLATFORM_BASE = os.getenv("MULTI_AGENT_PLATFORM_BASE", "http://web:5050").rstrip("/")
_MCP_SESSIONS: dict[str, queue.Queue] = {}


@app.after_request
def add_cors_headers(response):
    """Allow all domains to access the API without altering existing logic."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    request_headers = request.headers.get("Access-Control-Request-Headers")
    if request_headers:
        response.headers["Access-Control-Allow-Headers"] = request_headers
    else:
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    return response


def _notify_platform(selection: dict) -> None:
    """Best-effort push of the lifestyle model selection back to the platform."""

    if not _PLATFORM_BASE or not isinstance(selection, dict):
        return

    try:
        url = f"{_PLATFORM_BASE}/api/model_settings"
        payload = {"selection": {"lifestyle": selection}}
        headers = {"X-Agent-Origin": "lifestyle"}
        res = requests.post(url, json=payload, headers=headers, timeout=2.0)
        if not res.ok:
            app.logger.info("Platform model sync skipped (%s %s)", res.status_code, res.text)
    except requests.exceptions.RequestException as exc:
        app.logger.info("Platform model sync skipped (%s)", exc)


# ── Evaluation Routes ──
@app.route("/evaluation")
def evaluation_page():
    return render_template("evaluation.html")

@app.route("/api/evaluation/data", methods=["GET"])
def get_eval_data():
    return jsonify({
        "tasks": evaluation_manager.get_tasks(),
        "results": evaluation_manager.get_results()
    })

@app.route("/api/evaluation/tasks", methods=["POST", "DELETE"])
def manage_eval_tasks():
    if request.method == "DELETE":
        evaluation_manager.clear_data()
        return jsonify({"status": "cleared"})
    
    data = request.get_json() or {}
    action = data.get("action")
    if action == "add_samples":
        evaluation_manager.add_tasks(evaluation_manager.SAMPLE_TASKS)
        return jsonify({"status": "added"})
    
    return jsonify({"error": "Invalid action"}), 400

@app.route("/api/evaluation/run", methods=["POST"])
def run_eval_task():
    data = request.get_json() or {}
    task_id = data.get("task_id")
    if not task_id:
        return jsonify({"error": "task_id required"}), 400
    
    tasks = evaluation_manager.get_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    # Run RAG
    try:
        # Get current model info
        selection = current_selection("lifestyle")
        model_name = selection.get("model", "unknown")
        
        # Execute RAG (no persistence)
        rag_result = run_rag_answer(task["question"], persist_history=False)
        actual_answer = rag_result.get("answer", "")
        
        # Record result
        result = evaluation_manager.add_result(
            task_id=task_id,
            model=model_name,
            question=task["question"],
            expected_answer=task["expected_answer"],
            actual_answer=actual_answer
        )
        return jsonify({"result": result})
    except Exception as e:
        app.logger.exception("Evaluation run failed")
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluation/status", methods=["POST"])
def update_eval_status():
    data = request.get_json() or {}
    result_id = data.get("result_id")
    status = data.get("status")
    if not result_id or not status:
        return jsonify({"error": "Missing result_id or status"}), 400
    
    evaluation_manager.update_result_status(result_id, status)
    return jsonify({"status": "updated"})


@app.route("/rag_answer", methods=["POST"])
def rag_answer():
    """POST: { "question": "..." } → RAG で回答"""
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "質問を入力してください"}), 400
    try:
        result = run_rag_answer(question)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error during query processing:")
        return jsonify({"error": str(e)}), 500


@app.route("/agent_rag_answer", methods=["POST"])
def agent_rag_answer():
    """他エージェントからの問い合わせに応答するが、会話履歴には保存しない"""
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "質問を入力してください"}), 400
    try:
        result = run_rag_answer(question, persist_history=False)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error during external agent query processing:")
        return jsonify({"error": str(e)}), 500


@app.route("/reset_history", methods=["POST"])
def reset_history():
    """会話履歴をリセット"""
    ai_engine.reset_history()
    return jsonify({"status": "Conversation history reset."})


@app.route("/conversation_history", methods=["GET"])
def conversation_history():
    history = ai_engine.get_conversation_history()
    return jsonify({"conversation_history": history})


@app.route("/conversation_summary", methods=["GET"])
def conversation_summary():
    try:
        summary = ai_engine.get_conversation_summary()
        return jsonify({"summary": summary})
    except Exception as e:
        app.logger.exception("Error during conversation summarization:")
        return jsonify({"error": "会話の要約中にエラーが発生しました"}), 500


@app.route("/model_settings", methods=["GET", "POST"])
def update_model_settings():
    """Update or expose the active LLM model without restarting the service."""

    if request.method == "GET":
        selection = current_selection("lifestyle")
        return jsonify({"selection": {
            "provider": selection.get("provider"),
            "model": selection.get("model"),
            "base_url": selection.get("base_url"),
        }})

    data = request.get_json(silent=True) or {}
    selection = data if isinstance(data, dict) else {}
    try:
        update_override(selection if selection else None)
        ai_engine.refresh_llm(selection if selection else None)
        if request.headers.get("X-Platform-Propagation") != "1" and selection:
            _notify_platform(selection)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Failed to refresh model settings: %s", exc)
        return jsonify({"error": "モデル設定の更新に失敗しました。"}), 500
    return jsonify({"status": "ok", "applied": selection or "from_file"})


# --- MCP Server Bridge ---
@app.route("/mcp/sse")
def mcp_sse_endpoint():
    session_id = str(uuid.uuid4())
    input_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()
    _MCP_SESSIONS[session_id] = input_queue

    def run_server_loop():
        async def run():
            read_stream_send, read_stream_recv = anyio.create_memory_object_stream(10)
            write_stream_send, write_stream_recv = anyio.create_memory_object_stream(10)

            async def feed_input():
                while True:
                    try:
                        msg = await asyncio.to_thread(input_queue.get)
                        if msg is None:
                            break
                        try:
                            parsed = JSONRPCMessage.model_validate(msg)
                            await read_stream_send.send(parsed)
                        except Exception as exc:  # noqa: BLE001
                            app.logger.warning("MCP Parse Error: %s", exc)
                    except Exception:  # noqa: BLE001 - defensive loop guard
                        break
                await read_stream_send.aclose()

            async def consume_output():
                async with write_stream_recv:
                    async for msg in write_stream_recv:
                        try:
                            if hasattr(msg, "model_dump_json"):
                                data = msg.model_dump_json()
                            else:
                                data = json.dumps(msg)
                        except Exception as exc:  # noqa: BLE001
                            app.logger.warning("MCP serialization error: %s", exc)
                            continue
                        output_queue.put(f"event: message\ndata: {data}\n\n")
                output_queue.put(None)

            async with anyio.create_task_group() as tg:
                tg.start_soon(feed_input)
                tg.start_soon(consume_output)
                output_queue.put(f"event: endpoint\ndata: /mcp/messages?session_id={session_id}\n\n")

                try:
                    await mcp_server.run(
                        read_stream_recv,
                        write_stream_send,
                        initialization_options=mcp_server.create_initialization_options(),
                    )
                except Exception as exc:  # noqa: BLE001
                    app.logger.exception("MCP Run Error: %s", exc)

        try:
            asyncio.run(run())
        finally:
            _MCP_SESSIONS.pop(session_id, None)

    threading.Thread(target=run_server_loop, daemon=True).start()

    def generate():
        while True:
            try:
                msg = output_queue.get(timeout=25)
                if msg is None:
                    break
                yield msg
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/mcp/messages", methods=["POST"])
def mcp_messages_endpoint():
    session_id = request.args.get("session_id")
    if not session_id or session_id not in _MCP_SESSIONS:
        return jsonify({"error": "Session not found"}), 404

    _MCP_SESSIONS[session_id].put(request.json)
    return jsonify({"status": "accepted"}), 202


@app.route("/analyze_conversation", methods=["POST"])
def analyze_conversation():
    """
    外部エージェントから会話履歴を受け取り、VDBの知識で解決できる問題があれば支援メッセージを返す。
    
    リクエスト形式:
    {
        "conversation_history": [
            {"role": "User", "message": "..."},
            {"role": "AI", "message": "..."}
        ]
    }
    
    レスポンス形式:
    {
        "analyzed": true,
        "needs_help": true/false,
        "problem": "特定された問題" (needs_helpがtrueの場合),
        "support_message": "支援メッセージ" (needs_helpがtrueの場合),
        "sources": [...] (needs_helpがtrueの場合)
    }
    """
    try:
        data = request.get_json()
    except Exception as e:
        app.logger.exception("Error parsing JSON:")
        return jsonify({"error": "無効なJSON形式です"}), 400
    
    if data is None:
        return jsonify({"error": "リクエストボディが必要です"}), 400
    
    conversation_history = data.get("history") or data.get("conversation_history") or []
    
    if not conversation_history:
        return jsonify({"error": "会話履歴が空です"}), 400
    
    try:
        response = analyze_conversation_payload(conversation_history)
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Error during conversation analysis:")
        return jsonify({"error": "会話の分析中にエラーが発生しました"}), 500


@app.route("/")
def index():
    """トップページ（テンプレートは従来どおり）"""
    return render_template("index.html")


if __name__ == "__main__":
    # インデックス読み込みは ai_engine 側で一度だけ行われる
    app.run(host="0.0.0.0", port=5000, debug=True)
