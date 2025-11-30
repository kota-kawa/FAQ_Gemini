# Repository Guidelines

## Architecture & Module Ownership
- `app.py` is the single Flask entrypoint. It wires CORS, loads environment secrets, and exposes the HTTP APIs consumed by the UI and other agents:
  - `/rag_answer` stores the exchanged turns and powers the user chat.
  - `/agent_rag_answer` serves peer agents without touching the persisted history.
  - `/reset_history`, `/conversation_history`, and `/conversation_summary` manage/chat through `conversation_history.json`.
  - `/analyze_conversation` inspects an external transcript and, when家庭科系の知識で解決できると判断した場合, turns it into a concrete VDB query plus support message.
  - `/` renders `templates/index.html`, whose inline JS handles the summary accordion, optimistic chat updates, loading spinners, and fetches listed above.
- Retrieval and reasoning live in `ai_engine_faiss.py`. It loads every `home-topic-vdb/*/persist` FAISS store with `HuggingFaceEmbeddings` (`intfloat/multilingual-e5-large` on `EMBEDDING_DEVICE`), queries with `vector_retriever`, and calls `ChatOpenAI(model="gemini-2.5-flash")` through the Gemini-compatible OpenAI shim.
  - `conversation_history.json` stores alternating `User`/`AI` turns; `reset_history()` clears it and the summary endpoint reuses the same file.
  - `get_conversation_summary()` and `analyze_external_conversation()` each run bespoke prompts so tweaks there must respect the JSON contracts in `app.py`.
  - `ai_engine.py` is the legacy LlamaIndex variant backed by `constitution_vector_db/`; keep it intact unless you intentionally revive the old graph flow.
- Front-end assets live under `templates/` and `static/` (we currently only ship inline styles/JS). The UI expects JSON responses exactly as implemented in `app.py`; breaking shapes will surface immediately in fetch handlers.
- Data ingestion utilities (`docx_to_vector.py`, `csv_to_vector.py`, `jsonl_to_vector_faiss.py`, `json_to_vector.py`, `pdf_to_vector.py`, `txt_to_vector.py`, `xml_to_vector.py`, etc.) are the only scripts that should emit FAISS/LlamaIndex artifacts. Persist everything under `home-topic-vdb/` (FAISS) or `constitution_vector_db/` (legacy graph). The `docx_to_qa/` helpers convert raw materials into JSONL Q&A pairs before vectorizing.
- Docker tooling (`Dockerfile`, `docker-compose.yml`) mirrors the local workflow: the `qasystem` service runs `flask run` with reload and attaches to the external `${MULTI_AGENT_NETWORK}` for agent-to-agent calls.

## Configuration & Secrets
- Put secrets in `secrets.env` (see `env.txt` for placeholders). `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` must be defined; whichever exists first bootstraps both Gemini and LangChain's OpenAI client. Optional overrides: `OPENAI_BASE_URL`/`OPENAI_API_BASE`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`.
- Vector jobs read from the same `secrets.env`, so ensure the keys are exported before running any `*_to_vector*.py`.

## Build & Run Commands
- Create a virtual environment with `python -m venv .venv && source .venv/bin/activate` (or reuse the checked-in `venv/` when allowed) and install dependencies via `pip install -r requirements.txt`.
- Local dev server: `FLASK_ENV=development python app.py` (or `flask run` equivalent). The Docker stack can be launched with `docker compose up --build qasystem`.
- Rebuild FAISS stores whenever the source corpus changes: run the relevant ingestion script (for example `python docx_to_vector.py` or `python jsonl_to_vector_faiss.py`) and verify it writes only under the sanctioned vector folders.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; prefer explicit imports and descriptive helpers (e.g., `get_conversation_summary`, `analyze_external_conversation`).
- Use snake_case for Python identifiers, kebab-case for CLI entry points, and keep HTML ids/classes consistent with the dash-delimited patterns already present in `templates/index.html`.
- Front-end changes should remain vanilla JS/CSS unless there is a strong reason otherwise; keep animations/lightweight DOM helpers readable and progressively enhanced.
- Run `black` on large edits and lint via `python -m compileall` until additional tooling lands.

## Manual & Automated Testing
- There is no automated suite yet. Add `pytest` cases under `tests/` using the `test_<module>.py` naming pattern when you touch Python core logic.
- Smoke tests to run after app or retrieval changes:
  - `curl -X POST localhost:5000/rag_answer -H 'Content-Type: application/json' -d '{"question":"..."}` should return `{answer,sources}` and append to `conversation_history.json`.
  - `curl -X POST localhost:5000/agent_rag_answer ...` must succeed without modifying history.
  - `curl localhost:5000/conversation_summary` verifies the summarizer output, and `curl -X POST localhost:5000/analyze_conversation` with a crafted transcript should either surface `"needs_help": true` plus `support_message` or cleanly decline.
- Update `evaluation/query_pipeline_stats.json` or any other benchmark artifacts whenever retrieval/ranking behavior changes.

## Commit & Pull Request Guidelines
- Keep commits short, present-tense (English or Japanese), and optionally scoped (`docs:`, `fix:`). Each commit should represent one logical change and note any vector rebuilds or regenerated assets in the body.
- PR descriptions must cover the motivation, manual/automated test steps (`python app.py`, ingestion script invocations, curl checks), and link related tickets or cooperating agents so reviewers understand downstream impact.

## Critical Instruction / 重要指示
どれだけ時間とコストをかけてもよいので、確実に目標が達成されるようにしてほしい。
