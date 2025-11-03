# Repository Guidelines

## Project Structure & Module Organization
- `app.py` exposes the Flask entrypoint and HTTP routes consumed by the front end in `templates/index.html` and assets under `static/`.
- Retrieval logic lives in `ai_engine_faiss.py`, which loads FAISS indices from `home-topic-vdb/*/persist` and manages conversation state in `conversation_history.json`.
- Data ingestion scripts (`docx_to_vector.py`, `csv_to_vector.py`, `jsonl_to_vector_faiss.py`, etc.) generate vector stores; keep generated artifacts inside the pre-created `constitution_vector_db/` and `home-topic-vdb/` folders.
- Environment secrets belong in `.env` (sample values in `env.txt`); Docker assets sit alongside `docker-compose.yml` and `Dockerfile`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated interpreter (skip if you use the checked-in `venv/`).
- `pip install -r requirements.txt` installs the Flask app, LangChain integrations, and embedding models.
- `FLASK_ENV=development python app.py` launches the API locally with hot reload; production parity is available via `docker compose up --build qasystem`.
- `python docx_to_vector.py` (or the other `*_to_vector.py` scripts) refreshes vector indexes after adding source documents; keep API keys set when running these jobs.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; favor explicit imports and descriptive function names (`get_answer`, `load_conversation_history`).
- Prefer snake_case for Python symbols and kebab-case for new CLI entrypoints; HTML ids/classes should mirror existing dash-delimited patterns.
- Format heavy edits with `black` and check lint manually (`python -m compileall`) until formal tooling is added.

## Testing Guidelines
- No automated suite exists yet; when contributing Python modules, add `pytest` cases under a `tests/` directory and keep filenames `test_<module>.py`.
- Smoke-test retrieval flows by calling `POST /rag_answer` with `curl` or `httpie`, and verify indices via the vector scripts’ console logs.
- Keep evaluation artifacts (e.g., `evaluation/query_pipeline_stats.json`) updated if your change affects ranking quality.

## Commit & Pull Request Guidelines
- Match the current history: short, present-tense summaries (English or Japanese) with optional scopes (`docs:`, `fix:`) when helpful.
- One logical change per commit; include data regeneration notes in the body if indices were rebuilt.
- PRs should describe motivation, list testing steps (`python app.py`, ingestion scripts, manual API check), and link any tracker tickets or related agents.
