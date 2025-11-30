# Life Assistant Agent - Gemini Context

## Project Overview
The **Life Assistant Agent** is a Retrieval-Augmented Generation (RAG) system designed to answer lifestyle-related questions (cooking, cleaning, electronics, civil law, etc.). It operates as part of a multi-agent system, exposing a Flask-based API for user interaction and inter-agent communication. It uses FAISS for vector storage and LangChain for orchestration.

## Architecture
- **Entry Point (`app.py`):** A Flask application that provides endpoints for:
    - User Chat: `/rag_answer` (persists history).
    - Agent Chat: `/agent_rag_answer` (stateless).
    - Analysis: `/analyze_conversation` (monitors external chats to pro-actively offer help).
    - Management: `/reset_history`, `/conversation_summary`.
- **Core Logic (`ai_engine_faiss.py`):**
    - Manages the FAISS vector store loaded from `home-topic-vdb/`.
    - Uses `HuggingFaceEmbeddings` (`intfloat/multilingual-e5-large`) for vectorization.
    - Connects to LLMs (Gemini, OpenAI, Claude, etc.) via `model_selection.py`.
    - Handles prompt construction and context retrieval.
- **Data Ingestion:**
    - Scripts (`*_to_vector.py`) convert raw documents (DOCX, PDF, JSONL, etc.) into FAISS indices stored in `home-topic-vdb/`.
    - `docx_to_qa/` contains logic to pre-process documents into Q&A pairs for better retrieval quality.

## Key Files & Directories
- **`app.py`**: Main Flask server. Defines API routes and handles request validation.
- **`ai_engine_faiss.py`**: Core RAG engine. Loads FAISS indices and executes queries.
- **`ai_engine.py`**: Legacy LlamaIndex implementation (deprecated/backup).
- **`home-topic-vdb/`**: Directory containing persisted FAISS vector indices.
- **`docx_to_qa/`**: Helper scripts for converting DOCX files to QA pairs.
- **`secrets.env`**: Configuration file for API keys and environment variables.
- **`Dockerfile` / `docker-compose.yml`**: Container definition for the `qasystem` service.

## Setup & Development

### Prerequisites
- Python 3.10+
- Docker (optional but recommended)
- `secrets.env` file with API keys (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, etc.).

### Building and Running
**Using Docker:**
```bash
docker compose up --build qasystem
```
This starts the Flask app on port 5000.

**Local Development:**
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run --host=0.0.0.0 --port=5000 --reload
   ```

### Data Ingestion (Vector DB Update)
When adding new documents, you must rebuild the vector indices. Run the appropriate script for your data type:
```bash
# Example for processing DOCX files
python docx_to_vector.py

# Example for JSONL files
python jsonl_to_vector_faiss.py
```
Ensure `secrets.env` is loaded or env vars are set before running these scripts.

## Coding Conventions
- **Language:** Python 3.10+.
- **Style:** PEP 8. 4-space indentation.
- **Comments:** Japanese is preferred for documentation and complex logic explanations.
- **Structure:** Separation of concerns between API (`app.py`) and Logic (`ai_engine_faiss.py`).
- **Git:** Commit messages should be present-tense (English or Japanese) and descriptive.

## Testing
- Currently, there is no automated test suite.
- **Manual Verification:**
    - Use `curl` to test endpoints:
      ```bash
      curl -X POST localhost:5000/rag_answer -H 'Content-Type: application/json' -d '{"question":"洗濯機の使い方は？"}'
      ```
    - Check `conversation_history.json` to verify state persistence.

## Critical Instruction / 重要指示
どれだけ時間とコストをかけてもよいので、確実に目標が達成されるようにしてほしい。
