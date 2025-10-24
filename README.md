# FAQ_Gemini

Gemini API とローカルのベクトルデータベースを組み合わせて FAQ 形式の質問に回答する Flask ベースの RAG（Retrieval Augmented Generation）システムです。Web UI からの対話と REST API の両方を提供し、会話履歴や要約表示、複数フォーマットの資料取り込みに対応します。

## システム概要

### 全体構成

- **アプリケーション層 (`app.py`)**  
  Flask アプリが HTTP API とテンプレートを提供します。CORS 設定を含み、`/rag_answer`・`/reset_history`・`/conversation_history`・`/conversation_summary` を公開します。
- **推論エンジン (`ai_engine_faiss.py`)**  
  FAISS ベクトルストアと HuggingFace 埋め込み（`intfloat/multilingual-e5-large` が既定）を利用して文書を検索し、Gemini (`gemini-2.5-flash`) で回答を生成します。会話履歴は `conversation_history.json` に保存され、必要に応じて要約を作成します。
- **データレイヤー (`constitution_vector_db/`)**  
  各資料ごとに `persist/index.faiss` を持つサブディレクトリで構成されたベクトル DB 群です。アプリ起動時に全サブディレクトリを読み込み、単一のリトリーバとして統合します。
- **Web UI (`templates/index.html`, `static/`)**  
  ブラウザから質問を送信し、回答・参照ファイル・会話要約を表示するミニマルなチャット UI を提供します。

### ベクトル化スクリプト

複数形式の資料をインデックス化する CLI スクリプトを同梱しています。いずれも HuggingFace 埋め込み + Gemini を利用して LlamaIndex 形式のベクトル DB を生成します。

| スクリプト | 役割 | 想定入力ディレクトリ | 出力先ディレクトリ |
| --- | --- | --- | --- |
| `pdf_to_vector.py` | PDF のページ単位テキストをチャンク化して保存 | `./static/sample_FAQ_pdf/` | `./vdb2/vector_db_pdf/` |
| `csv_to_vector.py` | Q/A 形式または `text` カラムを持つ CSV を取り込み | `./static/sample_FAQ_csv/` | `./vdb2/vector_db_csv/` |
| `txt_to_vector.py` / `json_to_vector.py` / `jsonl_to_vector.py` / `xml_to_vector.py` / `docx_to_vector.py` | 各フォーマット向けインデクサー | 各スクリプトの冒頭に記載 | `./vdb2/` 配下 |
| `pdf_to_faiss.py` / `jsonl_to_vector_faiss.py` | 直接 FAISS 用のインデックスを構築（Docker ではなくローカル venv 実行を想定） | スクリプト内のパス参照 | `./constitution_vector_db/` ほか |

作成した `persist/` ディレクトリを `constitution_vector_db/<dataset>/persist/` に配置することで、Flask アプリから利用できます。

## ディレクトリ構成（抜粋）

```
FAQ_Gemini/
├─ app.py                  # Flask エントリーポイント
├─ ai_engine_faiss.py      # FAISS ベースの RAG エンジン
├─ ai_engine.py            # LlamaIndex Graph を利用する旧実装
├─ constitution_vector_db/ # FAISS インデックス（persist/index.faiss など）
├─ static/                 # Web UI のスタイル・スクリプト等
├─ templates/index.html    # チャット UI
├─ vdb2/                   # 形式別の LlamaIndex 永続化例
├─ requirements.txt        # 必須 Python パッケージ
├─ docker-compose.yml      # Docker 実行定義
└─ Dockerfile              # Flask アプリ用イメージ
```

## 必要環境

- Python 3.10 以上（Docker イメージは `python:3.10-slim` ベース）
- Google Gemini API キー
- CPU 上で動作する FAISS / HuggingFace 埋め込みライブラリ（`requirements.txt` に含まれます）

## 環境変数

`.env` またはシェル環境に以下を定義してください。

| 変数 | 用途 | 備考 |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini への認証 | **必須**。`ai_engine_faiss.py` 起動時に読み込まれます。 |
| `SECRET_KEY` | Flask セッション鍵 | 未設定時は `default_secret_key`。 |
| `EMBEDDING_MODEL_NAME` | 埋め込みモデル名 | 既定値: `intfloat/multilingual-e5-large`。 |
| `EMBEDDING_DEVICE` | 埋め込みの実行デバイス | 既定値: `cpu`。GPU を利用する場合は `cuda` などに変更。 |

## セットアップ手順（ローカル）

1. リポジトリをクローンし `.env` を作成します。
   ```bash
   git clone <this-repo>
   cd FAQ_Gemini
   cp .env.example .env  # サンプルがない場合は手動で作成し、API キーを記載
   ```
2. 仮想環境を作成し依存関係をインストールします。
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. 必要な資料をベクトル化します。例として PDF を取り込む場合:
   ```bash
   python pdf_to_vector.py
   ```
   生成された `vdb2/vector_db_pdf/<ファイル名>/persist/` を `constitution_vector_db/` 配下へコピーしてからアプリを起動してください。

## アプリの起動

### Python から直接起動

```bash
python app.py
# または Flask CLI を利用する場合
flask run --host=0.0.0.0 --port=5000
```

### Docker / Docker Compose

```bash
docker compose up --build
```

`docker-compose.yml` ではカレントディレクトリ全体を `/app` にマウントし、ホストの `5000` 番ポートを公開します。既存の `constitution_vector_db` をホスト側で準備しておく必要があります。

## REST API

| メソッド | パス | 役割 | 主なレスポンス |
| --- | --- | --- | --- |
| `POST` | `/rag_answer` | 質問テキストから回答を生成 | `{ "answer": <生成回答>, "sources": [<参照ファイル>] }` |
| `POST` | `/reset_history` | 会話履歴を消去 | `{ "status": "Conversation history reset." }` |
| `GET` | `/conversation_history` | 会話履歴の配列を取得 | `{ "conversation_history": [...] }` |
| `GET` | `/conversation_summary` | 会話全体の要約を取得 | `{ "summary": "..." }` |

エラー時は `error` キーを含む JSON が 4xx/5xx ステータスで返却されます。

## Web UI

- ブラウザで `http://localhost:5000/` にアクセスするとチャット UI が表示されます。
- 要約パネルの折りたたみ、会話履歴リセットボタン、参照ファイル表示などの機能を備えています。
- UI は単一 HTML + 組み込み CSS/JavaScript で構成されており、API 呼び出し先は同一オリジンを前提としています。

## 会話履歴とログ

- 質問・回答は `conversation_history.json` に追記されます。ファイルを削除するか `/reset_history` を呼び出すことでリセットできます。
- `logging.basicConfig(level=logging.DEBUG)` が設定されているため、サーバー起動時からデバッグログが標準出力に出力されます。

## よくある注意点

- `pdf_to_faiss.py` など一部スクリプトは Docker 内での実行を想定しておらず、ローカルの仮想環境で実行すると取り回しが容易です。
- `requirements.txt` には FAISS・LlamaIndex・LangChain 関連パッケージが含まれるため、インストール時間が長くなる場合があります。
- ベクトル DB が存在しない状態でアプリを起動すると `FAISS インデックスの初期化に失敗しました。` という例外になります。`constitution_vector_db` の内容を必ず準備してください。

## 参考

- `ai_engine.py` は LlamaIndex の ComposableGraph を使った旧構成です。現在の Flask アプリは `ai_engine_faiss.py` をデフォルトで読み込みますが、必要に応じて差し替えて利用できます。
- `langchain_version/` ディレクトリには動作検証時の情報が含まれています（開発時の参照用）。

