import os
import glob
import json
from dotenv import load_dotenv

# LlamaIndex の主要モジュール
from llama_index.core import Document, Settings, PromptHelper

# --- 変更点: 埋め込みを Gemini Embedding に切替 ----------------------------
# 旧: from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 新: Google GenAI Embedding ラッパーを使用（Gemini API / Vertex AI 両対応）
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# タスク種別や出力次元などの設定用（Gemini API 純正 SDK）
from google.genai import types
# ----------------------------------------------------------------------

# Gemini の LLM（生成）側は従来通り langchain-google-genai を使用
from langchain_google_genai import GoogleGenerativeAI

# .env ファイルから環境変数を読み込み（必要なら）
# ※ GOOGLE_API_KEY（Gemini API キー）を .env に設定してください
#    例: GOOGLE_API_KEY=AIzaSy...
load_dotenv()

# JSONL ファイルが配置されるディレクトリと、インデックスの永続化先ディレクトリ
JSONL_DIR = "./docx_to_qa/jsonl"
INDEX_DB_DIR = "./constitution_vector_db"
os.makedirs(INDEX_DB_DIR, exist_ok=True)

# テキスト分割用パラメータ
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # チャンク間の重複数


def process_jsonl(jsonl_path: str) -> list[Document]:
    """
    .jsonl を読み込み、レコードごとに Document を生成する。
    JSONL は { 'question': ..., 'answer': ... } または 'text' フィールドを想定。
    質問と回答のペアは 1 レコード = 1 ドキュメントとして格納し、
    元ファイル名と行番号などのメタデータを付与する。
    """
    documents = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                record = json.loads(line)
                # 質問応答形式の場合
                if 'question' in record and 'answer' in record:
                    text = f"Q: {record['question']}\nA: {record['answer']}"
                # テキストフィールドがある場合
                elif 'text' in record:
                    text = record['text']
                else:
                    continue

                metadata = {
                    'source': os.path.basename(jsonl_path),
                    'line': line_idx
                }

                if 'question' in record:
                    metadata['question'] = record['question']

                documents.append(Document(text=text, extra_info=metadata))
        return documents
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return []


# LLM のインスタンスを生成し、Settings に設定（生成モデルは既存のまま）
llm = GoogleGenerativeAI(model='gemini-2.5-flash')
Settings.llm = llm

# PromptHelper の初期化（max_tokens, chunk_size, chunk_overlap_ratio）
prompt_helper = PromptHelper(4096, CHUNK_SIZE, CHUNK_OVERLAP / CHUNK_SIZE)

# --- 変更点: 埋め込みモデルを Gemini Embedding に切替 -----------------------
# 推奨: インデクシング / 検索の双方で RETRIEVAL_DOCUMENT を指定する。
# 出力次元は 768/1536/3072 が推奨。ここではコスト/精度バランスで 768 を採用。
# モデル名は Gemini API の安定版 "gemini-embedding-001" を使用。
# ※ LlamaIndex の GoogleGenAIEmbedding は GOOGLE_API_KEY を自動検出します。
embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embedding_config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768
    ),
)
Settings.embed_model = embed_model
# ----------------------------------------------------------------------


def create_vector_indices():
    """
    JSONL ファイルごとにベクトルインデックスを生成し、ディスクに永続化する関数
    """
    jsonl_files = glob.glob(os.path.join(JSONL_DIR, '*.jsonl'))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {JSONL_DIR}")

    try:
        from llama_index import VectorStoreIndex
    except ImportError:
        from llama_index.core.indices.vector_store.base import VectorStoreIndex

    for jsonl_file in jsonl_files:
        name = os.path.splitext(os.path.basename(jsonl_file))[0]
        subdir = os.path.join(INDEX_DB_DIR, name)
        os.makedirs(subdir, exist_ok=True)

        docs = process_jsonl(jsonl_file)
        if not docs:
            print(f"No valid records in {jsonl_file}. Skipping...")
            continue

        index = VectorStoreIndex.from_documents(
            docs,
            llm=llm,
            prompt_helper=prompt_helper,
            embed_model=embed_model
        )

        persist_dir = os.path.join(subdir, 'persist')
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir)
        print(f"Index saved for {name} to {persist_dir}")


if __name__ == '__main__':
    create_vector_indices()
