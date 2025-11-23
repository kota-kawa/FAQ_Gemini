import os
import glob
import json
from env_loader import load_secrets_env

# LlamaIndex の主要モジュール
from llama_index.core import Document, Settings, PromptHelper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 修正：GoogleGenAI -> GoogleGenerativeAI に変更
from langchain_openai import ChatOpenAI

# secrets.env から環境変数を読み込み（必要なら）
load_secrets_env()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Gemini/OpenAI API key is not set. Please define GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in your environment.")

os.environ.setdefault("OPENAI_API_KEY", api_key)
base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta/openai/"
os.environ.setdefault("OPENAI_API_BASE", base_url)

# JSON ファイルが配置されるディレクトリと、インデックスの永続化先ディレクトリ
JSON_DIR = "./static/sample_FAQ_json"
INDEX_DB_DIR = "./vdb2/vector_db_json"

os.makedirs(INDEX_DB_DIR, exist_ok=True)

# テキスト分割用パラメータ
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # チャンク間の重複数

def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """与えられたテキストを固定長チャンクに分割するシンプルな実装"""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - chunk_overlap
    return chunks


def process_json(json_path: str) -> list[Document]:
    """
    JSON ファイルを読み込み、エントリごとにテキストをチャンクに分割して
    Document オブジェクトのリストを生成する。
    JSON はリストまたはオブジェクト形式で、'question'/'answer' ペアまたは 'text' キーを想定。
    各 Document には、元ファイル名とエントリ番号がメタデータとして付与される。
    """
    documents: list[Document] = []
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        base_meta = {'source': os.path.basename(json_path)}
        entries = data if isinstance(data, list) else [data]

        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            # 質問応答形式の場合
            if 'question' in entry and 'answer' in entry:
                text = f"Q: {entry['question']}\nA: {entry['answer']}"
            # テキストフィールドがある場合
            elif 'text' in entry:
                text = entry['text']
            else:
                continue

            metadata = base_meta.copy()
            metadata['entry'] = idx + 1

            for chunk in split_text(text):
                documents.append(Document(text=chunk, extra_info=metadata))

        return documents
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return []

# LLM のインスタンスを生成し、Settings に設定
llm = ChatOpenAI(model='gemini-2.0-flash')
Settings.llm = llm

# PromptHelper の初期化（max_tokens, chunk_size, chunk_overlap_ratio）
prompt_helper = PromptHelper(4096, CHUNK_SIZE, CHUNK_OVERLAP / CHUNK_SIZE)

# 埋め込みモデルを設定
embed_model = HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')
#embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-mpnet-base-v2')
#embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/stsb-xlm-r-multilingual')

Settings.embed_model = embed_model

def create_vector_indices():
    """
    JSON ファイルごとにベクトルインデックスを生成し、ディスクに永続化する関数
    """
    json_files = glob.glob(os.path.join(JSON_DIR, '*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {JSON_DIR}")

    try:
        from llama_index import VectorStoreIndex
    except ImportError:
        from llama_index.core.indices.vector_store.base import VectorStoreIndex

    for json_file in json_files:
        name = os.path.splitext(os.path.basename(json_file))[0]
        subdir = os.path.join(INDEX_DB_DIR, name)
        os.makedirs(subdir, exist_ok=True)

        docs = process_json(json_file)
        if not docs:
            print(f"No valid entries in {json_file}. Skipping...")
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
