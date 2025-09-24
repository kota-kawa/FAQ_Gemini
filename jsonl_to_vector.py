import os
import glob
import json
from dotenv import load_dotenv

# LlamaIndex の主要モジュール
from llama_index.core import Document, Settings, PromptHelper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 修正：GoogleGenAI -> GoogleGenerativeAI に変更
from langchain_google_genai import GoogleGenerativeAI

# .env ファイルから環境変数を読み込み（必要なら）
load_dotenv()

# JSONL ファイルが配置されるディレクトリと、インデックスの永続化先ディレクトリ
JSONL_DIR = "./docx_to_qa/jsonl/preamble.jsonl"
INDEX_DB_DIR = "./constitution_vector_db"
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


def process_jsonl(jsonl_path: str) -> list[Document]:
    """
    .jsonl を読み込み、レコードごとにテキストをチャンクに分割して
    Document オブジェクトのリストを生成する。
    JSONL は { 'question': ..., 'answer': ... } または 'text' フィールドを想定。
    各 Document には、元ファイル名と行番号がメタデータとして付与される。
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
                for chunk in split_text(text):
                    documents.append(Document(text=chunk, extra_info=metadata))
        return documents
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return []

# LLM のインスタンスを生成し、Settings に設定
llm = GoogleGenerativeAI(model='gemini-2.5-flash')
Settings.llm = llm

# PromptHelper の初期化（max_tokens, chunk_size, chunk_overlap_ratio）
prompt_helper = PromptHelper(4096, CHUNK_SIZE, CHUNK_OVERLAP / CHUNK_SIZE)

# 埋め込みモデルを設定
#embed_model = HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')
embed_model = HuggingFaceEmbedding(model_name='cl-nagoya/ruri-v3-310m')
#embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/stsb-xlm-r-multilingual')

Settings.embed_model = embed_model

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
