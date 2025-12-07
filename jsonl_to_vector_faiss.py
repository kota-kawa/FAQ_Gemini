import os
import glob
import json
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# JSONL ファイルが配置されるディレクトリと、インデックスの永続化先ディレクトリ
JSONL_DIR = "./docx_to_qa/qa_data_jsonl"
INDEX_DB_DIR = "./home-topic-vdb"
os.makedirs(INDEX_DB_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": EMBEDDING_DEVICE},
)

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


def process_jsonl(jsonl_path: str) -> List[Document]:
    """JSONL を読み込み、FAISS 用の Document 配列に変換する。"""
    documents: List[Document] = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)

                if 'question' in record and 'answer' in record:
                    text = f"Q: {record['question']}\nA: {record['answer']}"
                elif 'text' in record:
                    text = record['text']
                else:
                    continue

                metadata = {
                    'source': os.path.basename(jsonl_path),
                    'line': line_idx,
                }

                for chunk_id, chunk in enumerate(split_text(text), start=1):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                'chunk_id': chunk_id,
                            },
                        )
                    )
        return documents
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return []


def create_faiss_indices():
    """JSONL ファイルから FAISS ベクトルストアを生成し保存する。"""
    jsonl_files = glob.glob(os.path.join(JSONL_DIR, '*.jsonl'))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {JSONL_DIR}")

    for jsonl_file in jsonl_files:
        name = os.path.splitext(os.path.basename(jsonl_file))[0]
        subdir = os.path.join(INDEX_DB_DIR, name)
        os.makedirs(subdir, exist_ok=True)

        docs = process_jsonl(jsonl_file)
        if not docs:
            print(f"No valid records in {jsonl_file}. Skipping...")
            continue

        persist_dir = os.path.join(subdir, 'persist')
        os.makedirs(persist_dir, exist_ok=True)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(persist_dir)

        metadata_path = os.path.join(persist_dir, 'settings.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'chunk_size': CHUNK_SIZE,
                    'chunk_overlap': CHUNK_OVERLAP,
                    'embedding_model': EMBEDDING_MODEL_NAME,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"FAISS index saved for {name} to {persist_dir}")


if __name__ == '__main__':
    create_faiss_indices()
