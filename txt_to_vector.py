import os
import glob
from dotenv import load_dotenv

# LlamaIndex の主要モジュール
from llama_index.core import Document, Settings, PromptHelper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 修正：GoogleGenAI -> GoogleGenerativeAI に変更
from langchain_openai import ChatOpenAI

# .env ファイルから環境変数を読み込み（必要なら）
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Gemini/OpenAI API key is not set. Please define GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in your environment.")

os.environ.setdefault("OPENAI_API_KEY", api_key)
base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta/openai/"
os.environ.setdefault("OPENAI_API_BASE", base_url)

# TXT ファイルが配置されるディレクトリと、インデックスの永続化先ディレクトリ
TXT_DIR = "./static/sample_FAQ_txt"
INDEX_DB_DIR = "./vdb2/vector_db_txt"

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


def process_txt(txt_path: str) -> list[Document]:
    """
    TXT ファイルを読み込み、テキストをチャンクに分割して
    Document オブジェクトのリストを生成する。
    各 Document には、元ファイル名がメタデータとして付与される。
    """
    documents: list[Document] = []
    try:
        with open(txt_path, encoding='utf-8') as f:
            text = f.read()

        metadata = {
            'source': os.path.basename(txt_path)
        }
        for chunk in split_text(text):
            documents.append(Document(text=chunk, extra_info=metadata))
        return documents

    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
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
    TXT ファイルごとにベクトルインデックスを生成し、ディスクに永続化する関数
    """
    txt_files = glob.glob(os.path.join(TXT_DIR, '*.txt'))
    if not txt_files:
        raise FileNotFoundError(f"No TXT files found in {TXT_DIR}")

    try:
        from llama_index import VectorStoreIndex
    except ImportError:
        from llama_index.core.indices.vector_store.base import VectorStoreIndex

    for txt_file in txt_files:
        name = os.path.splitext(os.path.basename(txt_file))[0]
        subdir = os.path.join(INDEX_DB_DIR, name)
        os.makedirs(subdir, exist_ok=True)

        docs = process_txt(txt_file)
        if not docs:
            print(f"No valid content in {txt_file}. Skipping...")
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