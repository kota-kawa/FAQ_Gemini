import os
import glob
import PyPDF2
from env_loader import load_secrets_env

# secrets.envファイルから環境変数を読み込み（必要なら）
load_secrets_env()

# Gemini API キーの設定
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Gemini/OpenAI API key is not set. Please define GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in your environment.")

os.environ.setdefault("OPENAI_API_KEY", api_key)
base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta/openai/"
os.environ.setdefault("OPENAI_API_BASE", base_url)

# LlamaIndex の主要モジュール
from llama_index.core import Document, Settings, PromptHelper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_openai import ChatOpenAI

# PDFファイルが配置されるディレクトリと、インデックスの永続化先ディレクトリ
PDF_DIR = "./static/sample_FAQ_pdf"
INDEX_DB_DIR = "./vdb2/vector_db_pdf"
os.makedirs(INDEX_DB_DIR, exist_ok=True)

# テキスト分割用パラメータ
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # チャンク間の重複数

def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """与えられたテキストを固定長チャンクに分割するシンプルな実装"""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start = end - chunk_overlap
    return chunks

def process_pdf(pdf_path: str) -> list[Document]:
    """
    PDF をページごとに読み込み、テキストをチャンクに分割して Document オブジェクトのリストを生成する。
    各 Document には、元ファイル名とページ番号がメタデータとして付与される。
    """
    documents = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    metadata = {"source": os.path.basename(pdf_path), "page": page_num + 1}
                    for chunk in split_text(text):
                        documents.append(Document(text=chunk, extra_info=metadata))
        return documents
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return []

# LLM のインスタンスを生成し、Settings に直接設定する（例: GoogleGenerativeAIを使用）
llm = ChatOpenAI(model="gemini-2.0-flash")
Settings.llm = llm

# PromptHelper の初期化（(max_tokens, chunk_size, chunk_overlap_ratio) の順）
prompt_helper = PromptHelper(4096, CHUNK_SIZE, CHUNK_OVERLAP / CHUNK_SIZE)

# 埋め込みモデルは、ここでは HuggingFaceEmbedding を使用（OPENAI_API_KEY は不要）
embed_model = HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')
#embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-mpnet-base-v2')
#embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/stsb-xlm-r-multilingual')

Settings.embed_model = embed_model

def create_vector_indices():
    """
    PDF ごとに個別のベクトルインデックスを生成し、ディスクに永続化する関数
    """
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")
    
    try:
        from llama_index import VectorStoreIndex
    except ImportError:
        from llama_index.core.indices.vector_store.base import VectorStoreIndex

    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_subdir = os.path.join(INDEX_DB_DIR, pdf_name)
        os.makedirs(pdf_subdir, exist_ok=True)

        docs = process_pdf(pdf_file)
        if not docs:
            print(f"No valid text in {pdf_file}. Skipping...")
            continue

        index = VectorStoreIndex.from_documents(docs, llm=llm, prompt_helper=prompt_helper, embed_model=embed_model)

        persist_dir = os.path.join(pdf_subdir, "persist")
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir)
        print(f"Index saved for {pdf_name} to {persist_dir}")

if __name__ == "__main__":
    create_vector_indices()
