import os
import glob
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# LlamaIndex の主要モジュール
from llama_index.core import Document, Settings, PromptHelper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 修正：GoogleGenAI -> GoogleGenerativeAI に変更
from langchain_google_genai import GoogleGenerativeAI

# .env ファイルから環境変数を読み込み（必要なら）
load_dotenv()

# XML ファイルが配置されるディレクトリと、インデックスの永続化先ディレクトリ
XML_DIR = "./static/sample_FAQ_xml"
INDEX_DB_DIR = "./vdb2/vector_db_xml"

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


def process_xml(xml_path: str) -> list[Document]:
    """
    XML ファイルを読み込み、<entry> 要素ごとにテキストをチャンクに分割して
    Document オブジェクトのリストを生成する。
    <entry> 要素には 'question'/'answer' ペアまたは 'text' 要素を想定。
    各 Document には、元ファイル名とエントリ番号がメタデータとして付与される。
    """
    documents: list[Document] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        base_meta = {'source': os.path.basename(xml_path)}
        # ルート直下の各要素をエントリとして処理
        entries = list(root)

        for idx, entry in enumerate(entries):
            q_elem = entry.find('question')
            a_elem = entry.find('answer')
            text_elem = entry.find('text')

            if q_elem is not None and a_elem is not None and q_elem.text and a_elem.text:
                text = f"Q: {q_elem.text.strip()}\nA: {a_elem.text.strip()}"
            elif text_elem is not None and text_elem.text:
                text = text_elem.text.strip()
            else:
                continue

            metadata = base_meta.copy()
            metadata['entry'] = idx + 1

            for chunk in split_text(text):
                documents.append(Document(text=chunk, extra_info=metadata))

        return documents
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []

# LLM のインスタンスを生成し、Settings に設定
llm = GoogleGenerativeAI(model='gemini-2.0-flash')
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
    XML ファイルごとにベクトルインデックスを生成し、ディスクに永続化する関数
    """
    xml_files = glob.glob(os.path.join(XML_DIR, '*.xml'))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {XML_DIR}")

    try:
        from llama_index import VectorStoreIndex
    except ImportError:
        from llama_index.core.indices.vector_store.base import VectorStoreIndex

    for xml_file in xml_files:
        name = os.path.splitext(os.path.basename(xml_file))[0]
        subdir = os.path.join(INDEX_DB_DIR, name)
        os.makedirs(subdir, exist_ok=True)

        docs = process_xml(xml_file)
        if not docs:
            print(f"No valid entries in {xml_file}. Skipping...")
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
