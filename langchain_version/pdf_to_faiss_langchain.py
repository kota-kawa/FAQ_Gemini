import os
import glob
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# PDFディレクトリとベクトルDB保存先ディレクトリ
PDF_DIR = "./static/pdf"
VECTOR_DB_DIR = "./static/vector_db"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# テキスト分割設定
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)

# 埋め込みモデル設定（Sentence Transformers）
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

def process_pdf(pdf_path: str) -> list[Document]:
    """PDFファイルを処理してDocumentリストを生成"""
    documents = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    metadata = {
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
        # 取得したページ単位のDocumentを分割
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return []

def create_vector_db():
    """PDFごとに別々のベクトルDBを作成"""
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_subdir = os.path.join(VECTOR_DB_DIR, pdf_name)
        os.makedirs(pdf_subdir, exist_ok=True)

        # PDFからDocumentリストを作成
        docs = process_pdf(pdf_file)
        if not docs:
            print(f"No valid text in {pdf_file}. Skipping...")
            continue

        # FAISSベクトルストアを作成して保存（PDFごとに保存）
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(pdf_subdir)
        print(f"Vector database saved for {pdf_name} to {pdf_subdir}")

if __name__ == "__main__":
    create_vector_db()
