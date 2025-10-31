from flask import Flask, request, jsonify, render_template
import os
import json
from dotenv import load_dotenv

# 更新後の HuggingFaceEmbeddings のインポート（langchain_huggingface パッケージから）
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# .envファイルから環境変数を読み込み
load_dotenv()

app = Flask(__name__)

# .env からシークレットキーを設定する
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")  # 万が一.envに設定がなければデフォルト値を使用

# 環境変数からAPIキーを設定する
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Gemini/OpenAI API key is not set. Please define GOOGLE_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in your environment.")

os.environ.setdefault("OPENAI_API_KEY", api_key)
base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta/openai/"
os.environ.setdefault("OPENAI_API_BASE", base_url)


# 埋め込みモデルの初期化
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
llm = ChatOpenAI(model="gemini-2.0-flash")

# ベクトルデータベースの読み込み（ここは変更なし）
VECTOR_DB_DIR = "./static/vector_db"
def load_all_vector_dbs():
    if not os.path.exists(VECTOR_DB_DIR):
        raise RuntimeError(f"Directory not found: {VECTOR_DB_DIR}")
    subdirs = [
        d for d in os.listdir(VECTOR_DB_DIR)
        if os.path.isdir(os.path.join(VECTOR_DB_DIR, d))
    ]
    if not subdirs:
        raise RuntimeError(f"No vector DB subdirectories found in {VECTOR_DB_DIR}")
    merged_db = None
    for subdir in subdirs:
        db_path = os.path.join(VECTOR_DB_DIR, subdir)
        try:
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            if merged_db is None:
                merged_db = db
            else:
                merged_db.merge_from(db)
        except Exception as e:
            print(f"Failed to load vector DB from {db_path}: {str(e)}")
    if merged_db is None:
        raise RuntimeError("Failed to create a merged FAISS DB.")
    return merged_db

vector_db = load_all_vector_dbs()
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# カスタムの combine_prompt の定義（改善版）
combine_prompt = PromptTemplate(
    template="""あなたは、資料を基にユーザーの問いに対してサポートするためにアシスタントです。
    
以下の回答候補を統合して、最終的な回答を作成してください。  
【回答候補】  
{summaries}

【統合ルール】  
- もし用意されたドキュメント内の情報が十分でない場合には、ドキュメント内の情報が不足していることを明示してから、あなたの持つ一般的な知識や会話文脈を活用し、最適な回答を生成してください。  
- 可能な場合は、既に行われた会話内容からも補足情報を取り入れて、ユーザの質問に対する有用な回答を提供してください。  
- 各候補の根拠（参照ファイル情報）がある場合、その情報を保持してください。  
- 重複する参照は１つにまとめてください。 
- 回答が完全な情報を含むよう、可能な範囲で詳細かつ実践的に記述してください。
- 重要！　必ず日本語で回答をすること！

【回答例】
    【資料に答えがある場合】
    (質問例-スイッチが入らない時にはどうすればいい？)
    - そうですね、まずは電源ケーブルがしっかりと接続されているか確認してください。
        次に、バッテリーが充電されているか確認してください。
        もしそれでもスイッチが入らない場合は、取扱説明書のトラブルシューティングのページを参照するか、カスタマーサポートにご連絡ください。

    【資料に答えがない場合】
    (質問例-この製品の最新のファームウェアのリリース日はいつですか？)
    - 最新のファームウェアのリリース日については、現在用意されている資料には記載がありません。
        しかし、一般的には、製品のウェブサイトのサポートセクションや、メーカーからのメールマガジンなどで告知されることが多いです。
        そちらをご確認いただくか、直接メーカーにお問い合わせいただくことをお勧めします。
""",
    input_variables=["summaries"]
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # 複数チャンクを統合するチェーンタイプ
    retriever=retriever,
    chain_type_kwargs={
        "combine_prompt": combine_prompt,
    },
    return_source_documents=True
)

# --- 会話履歴を JSON ファイルで管理するための関数 ---
HISTORY_FILE = "conversation_history.json"

def load_conversation_history():
    """
    JSONファイルから会話履歴を読み込む。
    存在しなければ空のリストを返す。
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 会話履歴がリスト形式で保存されていることを前提とする
                return data.get("conversation_history", [])
        except Exception as e:
            print(f"履歴の読み込みエラー: {str(e)}")
            return []
    else:
        return []

def save_conversation_history(conversation_history):
    """
    会話履歴を JSON ファイルに保存する。
    インデントを付与して整形表示する。
    """
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversation_history": conversation_history}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"履歴の保存エラー: {str(e)}")

# --- エンドポイント ---
@app.route("/rag_answer", methods=["POST"])
def rag_answer():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "質問を入力してください"}), 400

    # JSONファイルから既存の会話履歴を取得（リスト形式）
    conversation_history = load_conversation_history()

    # ユーザからの質問を履歴に追加
    conversation_history.append({"role": "User", "message": question})

    # QAチェーンへ送信するため、会話履歴全体を文字列に変換
    query_with_history = "\n".join([f"{entry['role']}: {entry['message']}" for entry in conversation_history])

    try:
        result = qa_chain({"query": query_with_history})
        answer = result["result"]
        source_docs = result["source_documents"]

        ref_dict = {}
        for doc in source_docs:
            src = doc.metadata.get("source", "不明ファイル")
            page = doc.metadata.get("page", "不明")
            if src in ref_dict:
                ref_dict[src].add(str(page))
            else:
                ref_dict[src] = {str(page)}
        references = ", ".join(
            [f"{src} (page: {', '.join(sorted(pages, key=lambda x: int(x) if x.isdigit() else x))})"
             for src, pages in ref_dict.items()]
        )
        if "【参照：" not in answer:
            final_answer = answer + "\n\n【使用したファイル】\n" + references
        else:
            final_answer = answer

        # AIからの回答も履歴に追加
        conversation_history.append({"role": "AI", "message": final_answer})
        # 更新した履歴をファイルに保存
        save_conversation_history(conversation_history)

        return jsonify({
            "answer": final_answer,
            "sources": list(ref_dict.keys())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset_history", methods=["POST"])
def reset_history():
    """
    会話履歴をリセットするエンドポイント。
    JSONファイル内の履歴を空のリストに更新します。
    """
    try:
        # 空リストで履歴を上書き
        save_conversation_history([])
        return jsonify({"status": "Conversation history reset."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
