import os
import json
import logging
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Sequence
from dotenv import load_dotenv

# llama_index 関連
from llama_index.core import (
    ComposableGraph,
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
    PromptHelper,
)
from llama_index.core.settings import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import ResponseMode

# ── 変更点: 埋め込みを HuggingFace(E5) → Google Gemini Embedding に切替 ──
# 旧: from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai import types  # task_type や次元数などの指定用

# LLM は Gemini（LangChain 経由）を継続使用
from langchain_google_genai import GoogleGenerativeAI

# ── 環境変数 ──
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
logging.basicConfig(level=logging.DEBUG)

# ── チャンク設定（FAQ 例を想定） ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

prompt_helper = PromptHelper(
    context_window=4096,                # ← max_input_size の代替
    num_output=CHUNK_SIZE,
    chunk_overlap_ratio=CHUNK_OVERLAP / CHUNK_SIZE,
)

# ── LLM / 埋め込みモデル設定 ──
# ※ インデクシング側（jsonl_to_vector.py）は RETRIEVAL_DOCUMENT を使用。
#    本ファイル（検索側）はクエリ埋め込みなので RETRIEVAL_QUERY を使用し、次元数も一致(768)させる。
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embedding_config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768
    ),
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper

try:
    CROSS_ENCODER_RERANKER = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_n=6,
    )
except Exception:
    logging.exception("Cross-Encoder の初期化に失敗しました。フォールバックします。")
    CROSS_ENCODER_RERANKER = None


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.lower()
    return re.findall(r"[\w一-龠ぁ-んァ-ンー]+", normalized)


class SimpleBM25:
    def __init__(self, documents: Sequence[str], tokenizer=_tokenize_text):
        self._documents = list(documents)
        self._tokenizer = tokenizer
        self._tokenized_docs: List[List[str]] = []
        self._doc_freqs: List[Counter] = []
        self._df: defaultdict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        if not self._documents:
            return

        total_len = 0
        for doc in self._documents:
            tokens = self._tokenizer(doc)
            self._tokenized_docs.append(tokens)
            counter = Counter(tokens)
            self._doc_freqs.append(counter)
            for term in counter.keys():
                self._df[term] += 1
            total_len += len(tokens)
        self._avgdl = total_len / len(self._tokenized_docs) if self._tokenized_docs else 0.0
        self._idf_cache: dict[str, float] = {}

    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = self._df.get(term, 0)
        n_docs = len(self._tokenized_docs)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) if n_docs else 0.0
        self._idf_cache[term] = idf
        return idf

    def _score(self, query_tokens: Sequence[str], doc_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        if not self._tokenized_docs:
            return 0.0
        tokens = self._tokenized_docs[doc_index]
        if not tokens:
            return 0.0
        doc_len = len(tokens)
        score = 0.0
        freqs = self._doc_freqs[doc_index]
        for term in query_tokens:
            if term not in freqs:
                continue
            idf = self._idf(term)
            freq = freqs[term]
            denom = freq + k1 * (1 - b + b * doc_len / (self._avgdl or 1.0))
            score += idf * (freq * (k1 + 1)) / (denom or 1.0)
        return score

    def top_documents(self, query_tokens: Sequence[str], top_n: int = 5):
        if not self._tokenized_docs or not query_tokens:
            return []
        scores = [self._score(query_tokens, idx) for idx in range(len(self._tokenized_docs))]
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [
            (self._documents[i], scores[i], self._tokenized_docs[i])
            for i in ranked_indices
            if scores[i] > 0
        ]

    def top_terms(self, query_tokens: Sequence[str], top_docs: int = 5, top_terms: int = 8) -> List[str]:
        ranked_docs = self.top_documents(query_tokens, top_docs)
        if not ranked_docs:
            return []
        term_counter: Counter[str] = Counter()
        query_set = set(query_tokens)
        for _, _, tokens in ranked_docs:
            for token in tokens:
                if len(token) <= 1 or token in query_set:
                    continue
                term_counter[token] += 1
        return [term for term, _ in term_counter.most_common(top_terms)]


def _collect_corpus_texts(source_indices: Sequence) -> List[str]:
    corpus: List[str] = []
    for idx in source_indices:
        candidates: Iterable = []
        try:
            docstore = getattr(idx, "docstore", None)
            if docstore and getattr(docstore, "docs", None):
                candidates = docstore.docs.values()
            elif getattr(idx, "storage_context", None):
                storage = idx.storage_context
                if storage and getattr(storage, "docstore", None) and getattr(storage.docstore, "docs", None):
                    candidates = storage.docstore.docs.values()
        except Exception:
            logging.exception("Docstore からコーパス抽出に失敗しました。")
            continue

        for node in candidates:
            text: Optional[str] = None
            if hasattr(node, "text") and isinstance(node.text, str):
                text = node.text
            elif hasattr(node, "get_content"):
                try:
                    text = node.get_content()
                except Exception:
                    text = None
            if text:
                corpus.append(text)
    return corpus


def _generate_synonym_candidates(tokens: Sequence[str], candidate_terms: Sequence[str]) -> List[str]:
    synonyms: set[str] = set()
    for token in tokens:
        if len(token) <= 1:
            continue
        if token.endswith("s") and len(token) > 3:
            synonyms.add(token[:-1])
        if not token.endswith("s") and len(token) > 3:
            synonyms.add(f"{token}s")
        for candidate in candidate_terms:
            if candidate == token or len(candidate) <= 1:
                continue
            if token in candidate or candidate in token:
                synonyms.add(candidate)
                continue
            ratio = SequenceMatcher(None, token, candidate).ratio()
            if ratio >= 0.82:
                synonyms.add(candidate)
    return list(synonyms)


def _expand_query(question: str, bm25_helper: Optional[SimpleBM25]) -> str:
    tokens = _tokenize_text(question)
    if not tokens:
        return question

    expanded_terms: List[str] = []
    candidate_terms: List[str] = []
    if bm25_helper:
        candidate_terms = bm25_helper.top_terms(tokens, top_docs=6, top_terms=12)
        expanded_terms.extend(candidate_terms)

    synonym_terms = _generate_synonym_candidates(tokens, candidate_terms)
    expanded_terms.extend(synonym_terms)

    additional = [term for term in expanded_terms if term and term not in tokens]
    if not additional:
        return question

    expanded_query = question + " " + " ".join(dict.fromkeys(additional))
    logging.debug("Expanded query: %s -> %s", question, expanded_query)
    return expanded_query

# ── インデックス設定 ──
INDEX_DB_DIR = "./constitution_vector_db"
HISTORY_FILE = "conversation_history.json"

bm25_helper: Optional[SimpleBM25] = None


def load_all_indices():
    """./constitution_vector_db 以下を走査し、
    サブディレクトリごとの VectorStoreIndex をロードして返す"""
    if not os.path.exists(INDEX_DB_DIR):
        raise RuntimeError(f"Directory not found: {INDEX_DB_DIR}")

    subdirs = [
        d for d in os.listdir(INDEX_DB_DIR)
        if os.path.isdir(os.path.join(INDEX_DB_DIR, d))
    ]

    indices, summaries = [], []
    for subdir in subdirs:
        persist = os.path.join(INDEX_DB_DIR, subdir, "persist")
        if not os.path.exists(persist):
            logging.warning(f"Persist directory not found in {subdir}, skipping...")
            continue
        ctx = StorageContext.from_defaults(persist_dir=persist)
        idx = load_index_from_storage(ctx)
        indices.append(idx)
        summaries.append(f"ファイル: {subdir}")

    if not indices:
        raise RuntimeError("Failed to load any index.")
    return indices, summaries


# 一度だけロード
try:
    indices, index_summaries = load_all_indices()
    NUM_INDICES = len(indices)
    if NUM_INDICES == 1:
        graph_or_index = indices[0]
    else:
        graph_or_index = ComposableGraph.from_indices(
            VectorStoreIndex,
            indices,
            index_summaries=index_summaries,
        )
    bm25_corpus = _collect_corpus_texts(indices)
    bm25_helper = SimpleBM25(bm25_corpus) if bm25_corpus else None
except Exception:
    logging.exception("Indexの初期化に失敗しました。")
    graph_or_index = None
    NUM_INDICES = 0
    bm25_helper = None


# ── 会話履歴ユーティリティ ──
def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("conversation_history", [])
        except Exception as e:
            logging.exception(f"履歴の読み込みエラー: {e}")
    return []


def save_conversation_history(hist):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversation_history": hist}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.exception(f"履歴の保存エラー: {e}")


# ── プロンプト ──
COMBINE_PROMPT = """あなたは、資料を基にユーザーの問いに対してサポートするためのアシスタントです。

以下の回答候補を統合して、最終的な回答を作成してください。  
【回答候補】  
{summaries}

【統合ルール】  
- もし用意されたドキュメント内の情報が十分でない場合には、情報不足であることを明示し、その上であなたの知識で回答を生成してください。  
- 可能な限り、既に行われた会話内容からも補足情報を取り入れて、有用な回答を提供してください。  
- 各候補の根拠（参照ファイル情報）がある場合、その情報を保持してください。  
- 重複する参照は１つにまとめてください。 
- 回答が十分な情報を含むよう、可能な範囲で詳細に記述してください。  
- 重要！　必ず日本語で回答すること！

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
"""


# ── 公開 API ──
def get_answer(question: str):
    """質問文字列を受け取り、RAG 結果（answer, sources）を返す"""
    if graph_or_index is None:
        raise RuntimeError("インデックスが初期化されていません。")

    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    # 会話履歴に保存（ただし検索クエリ自体は履歴を結合せず、生の質問を用いる）
    history = load_conversation_history()
    history.append({"role": "User", "message": question})

    # クエリテキスト（ここでは生の質問を用いる）
    query_text = _expand_query(question, bm25_helper)

    # クエリエンジン生成（各サブインデックスを横断）
    child_kwargs = {
        "similarity_top_k": 10,
        "similarity_threshold": None,
    }
    node_postprocessors = []
    if CROSS_ENCODER_RERANKER is not None:
        node_postprocessors.append(CROSS_ENCODER_RERANKER)
        child_kwargs["node_postprocessors"] = node_postprocessors

    query_engine_kwargs = dict(
        prompt_template=COMBINE_PROMPT,
        graph_query_kwargs={"top_k": NUM_INDICES},
        child_query_kwargs=child_kwargs,
        response_mode=ResponseMode.COMPACT,
    )
    if node_postprocessors:
        query_engine_kwargs["node_postprocessors"] = node_postprocessors

    query_engine = graph_or_index.as_query_engine(**query_engine_kwargs)

    # 実行
    response = query_engine.query(query_text)
    answer = response.response

    # 参照上位 3 ファイル抽出
    nodes = getattr(response, "source_nodes", []) or []
    if nodes and CROSS_ENCODER_RERANKER is not None:
        filtered = []
        for node in nodes:
            score = getattr(node, "score", None)
            if score is None or score >= 0.1:
                filtered.append(node)
        if filtered:
            nodes = filtered
    sorted_nodes = sorted(nodes, key=lambda n: getattr(n, "score", 0), reverse=True)

    top_srcs = []
    for n in sorted_nodes:
        meta = getattr(n, "extra_info", {}) or {}
        src = meta.get("source") or getattr(n, "metadata", {}).get("source")
        if src and src != "不明ファイル" and src not in top_srcs:
            top_srcs.append(src)
        if len(top_srcs) == 3:
            break

    ref_dict = {s: set() for s in top_srcs}
    for n in nodes:
        meta = getattr(n, "extra_info", {}) or {}
        s = meta.get("source") or getattr(n, "metadata", {}).get("source")
        if s in ref_dict:
            pg = meta.get("page") or getattr(n, "metadata", {}).get("page") or "不明"
            ref_dict[s].add(str(pg))

    if ref_dict:
        refs = ", ".join(
            f"{s} (page: {', '.join(sorted(pgs))})" for s, pgs in ref_dict.items()
        )
        final = answer + "\n\n【使用したファイル】\n" + refs
    else:
        final = answer

    # 履歴保存
    history.append({"role": "AI", "message": final})
    save_conversation_history(history)

    return final, list(ref_dict.keys())


def reset_history():
    """conversation_history.json を空にする"""
    save_conversation_history([])
