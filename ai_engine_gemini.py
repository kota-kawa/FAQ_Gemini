import inspect
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
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
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.llms.langchain import LangChainLLM
try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    FlagEmbeddingReranker = None  # type: ignore[assignment]
    _FLAG_RERANKER_IMPORT_ERROR = exc
else:
    _FLAG_RERANKER_IMPORT_ERROR = None

# ── 変更点: 埋め込みを HuggingFace(E5) → Google Gemini Embedding に切替 ──
# 旧: from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai import types  # task_type や次元数などの指定用

# LLM は Gemini（LangChain 経由）を継続使用
from langchain_google_genai import GoogleGenerativeAI

# 形態素解析 / 類義語取得
try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer
except ImportError:  # pragma: no cover - runtime fallback
    sudachi_dictionary = None
    sudachi_tokenizer = None

try:  # 任意: 日本語 WordNet
    from wnja import WordNet as JapaneseWordNet
except Exception:  # pragma: no cover - optional dependency
    JapaneseWordNet = None

# ── 環境変数 ──
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
logging.basicConfig(level=logging.DEBUG)

if _FLAG_RERANKER_IMPORT_ERROR is not None:
    logging.warning(
        "FlagEmbeddingReranker の読み込みに失敗しました。"
        "パッケージ 'llama-index-postprocessor-flag-embedding-reranker>=0.3.0' の"
        " インストール状況を確認してください: %s",
        _FLAG_RERANKER_IMPORT_ERROR,
    )

# ── チャンク設定（FAQ 例を想定） ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

prompt_helper = PromptHelper(
    context_window=4096,                # ← max_input_size の代替
    num_output=CHUNK_SIZE,
    chunk_overlap_ratio=CHUNK_OVERLAP / CHUNK_SIZE,
)

# ── LLM / 埋め込みモデル設定 ──
# インデクシング / 検索の双方で RETRIEVAL_DOCUMENT を使用し、次元数を統一する。
_gemini_chat_model = GoogleGenerativeAI(model="gemini-2.5-flash")

# llama_index の LLM ラッパーは LangChainLLM を利用する。一方で、
# クエリ書き換えや類似質問生成では LangChain 側の `invoke` API を直接
# 呼び出すため、両者を明確に分離して保持しておく。
GENERATION_LLM = _gemini_chat_model
LLAMA_INDEX_LLM = LangChainLLM(llm=_gemini_chat_model, system_prompt="")

embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embedding_config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768
    ),
)

Settings.llm = LLAMA_INDEX_LLM
Settings.embed_model = embed_model
Settings.prompt_helper = prompt_helper

EVAL_DIR = Path("evaluation")
EVAL_RESULTS_FILE = EVAL_DIR / "latest_rerank_metrics.json"
COST_LOG_FILE = EVAL_DIR / "query_pipeline_stats.json"
KNOWN_QUESTIONS_FILE = EVAL_DIR / "known_questions.jsonl"

DEFAULT_CHILD_QUERY_KWARGS: Dict[str, Optional[int]] = {
    "similarity_top_k": 20,
    "similarity_threshold": None,
}
BM25_TOP_DOCS = 6
BM25_TOP_TERMS = 15
RRF_K = 60
RRF_TOP_N = 12
RERANK_TOP_N = 8
RERANK_SCORE_THRESHOLD: Optional[float] = None
SYNTHESIS_TOP_K = 6

_SUDACHI_LOCK = threading.Lock()
_SUDACHI_TOKENIZER = None
_SUDACHI_SPLIT_MODE = None
_WORDNET_INSTANCE = None
_TOKEN_PATTERN = re.compile(r"[\w一-龠ぁ-んァ-ンー]+")
_CONTENT_POS_PREFIXES: Set[str] = {"名詞", "動詞", "形容詞", "副詞"}
_EXCLUDE_SECOND_POS: Set[str] = {"数詞", "非自立"}


@dataclass
class QueryPipelineStats:
    llm_calls: int = 0
    total_latency_sec: float = 0.0
    total_prompt_chars: int = 0
    total_response_chars: int = 0
    total_token_estimate: int = 0

    def record(self, prompt_chars: int, response_chars: int, latency: float, token_estimate: Optional[int] = None) -> None:
        self.llm_calls += 1
        self.total_latency_sec += latency
        self.total_prompt_chars += max(prompt_chars, 0)
        self.total_response_chars += max(response_chars, 0)
        if token_estimate:
            self.total_token_estimate += max(token_estimate, 0)


class QueryCostMonitor:
    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._lock = threading.Lock()
        self._stats = QueryPipelineStats()

    def record(self, prompt_chars: int, response_chars: int, latency: float, token_estimate: Optional[int] = None) -> None:
        with self._lock:
            self._stats.record(prompt_chars, response_chars, latency, token_estimate)
            self._dump_locked()

    def _dump_locked(self) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(self._stats), f, ensure_ascii=False, indent=2)
        except Exception:
            logging.exception("クエリコスト情報の書き込みに失敗しました。")


cost_monitor = QueryCostMonitor(COST_LOG_FILE)


def _get_sudachi_tokenizer():
    global _SUDACHI_TOKENIZER, _SUDACHI_SPLIT_MODE
    if sudachi_dictionary is None or sudachi_tokenizer is None:
        return None, None
    if _SUDACHI_TOKENIZER is None:
        with _SUDACHI_LOCK:
            if _SUDACHI_TOKENIZER is None:
                try:
                    _SUDACHI_TOKENIZER = sudachi_dictionary.Dictionary().create()
                    _SUDACHI_SPLIT_MODE = sudachi_tokenizer.Tokenizer.SplitMode.C
                except Exception:
                    logging.exception("SudachiPy の初期化に失敗しました。フォールバックします。")
                    _SUDACHI_TOKENIZER = None
                    _SUDACHI_SPLIT_MODE = None
    return _SUDACHI_TOKENIZER, _SUDACHI_SPLIT_MODE


def _get_japanese_wordnet():
    global _WORDNET_INSTANCE
    if JapaneseWordNet is None:
        return None
    if _WORDNET_INSTANCE is None:
        try:
            _WORDNET_INSTANCE = JapaneseWordNet()
        except Exception:
            logging.exception("Japanese WordNet の初期化に失敗しました。")
            _WORDNET_INSTANCE = None
    return _WORDNET_INSTANCE


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    tokenizer_obj, split_mode = _get_sudachi_tokenizer()
    if tokenizer_obj is None or split_mode is None:
        normalized = text.lower()
        return _TOKEN_PATTERN.findall(normalized)

    tokens: List[str] = []
    try:
        morphemes = tokenizer_obj.tokenize(text, mode=split_mode)
    except Exception:
        logging.exception("SudachiPy でのトークン化に失敗しました。フォールバックします。")
        normalized = text.lower()
        return _TOKEN_PATTERN.findall(normalized)

    for morpheme in morphemes:
        pos = morpheme.part_of_speech() or ()
        if not pos or pos[0] not in _CONTENT_POS_PREFIXES:
            continue
        if len(pos) > 1 and pos[1] in _EXCLUDE_SECOND_POS:
            continue
        lemma = morpheme.dictionary_form()
        if lemma == "*":
            lemma = morpheme.normalized_form()
        lemma = (lemma or "").strip()
        if lemma:
            tokens.append(lemma)

    if tokens:
        return tokens

    normalized = text.lower()
    return _TOKEN_PATTERN.findall(normalized)


def _filter_candidate_terms_by_pos(terms: Sequence[str]) -> List[str]:
    tokenizer_obj, split_mode = _get_sudachi_tokenizer()
    if tokenizer_obj is None or split_mode is None:
        return [t for t in dict.fromkeys(terms) if len(t) > 1]

    filtered: List[str] = []
    for term in terms:
        try:
            morphemes = tokenizer_obj.tokenize(term, mode=split_mode)
        except Exception:
            continue
        for morpheme in morphemes:
            pos = morpheme.part_of_speech() or ()
            if not pos or pos[0] not in _CONTENT_POS_PREFIXES:
                continue
            if len(pos) > 1 and pos[1] in _EXCLUDE_SECOND_POS:
                continue
            lemma = morpheme.dictionary_form()
            if lemma == "*":
                lemma = morpheme.normalized_form()
            lemma = (lemma or term).strip()
            if lemma:
                filtered.append(lemma)
                break
    return list(dict.fromkeys(filtered))


def _lookup_japanese_synonyms(term: str) -> List[str]:
    wordnet = _get_japanese_wordnet()
    if wordnet is None:
        return []
    synonyms: Set[str] = set()
    try:
        for synset in wordnet.synsets(term):  # type: ignore[attr-defined]
            for lemma in getattr(synset, "lemmas", lambda: [])():
                name = getattr(lemma, "name", lambda: "")()
                if not name:
                    continue
                name = name.replace("_", " ").strip()
                if name and name != term:
                    synonyms.add(name)
    except Exception:
        logging.debug("WordNet での類義語取得に失敗しました。", exc_info=True)
    return list(synonyms)


if FlagEmbeddingReranker is not None:
    reranker_kwargs = {
        "model": "BAAI/bge-reranker-v2-m3",
        "top_n": RERANK_TOP_N,
    }
    try:
        reranker_params = inspect.signature(FlagEmbeddingReranker).parameters
    except (TypeError, ValueError):  # pragma: no cover - dynamic inspection safety
        reranker_params = {}
    if "similarity_threshold" in reranker_params:
        if RERANK_SCORE_THRESHOLD is not None:
            reranker_kwargs["similarity_threshold"] = RERANK_SCORE_THRESHOLD
    elif RERANK_SCORE_THRESHOLD is not None:
        logging.debug(
            "FlagEmbeddingReranker が similarity_threshold を未サポートのため、"
            "後段のスコアフィルタで代替します。"
        )
    try:
        FLAG_RERANKER = FlagEmbeddingReranker(**reranker_kwargs)
    except Exception:
        logging.exception("FlagEmbedding リランカーの初期化に失敗しました。フォールバックします。")
        FLAG_RERANKER = None
else:
    FLAG_RERANKER = None


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


def _parse_generated_questions(raw_text: str, max_variants: int) -> List[str]:
    if not raw_text:
        return []
    candidates: List[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.lstrip("-•*0123456789. ")
        if stripped:
            candidates.append(stripped)
        if len(candidates) >= max_variants:
            break
    return candidates[:max_variants]


def generate_semantic_queries(question: str, max_variants: int = 3) -> List[str]:
    if not question:
        return []
    prompt = (
        "以下の質問と意味が近い日本語の質問文を{count}件生成してください。"
        "箇条書きで1行ずつ出力し、元質問の語順だけを変える単純な言い換えは避けてください。\n\n"
        "元質問: {question}"
    ).format(count=max_variants, question=question)

    start = time.perf_counter()
    try:
        raw_response = GENERATION_LLM.invoke(prompt)
        if isinstance(raw_response, str):
            response_text = raw_response
        else:
            response_text = getattr(raw_response, "content", "") or str(raw_response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        variants = _parse_generated_questions(response_text, max_variants)
        latency = time.perf_counter() - start
        token_estimate = None
        usage = getattr(raw_response, "response_metadata", None)
        if isinstance(usage, dict):
            token_estimate = usage.get("token_count") or usage.get("total_tokens")
        cost_monitor.record(len(prompt), len(response_text), latency, token_estimate)
        logging.debug("LLM生成クエリ: %s", variants)
        return [v for v in variants if v and v != question]
    except Exception:
        latency = time.perf_counter() - start
        cost_monitor.record(len(prompt), 0, latency, None)
        logging.exception("Gemini によるクエリ拡張生成に失敗しました。")
        return []


def reciprocal_rank_fusion(result_sets: Sequence[Sequence[NodeWithScore]], top_n: int = RRF_TOP_N, k: int = RRF_K) -> List[NodeWithScore]:
    if not result_sets:
        return []
    fused_scores: Dict[str, float] = {}
    node_lookup: Dict[str, NodeWithScore] = {}

    for nodes in result_sets:
        for rank, node in enumerate(nodes):
            node_obj = getattr(node, "node", None)
            node_id = getattr(node_obj, "node_id", None) or getattr(node_obj, "id_", None) or getattr(node, "id", None)
            if node_id is None:
                node_id = str(id(node_obj))
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)
            if node_id not in node_lookup:
                node_lookup[node_id] = node

    ranked_ids = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    fused_nodes: List[NodeWithScore] = []
    for node_id, score in ranked_ids[:top_n]:
        original = node_lookup[node_id]
        fused_nodes.append(NodeWithScore(node=original.node, score=score))
    return fused_nodes


def _apply_reranker(nodes: Sequence[NodeWithScore], query: str) -> List[NodeWithScore]:
    if not nodes:
        return []
    if FLAG_RERANKER is None:
        return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[:RERANK_TOP_N]

    try:
        reranked = FLAG_RERANKER.postprocess_nodes(list(nodes), query_str=query)
    except Exception:
        logging.exception("FlagEmbedding リランカーでの再ランキングに失敗しました。")
        return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[:RERANK_TOP_N]

    if RERANK_SCORE_THRESHOLD is None:
        filtered: List[NodeWithScore] = list(reranked)
    else:
        filtered = []
        for node in reranked:
            score = getattr(node, "score", None)
            if score is None or score >= RERANK_SCORE_THRESHOLD:
                filtered.append(node)
    if not filtered:
        filtered = list(reranked)[:RERANK_TOP_N]
    return filtered[:RERANK_TOP_N]


def _retrieve_nodes_for_query(query_text: str):
    if RETRIEVAL_QUERY_ENGINE is None:
        return []
    try:
        response = RETRIEVAL_QUERY_ENGINE.query(query_text)
    except Exception:
        logging.exception("クエリエンジンによる検索に失敗しました。")
        return []
    return getattr(response, "source_nodes", []) or []


def _collect_results_for_queries(queries: Sequence[str]) -> List[List[NodeWithScore]]:
    results: List[List[NodeWithScore]] = []
    for query in queries:
        expanded = _rewrite_query(query, bm25_helper)
        nodes = _retrieve_nodes_for_query(expanded)
        if nodes:
            results.append(nodes)
    return results


def retrieve_reranked_nodes(question: str, use_llm_expansion: bool = True) -> Tuple[List[NodeWithScore], List[str]]:
    variants = generate_semantic_queries(question) if use_llm_expansion else []
    query_list = [question] + [v for v in variants if v]
    deduped_queries = list(dict.fromkeys(query_list))
    result_sets = _collect_results_for_queries(deduped_queries)

    if result_sets:
        fused_nodes = reciprocal_rank_fusion(result_sets)
    else:
        fallback_nodes = _retrieve_nodes_for_query(_rewrite_query(question, bm25_helper))
        fused_nodes = [
            NodeWithScore(node=getattr(node, "node", node), score=getattr(node, "score", None))
            for node in fallback_nodes
        ]

    reranked_nodes = _apply_reranker(fused_nodes, question)
    return reranked_nodes, variants


def _extract_source_info(node: NodeWithScore) -> Tuple[Optional[str], Optional[str]]:
    metadata = getattr(node, "extra_info", {}) or getattr(node, "metadata", {}) or {}
    source = metadata.get("source") or metadata.get("file")
    page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_index")
    if page is not None:
        page = str(page)
    return source, page


def _build_reference_dict(nodes: Sequence[NodeWithScore]) -> Dict[str, Set[str]]:
    references: Dict[str, Set[str]] = {}
    for node in nodes:
        source, page = _extract_source_info(node)
        if not source:
            continue
        if source not in references:
            references[source] = set()
        if page:
            references[source].add(page)
    return references


def evaluate_known_questions(
    file_path: Path = KNOWN_QUESTIONS_FILE,
    recall_targets: Sequence[int] = (1, 3, 5, 10),
) -> Optional[Dict[str, float]]:
    if graph_or_index is None or RETRIEVAL_QUERY_ENGINE is None:
        logging.info("インデックス未初期化のため評価をスキップします。")
        return None

    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        logging.info("既知質問セットが存在しないため評価をスキップします: %s", path)
        return None

    recall_counts: Dict[int, float] = {k: 0.0 for k in recall_targets}
    mrr_sum = 0.0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("評価データの JSON パースに失敗しました: %s", line)
                continue
            question = record.get("question")
            relevant_sources = record.get("relevant_sources") or []
            if not question or not relevant_sources:
                continue

            nodes, _ = retrieve_reranked_nodes(question, use_llm_expansion=False)
            ranked_sources = [
                src for src, _ in (_extract_source_info(node) for node in nodes) if src
            ]
            if not ranked_sources:
                continue

            total += 1
            relevant_set = set(relevant_sources)
            for k in recall_targets:
                top_sources = ranked_sources[:k]
                if any(src in relevant_set for src in top_sources):
                    recall_counts[k] += 1

            reciprocal_rank = 0.0
            for idx, src in enumerate(ranked_sources):
                if src in relevant_set:
                    reciprocal_rank = 1.0 / (idx + 1)
                    break
            mrr_sum += reciprocal_rank

    if total == 0:
        logging.info("評価対象の質問が見つからなかったため、結果を保存しません。")
        return None

    metrics: Dict[str, float] = {
        f"recall@{k}": recall_counts[k] / total for k in recall_targets
    }
    metrics["mrr"] = mrr_sum / total
    metrics["evaluated_questions"] = float(total)

    try:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with EVAL_RESULTS_FILE.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("評価結果の保存に失敗しました。")

    logging.info("Reranker evaluation metrics: %s", metrics)
    return metrics

_QUERY_REWRITE_CACHE: Dict[str, str] = {}


def _rewrite_query(question: str, bm25_helper: Optional[SimpleBM25]) -> str:
    normalized = (question or "").strip()
    if not normalized:
        return question

    cached = _QUERY_REWRITE_CACHE.get(normalized)
    if cached:
        return cached

    keyword_suggestions: List[str] = []
    tokens = _tokenize_text(normalized)
    if bm25_helper and tokens:
        candidate_terms = bm25_helper.top_terms(tokens, top_docs=BM25_TOP_DOCS, top_terms=BM25_TOP_TERMS)
        filtered_terms = _filter_candidate_terms_by_pos(candidate_terms)
        keyword_suggestions.extend(filtered_terms)
        for term in filtered_terms:
            keyword_suggestions.extend(_lookup_japanese_synonyms(term))

    # 重複を除去し、空白のみの語を削除
    deduped_suggestions: List[str] = []
    seen_terms: Set[str] = set()
    for term in keyword_suggestions:
        cleaned = term.strip()
        if not cleaned or cleaned in seen_terms:
            continue
        seen_terms.add(cleaned)
        deduped_suggestions.append(cleaned)

    prompt_lines = [
        "以下のユーザー質問を、検索の成功率が高くなるように自然な日本語で1文に書き換えてください。",
        "質問の意図や対象は絶対に変えず、不要な説明や出力形式の指示は付けないでください。",
        "検索に役立つキーワードがあれば自然な形で含めてください。",
    ]
    if deduped_suggestions:
        prompt_lines.append("参考になりそうなキーワード候補: " + ", ".join(deduped_suggestions[:BM25_TOP_TERMS]))
    prompt_lines.append(f"ユーザー質問: {normalized}")
    prompt_lines.append("書き換え後の質問:")

    prompt = "\n".join(prompt_lines)

    start = time.perf_counter()
    try:
        raw_response = GENERATION_LLM.invoke(prompt)
        if isinstance(raw_response, str):
            response_text = raw_response
        else:
            response_text = getattr(raw_response, "content", "") or str(raw_response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        # 先頭の非空行を採用
        rewritten_candidates = [line.strip() for line in response_text.splitlines() if line.strip()]
        rewritten_query = rewritten_candidates[0] if rewritten_candidates else normalized

        latency = time.perf_counter() - start
        token_estimate = None
        usage = getattr(raw_response, "response_metadata", None)
        if isinstance(usage, dict):
            token_estimate = usage.get("token_count") or usage.get("total_tokens")
        cost_monitor.record(len(prompt), len(response_text), latency, token_estimate)

        _QUERY_REWRITE_CACHE[normalized] = rewritten_query
        logging.debug("Rewritten query: %s -> %s", question, rewritten_query)
        return rewritten_query
    except Exception:
        latency = time.perf_counter() - start
        cost_monitor.record(len(prompt), 0, latency, None)
        logging.exception("Gemini によるクエリ書き換えに失敗しました。")
        _QUERY_REWRITE_CACHE[normalized] = normalized
        return normalized

# ── インデックス設定 ──
INDEX_DB_DIR = "./constitution_vector_db"
HISTORY_FILE = "conversation_history.json"

bm25_helper: Optional[SimpleBM25] = None
RETRIEVAL_QUERY_ENGINE = None
RESPONSE_SYNTHESIZER = None
COMBINE_PROMPT_TEMPLATE: Optional[PromptTemplate] = None


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
    RETRIEVAL_QUERY_ENGINE = graph_or_index.as_query_engine(
        graph_query_kwargs={"top_k": NUM_INDICES},
        child_query_kwargs=DEFAULT_CHILD_QUERY_KWARGS,
        response_mode=ResponseMode.NO_TEXT,
    )
except Exception:
    logging.exception("Indexの初期化に失敗しました。")
    graph_or_index = None
    NUM_INDICES = 0
    bm25_helper = None
    RETRIEVAL_QUERY_ENGINE = None


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

try:
    COMBINE_PROMPT_TEMPLATE = PromptTemplate(COMBINE_PROMPT)
except Exception:
    logging.exception("統合プロンプトの初期化に失敗しました。")
    COMBINE_PROMPT_TEMPLATE = None

if graph_or_index is not None and COMBINE_PROMPT_TEMPLATE is not None:
    try:
        RESPONSE_SYNTHESIZER = get_response_synthesizer(
            llm=LLAMA_INDEX_LLM,
            prompt_helper=prompt_helper,
            text_qa_template=COMBINE_PROMPT_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
        )
    except Exception:
        logging.exception("応答生成器の初期化に失敗しました。")
        RESPONSE_SYNTHESIZER = None

try:
    evaluate_known_questions()
except Exception:
    logging.exception("再ランキング評価の実行に失敗しました。")


# ── 公開 API ──
def get_answer(question: str):
    """質問文字列を受け取り、RAG 結果（answer, sources）を返す"""
    if graph_or_index is None or RESPONSE_SYNTHESIZER is None:
        raise RuntimeError("インデックスまたは応答生成器が初期化されていません。")

    question = question.strip()
    if not question:
        raise ValueError("質問を入力してください。")

    history = load_conversation_history()
    history.append({"role": "User", "message": question})

    reranked_nodes, generated_variants = retrieve_reranked_nodes(question, use_llm_expansion=True)

    nodes_for_answer = reranked_nodes[:SYNTHESIS_TOP_K] if reranked_nodes else []
    if not nodes_for_answer:
        nodes_for_answer = _retrieve_nodes_for_query(_rewrite_query(question, bm25_helper))

    response = RESPONSE_SYNTHESIZER.synthesize(question, nodes_for_answer)
    answer_text = getattr(response, "response", str(response))

    reference_dict = _build_reference_dict(nodes_for_answer)
    top_sources = list(reference_dict.keys())[:3]
    if reference_dict:
        refs = ", ".join(
            f"{src} (page: {', '.join(sorted(pages))})" if pages else src
            for src, pages in reference_dict.items()
        )
        final_answer = answer_text + "\n\n【使用したファイル】\n" + refs
    else:
        final_answer = answer_text

    history.append({"role": "AI", "message": final_answer, "generated_queries": generated_variants})
    save_conversation_history(history)

    return final_answer, top_sources


def reset_history():
    """conversation_history.json を空にする"""
    save_conversation_history([])
