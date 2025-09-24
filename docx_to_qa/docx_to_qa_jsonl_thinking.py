#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docx_to_qa_jsonl_thinking.py
- .docx → チャンク → Q/A → JSONL
- 機能:
  * Gemini OpenAI互換: top-level reasoning_effort（none/low/medium/high）
  * OpenAI本家: reasoning={"effort": "..."} を送信
  * （任意）Geminiネイティブ thinkingBudget を extra_body.google.thinking_config で指定（--think-budget）
  * DOCX: 段落/表の出現順保持で抽出
  * 既存JSONLを読み戻して重複回避
  * 手書きリトライ（--max-retries 反映）
  * 出力JSONLとステートの**親ディレクトリ自動作成** ← ★今回の追加

依存:
  pip install openai python-dotenv python-docx tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set, Optional

from dotenv import load_dotenv
from tqdm import tqdm

# python-docx（順序保持）
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph

# OpenAI SDK（OpenAI互換でも利用）
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, InternalServerError


# =========================
# 設定
# =========================

@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_retries: int = 3
    rate_wait: float = 0.0
    think_effort: Optional[str] = None   # "none" | "low" | "medium" | "high"
    think_budget: Optional[int] = None   # -1=dynamic, 0=disable(Flash/Lite), N>=1

    @property
    def is_gemini_compat(self) -> bool:
        return "generativelanguage.googleapis.com" in (self.base_url or "")

    @property
    def is_openai(self) -> bool:
        return "api.openai.com" in (self.base_url or "") or (self.base_url or "") == ""


SYSTEM_PROMPT = (
    "You are a meticulous annotator that converts source text into exhaustive Q/A pairs.\n"
    "Constraints:\n"
    "1) Use ONLY the provided text; do NOT add external knowledge.\n"
    "2) Cover ALL atomic facts, definitions, enumerations, properties, constraints, procedures, and relationships present in the text.\n"
    "3) Keep each question focused on ONE fact/topic whenever possible.\n"
    "4) Answers must be self-contained, precise, and directly supported by the text.\n"
    "5) Output STRICT JSON with this schema ONLY: [{\"question\": str, \"answer\": str}, ...]. No preface, no trailing notes.\n"
    "6) Preserve the language of the source text (e.g., Japanese text → Japanese Q/A).\n"
)

USER_PROMPT_TEMPLATE = (
    "Convert the following SOURCE TEXT into an exhaustive set of Q/A pairs.\n"
    "Return ONLY a JSON array of objects with keys 'question' and 'answer'.\n\n"
    "SOURCE TEXT:\n"
    "{chunk}\n"
)

# =========================
# ユーティリティ
# =========================

def ensure_parent_dir(path: Optional[str]) -> None:
    """
    path の親ディレクトリを作成（既にあれば何もしない）。
    path が None/空 の場合は無視。
    """
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)

# =========================
# DOCX 抽出（段落/表の出現順を保持）
# =========================

def iter_block_items(doc: Document):
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    parts: List[str] = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            t = (block.text or "").strip()
            if t:
                parts.append(t)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = []
                for cell in row.cells:
                    tx = (cell.text or "").strip()
                    if tx:
                        cells.append(tx)
                if cells:
                    parts.append(" | ".join(cells))
    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =========================
# チャンク分割
# =========================

def chunk_text(text: str, max_chars: int = 8000, overlap: int = 400) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return chunks

# =========================
# OpenAI互換呼び出し（手書きリトライ）
# =========================

class GeminiClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = OpenAI(api_key=self.s.api_key, base_url=self.s.base_url)

    def qa_from_chunk(self, chunk: str) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk=chunk)},
        ]

        last_err: Optional[Exception] = None
        for attempt in range(1, max(1, self.s.max_retries) + 1):
            try:
                params: Dict[str, Any] = dict(
                    model=self.s.model,
                    temperature=self.s.temperature,
                    messages=messages,
                )

                # --- thinking / reasoning の付与 ---
                if self.s.is_gemini_compat:
                    # Gemini の OpenAI互換は reasoning_effort をトップレベルで受け付ける
                    if self.s.think_budget is not None:
                        params["extra_body"] = {
                            "google": {
                                "thinking_config": {
                                    "thinking_budget": self.s.think_budget
                                }
                            }
                        }
                    elif self.s.think_effort:
                        params["reasoning_effort"] = self.s.think_effort
                else:
                    # OpenAI 本家の推論モデル（o4/o3 など）
                    if self.s.think_effort:
                        params["reasoning"] = {"effort": self.s.think_effort}

                resp = self.client.chat.completions.create(**params)

                if self.s.rate_wait > 0:
                    time.sleep(self.s.rate_wait)

                content = resp.choices[0].message.content if resp and resp.choices else ""
                return self._parse_json_array(content, fallback_repair=True)

            except (APIError, RateLimitError, APITimeoutError, InternalServerError) as e:
                last_err = e
                time.sleep(min(30.0, (2.0 ** (attempt - 1)) + random.uniform(0, 0.5)))
                continue
            except Exception as e:
                last_err = e
                break

        raise last_err if last_err else RuntimeError("Unknown error in qa_from_chunk")

    @staticmethod
    def _parse_json_array(text: str, fallback_repair: bool = True) -> List[Dict[str, str]]:
        if not text:
            raise ValueError("Empty response from model.")
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = fence.group(1) if fence else text

        start_idx = candidate.find("[")
        end_idx = candidate.rfind("]")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            if not fallback_repair:
                raise ValueError("Failed to locate JSON array in response.")
            objs = re.findall(r"\{.*?\}", candidate, flags=re.DOTALL)
            repaired = []
            for o in objs:
                try:
                    it = json.loads(o)
                    q = str(it.get("question", "")).strip()
                    a = str(it.get("answer", "")).strip()
                    if q and a:
                        repaired.append({"question": q, "answer": a})
                except json.JSONDecodeError:
                    continue
            if not repaired:
                raise ValueError("Repair failed: no valid objects.")
            tqdm.write(f"[Parser] repaired {len(repaired)} item(s) from non-array text.")
            return repaired

        json_str = candidate[start_idx:end_idx + 1]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            if not fallback_repair:
                raise
            data = json.loads(re.sub(r",\s*]", "]", json_str))

        if not isinstance(data, list):
            raise ValueError("Parsed data is not a list.")

        cleaned: List[Dict[str, str]] = []
        for it in data:
            if isinstance(it, dict):
                q = str(it.get("question", "")).strip()
                a = str(it.get("answer", "")).strip()
                if q and a:
                    cleaned.append({"question": q, "answer": a})
        return cleaned

# =========================
# 重複検知（既存JSONLも読む）
# =========================

def normalize_question(q: str) -> str:
    s = re.sub(r"\s+", " ", q.strip())
    return s.lower()

def load_existing_questions(jsonl_path: str) -> Set[str]:
    seen: Set[str] = set()
    if not os.path.exists(jsonl_path):
        return seen
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                q = str(rec.get("question", "")).strip()
                if q:
                    seen.add(normalize_question(q))
            except Exception:
                continue
    return seen

# =========================
# ステート（再開用）
# =========================

@dataclass
class RunState:
    processed: Set[int]
    def to_json(self) -> Dict[str, Any]:
        return {"processed": sorted(list(self.processed))}
    @staticmethod
    def from_json(d: Dict[str, Any]) -> "RunState":
        return RunState(processed=set(d.get("processed", [])))

def load_state(path: Optional[str]) -> RunState:
    if not path or not os.path.exists(path):
        return RunState(processed=set())
    with open(path, "r", encoding="utf-8") as f:
        return RunState.from_json(json.load(f))

def save_state(path: Optional[str], state: RunState) -> None:
    if not path:
        return
    ensure_parent_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_json(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# =========================
# メイン処理
# =========================

def build_settings(args: argparse.Namespace) -> Settings:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if args.api_key:
        api_key = args.api_key
    if not api_key:
        print("ERROR: API key not found. Set GEMINI_API_KEY (or OPENAI_API_KEY) in .env or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    if args.base_url:
        base_url = args.base_url

    model = args.model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    think_effort = None
    think_budget = None
    if args.think:
        think_effort = args.think_effort
        think_budget = args.think_budget
    if think_effort and think_budget is not None:
        print("[warn] --think-effort と --think-budget は同時指定できません。--think-budget を優先します。", file=sys.stderr)
        think_effort = None

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        rate_wait=args.rate_wait,
        think_effort=think_effort,
        think_budget=think_budget,
    )

def process_docx_to_jsonl(
    in_docx: str,
    out_jsonl: str,
    settings: Settings,
    chunk_chars: int = 8000,
    chunk_overlap: int = 400,
    workers: int = 2,
    state_path: Optional[str] = None,
) -> None:
    text = extract_text_from_docx(in_docx)
    if not text:
        print("No text extracted from DOCX.", file=sys.stderr)
        return

    chunks = chunk_text(text, max_chars=chunk_chars, overlap=chunk_overlap)
    total_chunks = len(chunks)
    print(f"Extracted {total_chunks} chunk(s).")

    # 出力/ステートの親ディレクトリ作成（無ければ）
    ensure_parent_dir(out_jsonl)
    ensure_parent_dir(state_path)

    seen_questions = load_existing_questions(out_jsonl)
    if seen_questions:
        print(f"Loaded {len(seen_questions)} existing question(s) from {out_jsonl}.")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    state = load_state(state_path)
    client = GeminiClient(settings)
    pending = [i for i in range(total_chunks) if i not in state.processed]

    def task(idx: int, chunk: str):
        return idx, client.qa_from_chunk(chunk)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, tqdm(total=len(pending), desc="Chunks") as pbar:
        future_map = {ex.submit(task, i, chunks[i]): i for i in pending}
        # 'a' で開くとファイルが無くても新規作成される
        with open(out_jsonl, "a", encoding="utf-8") as fout:
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    _, qa_list = fut.result()
                except Exception as e:
                    print(f"[Chunk {idx}] ERROR: {e}", file=sys.stderr)
                    pbar.update(1)
                    continue

                written = 0
                for item in qa_list:
                    qn = normalize_question(item["question"])
                    if not qn or qn in seen_questions:
                        continue
                    seen_questions.add(qn)
                    fout.write(json.dumps({"question": item["question"], "answer": item["answer"]}, ensure_ascii=False) + "\n")
                    written += 1

                state.processed.add(idx)
                save_state(state_path, state)
                pbar.update(1)
                tqdm.write(f"[Chunk {idx}] wrote {written} Q/A")

    print("Done.")

# =========================
# CLI
# =========================

def main():
    p = argparse.ArgumentParser(description="Convert DOCX to exhaustive Q/A JSONL via OpenAI-compatible endpoint.")
    p.add_argument("input_docx", help="入力 .docx パス")
    p.add_argument("output_jsonl", help="出力 .jsonl パス（追記）")

    p.add_argument("--chunk-chars", type=int, default=8000, help="チャンク文字数上限（既定: 8000）")
    p.add_argument("--chunk-overlap", type=int, default=400, help="チャンク重なり文字数（既定: 400）")
    p.add_argument("--workers", type=int, default=2, help="並列ワーカー数（既定: 2）")
    p.add_argument("--state", dest="state_path", default=None, help="再開用ステートファイルパス（例: ./state/qa_state.json）")

    p.add_argument("--temperature", type=float, default=0.2, help="生成温度（既定: 0.2）")
    p.add_argument("--max-retries", type=int, default=3, help="API リトライ回数（既定: 3）")
    p.add_argument("--rate-wait", type=float, default=0.0, help="各API成功後の待機秒（既定: 0）")

    p.add_argument("--api-key", default=None, help="API キー（.env より優先）")
    p.add_argument("--base-url", default=None, help="OpenAI 互換エンドポイント（.env より優先）")
    p.add_argument("--model", default=None, help="モデル名（.env の GEMINI_MODEL より優先）")

    # thinking / reasoning
    p.add_argument("--think", action="store_true", help="thinking/reasoning を有効化（対応エンドポイント/モデルが必要）")
    p.add_argument("--think-effort", choices=["none", "low", "medium", "high"], default="medium",
                   help="OpenAI互換: reasoning_effort / OpenAI本家: reasoning.effort（既定: medium）")
    p.add_argument("--think-budget", type=int, default=None,
                   help="Geminiネイティブ thinkingBudget（-1=dynamic, 0=disable(Flash/Lite), N>=1）。effortと併用不可。")

    args = p.parse_args()
    s = build_settings(args)

    process_docx_to_jsonl(
        in_docx=args.input_docx,
        out_jsonl=args.output_jsonl,
        settings=s,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        workers=args.workers,
        state_path=args.state_path,
    )

if __name__ == "__main__":
    main()
