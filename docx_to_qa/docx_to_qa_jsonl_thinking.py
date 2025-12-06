#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docx_to_qa_jsonl_thinking.py
- .docx を走査して Q/A を JSONL に書き出すバッチ。
- 実行は常に:  python3 docx_to_qa_jsonl_thinking.py
- thinking は常に 'high'。オプション不要。

主な改良点:
* モデル返答を JSON に強制（json_schema -> json_object -> なし の順で試行）
* パースの堅牢化（配列, {"pairs":[...]}, ```json ブロック```, Q: / A: 形式）
* モデルが flash 系なら自動で gemini-2.5-pro に切替（thinking 対応のため）
* 先頭200文字のデバッグ出力（最初の失敗のみ）
* **ステートファイル機能を完全無効化（./state を読み書きしない）**
"""

import os
import re
import sys
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, InternalServerError

_ENV_FILES = [
    Path(__file__).resolve().parent / "secrets.env",
    Path(__file__).resolve().parents[1] / "secrets.env",
]
for env_path in _ENV_FILES:
    load_dotenv(env_path, override=False)

# =========================
# 設定と定数
# =========================

SYSTEM_PROMPT = (
    "You are a meticulous annotator that converts source text into exhaustive Japanese Q/A pairs.\n"
    "Rules:\n"
    "1) Use ONLY the provided text; do not add outside knowledge.\n"
    "2) Cover ALL atomic facts, definitions, enumerations, properties, constraints, procedures, exceptions, and relationships.\n"
    "3) Keep each question focused on ONE fact/topic whenever possible.\n"
    "4) Output JSON **ONLY**.\n"
    "5) Each item: {\"question\": string, \"answer\": string}.\n"
)

USER_PROMPT_TEMPLATE = (
    "Convert the following text chunk into exhaustive Q/A pairs.\n"
    "Return ONLY JSON, no commentary.\n"
    "TEXT CHUNK:\n"
    "------------------\n"
    "{chunk}\n"
    "------------------\n"
    "Your JSON **must** be an object with key `pairs` which is an array of objects with keys `question` and `answer`.\n"
)

RETRY_PROMPT_APPEND = (
    "\nIMPORTANT: The previous output was invalid. Return ONLY valid JSON object like:\n"
    "{\"pairs\": [{\"question\": \"...\", \"answer\": \"...\"}, ...]}\n"
)

# =========================
# ユーティリティ
# =========================

def ensure_parent_dir(path: str) -> None:
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# =========================
# DOCX 読み込み（順序保持）
# =========================

def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    lines: List[str] = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            t = block.text.strip()
            if t:
                lines.append(t)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = [normalize_ws(c.text) for c in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
    return "\n".join(lines)

def chunk_text(text: str, max_chars: int = 5000, overlap: int = 400) -> List[str]:
    text = text or ""
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        if j >= n:
            break
        i = j - overlap
    return chunks

# =========================
# OpenAI 互換クライアント
# =========================

@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_retries: int = 4
    think_effort: Optional[str] = "high"  # 固定
    think_budget: Optional[int] = None    # Geminiネイティブ budget（未使用: OpenAI互換）
    rate_wait: float = 0.0

    @property
    def is_gemini_compat(self) -> bool:
        return "generativelanguage.googleapis.com" in (self.base_url or "")

class ChatClient:
    def __init__(self, s: Settings):
        self.s = s
        self.client = OpenAI(api_key=s.api_key, base_url=s.base_url)

    def _build_params(self, messages: List[Dict[str, str]], response_format_mode: str) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(
            model=self.s.model,
            temperature=self.s.temperature,
            messages=messages,
        )

        # reasoning / thinking
        if self.s.is_gemini_compat:
            # Gemini OpenAI互換: reasoning_effort or extra_body.google.thinking_config
            if self.s.think_budget is not None:
                params["extra_body"] = {"google": {"thinking_config": {"thinking_budget": self.s.think_budget}}}
            else:
                params["reasoning_effort"] = self.s.think_effort or "high"
        else:
            params["reasoning"] = {"effort": self.s.think_effort or "high"}

        # response_format
        if response_format_mode == "json_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "qa_pairs",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "pairs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "answer": {"type": "string"}
                                    },
                                    "required": ["question", "answer"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["pairs"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        elif response_format_mode == "json_object":
            params["response_format"] = {"type": "json_object"}

        return params

    def qa_from_chunk(self, chunk: str, debug_tag: str = "") -> List[Dict[str, str]]:
        """
        返答を堅牢に JSON パース。複数の response_format で試行。
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk=chunk)},
        ]

        formats_to_try = ["json_schema", "json_object", "none"]
        last_err: Optional[Exception] = None
        first_failure_logged = False

        for attempt in range(1, self.s.max_retries + 1):
            for fmt in formats_to_try:
                try:
                    params = self._build_params(messages, "none" if attempt > 1 and fmt == "json_schema" else fmt)
                    resp = self.client.chat.completions.create(**params)
                    content = (resp.choices[0].message.content if resp and resp.choices else "") or ""

                    qa = self._parse_response(content)
                    if qa:
                        return qa

                    # 失敗時ログ（最初の1回だけ）
                    if not first_failure_logged:
                        preview = normalize_ws(content)[:200]
                        tqdm.write(f"[DEBUG]{('['+debug_tag+']') if debug_tag else ''} Unparsable output preview: {preview}")
                        first_failure_logged = True

                except (APIError, RateLimitError, APITimeoutError, InternalServerError) as e:
                    last_err = e
                    time.sleep(min(30.0, (2.0 ** (attempt - 1)) + random.uniform(0, 0.5)))
                    continue
                except Exception as e:
                    last_err = e
                    break

            # 追いリトライ: プロンプトを強化
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk=chunk) + RETRY_PROMPT_APPEND},
            ]

        if last_err:
            raise last_err
        return []

    @staticmethod
    def _parse_response(text: str) -> List[Dict[str, str]]:
        """
        返答を多段で解析:
        1) 純 JSON: {"pairs":[...]}
        2) 純 JSON: [ ... ]
        3) ```json ... ``` 抽出
        4) Q: / A: パターン抽出
        """
        if not text:
            return []

        def as_list(obj) -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            for it in obj:
                if isinstance(it, dict):
                    q = normalize_ws(it.get("question", ""))
                    a = normalize_ws(it.get("answer", ""))
                    if q and a:
                        out.append({"question": q, "answer": a})
            return out

        # --- 1) {"pairs":[...]}
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("pairs"), list):
                return as_list(data["pairs"])
            if isinstance(data, list):
                return as_list(data)
        except Exception:
            pass

        # --- 2) ```json ... ```
        m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S | re.I)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, dict) and isinstance(data.get("pairs"), list):
                    return as_list(data["pairs"])
                if isinstance(data, list):
                    return as_list(data)
            except Exception:
                pass

        # --- 3) 末尾の配列だけ抜く
        m = re.search(r"(\[\s*\{.*\}\s*\])\s*$", text, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, list):
                    return as_list(data)
            except Exception:
                pass

        # --- 4) Q: / A: を素朴に抽出（最後の砦）
        qa_list: List[Dict[str, str]] = []
        blocks = re.split(r"(?:^|\n)Q\s*[:：]", text)
        for b in blocks[1:]:
            parts = re.split(r"\nA\s*[:：]", b, maxsplit=1)
            if len(parts) == 2:
                q = normalize_ws(parts[0])
                a = normalize_ws(parts[1].split("\nQ:")[0])
                if q and a:
                    qa_list.append({"question": q, "answer": a})
        return qa_list

# =========================
# JSONL
# =========================

def load_existing_questions(jsonl_path: str) -> Set[str]:
    """
    既存 JSONL の重複排除用。ステート無効でも、重複Qの多重書き込みは避ける。
    """
    seen: Set[str] = set()
    if not os.path.exists(jsonl_path):
        return seen
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = normalize_ws(obj.get("question", ""))
                    if q:
                        seen.add(q)
                except Exception:
                    continue
    except Exception:
        pass
    return seen

def append_jsonl(jsonl_path: str, qa_list: List[Dict[str, str]], seen: Set[str]) -> int:
    if not qa_list:
        return 0
    ensure_parent_dir(jsonl_path)
    wrote = 0
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for qa in qa_list:
            q = normalize_ws(qa.get("question", ""))
            a = normalize_ws(qa.get("answer", ""))
            if not q or not a:
                continue
            if q in seen:
                continue
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            seen.add(q)
            wrote += 1
    return wrote

# =========================
# 単一ファイル処理（ステートなし）
# =========================

def process_docx_to_jsonl(
    in_docx: str,
    out_jsonl: str,
    settings: Settings,
    chunk_chars: int = 5000,
    chunk_overlap: int = 400,
    workers: int = 2,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    text = extract_text_from_docx(in_docx)
    if not text.strip():
        print(f"[WARN] No text extracted: {in_docx}", file=sys.stderr)
        return

    chunks = chunk_text(text, max_chars=chunk_chars, overlap=chunk_overlap)
    if not chunks:
        print(f"[WARN] No chunks for: {in_docx}", file=sys.stderr)
        return

    ensure_parent_dir(out_jsonl)

    seen = load_existing_questions(out_jsonl)
    if seen:
        print(f"Loaded {len(seen)} existing Qs from {out_jsonl}")

    client = ChatClient(settings)
    indices = list(range(len(chunks)))

    def task(idx: int) -> Tuple[int, List[Dict[str, str]]]:
        qa = client.qa_from_chunk(chunks[idx], debug_tag=f"{Path(in_docx).name}#{idx}")
        return idx, qa

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, tqdm(total=len(indices), desc="Chunks") as pbar:
        futmap = {ex.submit(task, i): i for i in indices}
        for fut in as_completed(futmap):
            idx = futmap[fut]
            try:
                _, qa_list = fut.result()
            except Exception as e:
                # ステートなし: エラーでもスキップのみ。再実行すると再トライされる。
                print(f"[Chunk {idx}] ERROR: {e}", file=sys.stderr)
                pbar.update(1)
                continue

            wrote = append_jsonl(out_jsonl, qa_list, seen)
            pbar.update(1)
            tqdm.write(f"[Chunk {idx}] wrote {wrote} Q/A")

    print("Done.")

# =========================
# メイン（ディレクトリ一括・ステートなし）
# =========================

def main():
    # ハードコード + 環境変数で上書き可
    for env_path in _ENV_FILES:
        load_dotenv(env_path, override=False)

    INPUT_DIR = Path(os.getenv("DOCX_IN_DIR", "./home-topic")).resolve()
    OUTPUT_DIR = Path(os.getenv("JSONL_OUT_DIR", "./qa_data_jsonl")).resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or OPENAI_API_KEY in secrets.env", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    # flash 強制回避（thinkingが効かないため）
    if "flash" in model.lower():
        print(f"[Info] Model '{model}' does not support reasoning high. Using 'gemini-2.5-pro' instead.")
        model = "gemini-2.5-pro"

    settings = Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.2,
        max_retries=4,
        think_effort="high",
        think_budget=None,
        rate_wait=0.0,
    )

    # 入力 .docx 一覧（~$ など一時ファイル除外）
    docs = sorted([p for p in INPUT_DIR.glob("*.docx") if p.is_file() and not p.name.startswith("~$")])
    if not docs:
        print(f"[Info] No .docx files found in {INPUT_DIR}")
        return

    print(f"[Start] Converting {len(docs)} file(s)")
    print(f"  INPUT_DIR = {INPUT_DIR}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"  Model={settings.model} BaseURL={settings.base_url} ReasoningEffort=high")
    print(f"  State=disabled (no resume; every run reprocesses all chunks)")

    for docx_path in docs:
        out_path = OUTPUT_DIR / f"{docx_path.stem}.jsonl"

        print(f"\n[File] {docx_path.name} -> {out_path.name}")
        try:
            process_docx_to_jsonl(
                in_docx=str(docx_path),
                out_jsonl=str(out_path),
                settings=settings,
                chunk_chars=5000,
                chunk_overlap=400,
                workers=2,
            )
        except Exception as e:
            print(f"[Error] {docx_path.name}: {e}", file=sys.stderr)
            continue

    print("\n[Done] All files processed.")

if __name__ == "__main__":
    main()
