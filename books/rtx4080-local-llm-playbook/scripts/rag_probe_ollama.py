#!/usr/bin/env python3
"""Minimal RAG-style Ollama probe for chapter 6.

This script does not build a vector database yet. It only checks the generation
side of a RAG pipeline: when a short retrieved context is placed in the prompt,
does the local model answer stably, and how long does it take?
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:4b"
OPTIONS = {"temperature": 0.2, "num_predict": 512}
PROMPT = """あなたはローカルLLM実験ログのRAG回答器です。次のコンテキストだけを根拠に、日本語で短く答えてください。コンテキストにないことは推測しないでください。

[context]
- RTX 4080のVRAMは16GB。
- Ollamaはモデルを雑に入れ替えて試す入口として楽。
- vLLMはOpenAI互換APIとして固定運用し、複数リクエストをさばく用途で強い。
- 0.5BモデルでもvLLMはKV cache用に大きくVRAMを確保した。

[question]
RTX 4080でRAGの回答生成を動かすなら、OllamaとvLLMをどう使い分けるべき？"""


def main() -> None:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": OPTIONS,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())
    wall = time.perf_counter() - start

    eval_count = body.get("eval_count") or 0
    eval_duration = body.get("eval_duration") or 0
    eval_sec = eval_duration / 1e9 if eval_duration else None
    response = body.get("response", "")

    result = {
        "label": "rag_from_fixed_context",
        "model": MODEL,
        "wall_sec": round(wall, 3),
        "eval_count": eval_count,
        "eval_sec": round(eval_sec, 3) if eval_sec else None,
        "tok_per_sec_eval": round(eval_count / eval_sec, 2) if eval_sec else None,
        "empty": not bool(response.strip()),
        "response_preview": response[:700],
    }

    out_path = Path(__file__).resolve().parents[1] / "experiments" / "rag_probe_ollama.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
