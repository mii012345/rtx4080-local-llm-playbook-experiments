#!/usr/bin/env python3
"""Manual model router probe for chapter 9."""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:11434/api/generate"

TASKS = [
    {
        "task": "draft",
        "model": "qwen3.5:4b",
        "prompt": "RTX 4080でローカルLLMを試す記事の導入を、日本語で3文だけ書いて。",
    },
    {
        "task": "code_patch",
        "model": "deepseek-coder-v2:16b",
        "prompt": "Pythonでurllib.request.urlopenにtimeoutとHTTPError処理を足す最小コードを出して。説明は短く。",
    },
    {
        "task": "rag_answer",
        "model": "qwen3.5:9b",
        "prompt": "次のcontextだけを根拠に短く答えて。\n[context] RTX 4080は16GB VRAM。Ollamaはモデル探索に楽。vLLMはOpenAI互換APIと並列処理に向く。\n[question] RAG用途ではどちらを使う？",
    },
]


def generate(task: dict) -> dict:
    payload = {
        "model": task["model"],
        "prompt": task["prompt"],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 320},
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
    response = body.get("response", "")
    eval_count = body.get("eval_count") or 0
    eval_sec = (body.get("eval_duration") or 0) / 1e9
    return {
        "task": task["task"],
        "model": task["model"],
        "wall_sec": round(wall, 3),
        "eval_count": eval_count,
        "eval_sec": round(eval_sec, 3) if eval_sec else None,
        "tok_per_sec_eval": round(eval_count / eval_sec, 2) if eval_sec else None,
        "empty": not bool(response.strip()),
        "response_preview": response[:700],
    }


def main() -> None:
    results = [generate(task) for task in TASKS]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": "manual_model_router_probe",
        "router_rule": {
            "draft": "qwen3.5:4b",
            "code_patch": "deepseek-coder-v2:16b",
            "rag_answer": "qwen3.5:9b",
        },
        "results": results,
    }
    out_path = BOOK_DIR / "experiments" / "manual_model_router.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
