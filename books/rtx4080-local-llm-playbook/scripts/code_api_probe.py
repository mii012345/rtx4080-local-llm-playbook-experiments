#!/usr/bin/env python3
"""Probe Ollama's OpenAI-compatible API for a small code-fix task.

This is not an IDE integration. It checks the server/API path that an editor
plugin could call: /v1/chat/completions with a code-oriented local model.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "deepseek-coder-v2:16b"
PROMPT = """次のPython関数を、timeoutとHTTPエラー処理つきに直してください。説明は短く、コードを中心に出してください。

import urllib.request

def fetch_json(url):
    data = urllib.request.urlopen(url).read()
    return json.loads(data)"""


def main() -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "あなたは安全で簡潔なPythonコードレビュアーです。"},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
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
    text = body["choices"][0]["message"].get("content", "")
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": "code_patch_openai_compatible_api",
        "api": BASE_URL,
        "model": MODEL,
        "prompt": PROMPT,
        "wall_sec": round(wall, 3),
        "completion_tokens": completion_tokens,
        "tok_per_sec_wall": round(completion_tokens / wall, 2) if wall and completion_tokens else None,
        "empty": not bool(text.strip()),
        "response": text,
        "response_preview": text[:900],
    }
    out_path = BOOK_DIR / "experiments" / "code_api_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
