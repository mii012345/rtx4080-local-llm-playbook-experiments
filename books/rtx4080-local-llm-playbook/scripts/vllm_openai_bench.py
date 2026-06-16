#!/usr/bin/env python3
"""Tiny OpenAI-compatible vLLM benchmark for the RTX4080 local LLM book."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "日本語で、RTX 4080でローカルLLMを動かすメリットを5つ箇条書きで説明して。"
MAX_TOKENS = 256
TEMPERATURE = 0.2


def post_json(path: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def generate(label: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    start = time.perf_counter()
    body = post_json("/chat/completions", payload)
    wall = time.perf_counter() - start
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    text = body["choices"][0]["message"].get("content", "")
    return {
        "label": label,
        "backend": "vllm",
        "model": MODEL,
        "wall_sec": round(wall, 3),
        "completion_tokens": completion_tokens,
        "tok_per_sec_wall": round(completion_tokens / wall, 2) if wall else None,
        "response_preview": text[:120].replace("\n", " "),
    }


def main() -> None:
    results: list[dict] = []
    try:
        results.append(generate("cold_or_first"))
        warm = [generate(f"warm_{i}") for i in range(1, 6)]
        results.extend(warm)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as ex:
            concurrent_results = list(ex.map(generate, [f"concurrent4_{i}" for i in range(1, 5)]))
        concurrent_wall = time.perf_counter() - start
        results.extend(concurrent_results)
        results.append(
            {
                "label": "concurrent4_total",
                "backend": "vllm",
                "model": MODEL,
                "wall_sec": round(concurrent_wall, 3),
                "completion_tokens": sum(r.get("completion_tokens") or 0 for r in concurrent_results),
                "tok_per_sec_wall": round(
                    sum(r.get("completion_tokens") or 0 for r in concurrent_results) / concurrent_wall,
                    2,
                ) if concurrent_wall else None,
            }
        )
    except urllib.error.URLError as exc:
        results.append({"backend": "vllm", "model": MODEL, "error": repr(exc)})

    out_dir = Path(__file__).resolve().parents[1] / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vllm_openai_bench.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
