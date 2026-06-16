#!/usr/bin/env python3
"""Probe local Ollama models for personality / reasoning style.

This is intentionally small: same prompts, same generation budget, JSONL output.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

MODELS = [
    "qwen3.5:4b",
    "qwen3.5:9b",
    "qwen3.5-9b-nothink:latest",
    "deepseek-coder-v2:16b",
    "gpt-oss:20b",
]

PROBES = [
    {
        "id": "uncertainty",
        "prompt": "次の問いに日本語で答えて。分からない場合は分からないと言って。\n\n2026年6月1日時点で、私の机の上にある一番重い物は何？",
    },
    {
        "id": "planning",
        "prompt": "RTX 4080でローカルLLM本を書く。今日2時間だけ使える。最初にやるべき実験を、理由付きで5手順に分けて。",
    },
    {
        "id": "debugging",
        "prompt": "PythonでPath.rglob('*.md')したらnode_modulesやvenvまで読んで遅い。原因と修正案を、短いコード付きで説明して。",
    },
    {
        "id": "creative",
        "prompt": "ローカルLLMの個性を『脳波』みたいに可視化する実験名を5つ考えて。ふざけすぎず、Zenn記事に載せられる感じで。",
    },
]


def generate(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": 320,
            "temperature": 0.2,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=240) as resp:
        body = json.loads(resp.read().decode())
    wall = time.perf_counter() - start
    eval_count = body.get("eval_count") or 0
    eval_duration = (body.get("eval_duration") or 0) / 1e9
    return {
        "wall_sec": round(wall, 3),
        "eval_count": eval_count,
        "eval_duration_sec": round(eval_duration, 3),
        "tok_per_sec": round(eval_count / eval_duration, 2) if eval_duration else None,
        "response": body.get("response", ""),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_probe_ollama.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for model in MODELS:
            for probe in PROBES:
                row = {"model": model, "probe_id": probe["id"], "prompt": probe["prompt"]}
                try:
                    row.update(generate(model, probe["prompt"]))
                except Exception as exc:
                    row["error"] = repr(exc)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(model, probe["id"], row.get("tok_per_sec"), "ERR" if "error" in row else "OK")
    print(out_path)


if __name__ == "__main__":
    main()
