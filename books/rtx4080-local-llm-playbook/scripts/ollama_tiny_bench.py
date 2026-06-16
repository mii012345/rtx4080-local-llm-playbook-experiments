#!/usr/bin/env python3
"""Tiny Ollama benchmark for the RTX4080 local LLM book.

Run from the repository root:
    python3 books/rtx4080-local-llm-playbook/scripts/ollama_tiny_bench.py

The script writes experiments/ollama_tiny_bench.json so chapter 2 has a
reproducible source file instead of a copied terminal log.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODELS = ["qwen3.5:0.8b", "qwen3.5:4b", "qwen3.5:9b"]
PROMPT = "日本語で、RTX 4080でローカルLLMを動かすメリットを5つ箇条書きで説明して。"
OPTIONS = {"num_predict": 256, "temperature": 0.2}
BASE_URL = "http://127.0.0.1:11434/api/generate"


def cmd_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def nvidia_smi() -> str | None:
    return cmd_output([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,driver_version",
        "--format=csv,noheader",
    ])


def generate(model: str) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": OPTIONS,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    before_gpu = nvidia_smi()
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=240) as resp:
        body = json.loads(resp.read().decode())
    wall = time.perf_counter() - start
    after_gpu = nvidia_smi()
    eval_count = body.get("eval_count") or 0
    eval_duration = (body.get("eval_duration") or 0) / 1e9
    prompt_eval_duration = (body.get("prompt_eval_duration") or 0) / 1e9
    return {
        "model": model,
        "wall_sec": round(wall, 3),
        "eval_count": eval_count,
        "eval_duration_sec": round(eval_duration, 3),
        "prompt_eval_duration_sec": round(prompt_eval_duration, 3),
        "tok_per_sec": round(eval_count / eval_duration, 2) if eval_duration else None,
        "empty": not bool((body.get("response") or "").strip()),
        "response_preview": body.get("response", "")[:240].replace("\n", " "),
        "gpu_before": before_gpu,
        "gpu_after": after_gpu,
        "raw_timings_ns": {
            "total_duration": body.get("total_duration"),
            "load_duration": body.get("load_duration"),
            "prompt_eval_duration": body.get("prompt_eval_duration"),
            "eval_duration": body.get("eval_duration"),
        },
    }


def main() -> None:
    results = []
    for model in MODELS:
        try:
            results.append(generate(model))
        except Exception as exc:
            results.append({"model": model, "error": repr(exc)})

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "n=1 cold-ish sequential run. wall_sec includes load/model-switch overhead; tok_per_sec uses Ollama eval_duration.",
        "prompt": PROMPT,
        "options": OPTIONS,
        "results": results,
    }
    out_dir = Path(__file__).resolve().parents[1] / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ollama_tiny_bench.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
