#!/usr/bin/env python3
"""Small downstream-task probes for the RTX4080 local LLM book.

This is not a full benchmark. It sends a few practical prompts to locally
available Ollama models and records wall time, Ollama timing fields, and a short
preview so the chapters can be grounded in actual outputs.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:11434/api/generate"
OPTIONS = {"temperature": 0.2, "num_predict": 512}

TASKS = [
    {
        "label": "rag_from_fixed_context",
        "model": "qwen3.5:4b",
        "prompt": """あなたはローカルLLM実験ログのRAG回答器です。次のコンテキストだけを根拠に、日本語で短く答えてください。コンテキストにないことは推測しないでください。\n\n[context]\n- RTX 4080のVRAMは16GB。\n- Ollamaはモデルを雑に入れ替えて試す入口として楽。\n- vLLMはOpenAI互換APIとして固定運用し、複数リクエストをさばく用途で強い。\n- 0.5BモデルでもvLLMはKV cache用に大きくVRAMを確保した。\n\n[question]\nRTX 4080でRAGの回答生成を動かすなら、OllamaとvLLMをどう使い分けるべき？""",
    },
    {
        "label": "code_completion_patch",
        "model": "deepseek-coder-v2:16b",
        "prompt": """次のPython関数を、timeoutとHTTPエラー処理つきに直してください。説明は短く、コードを中心に出してください。\n\nimport urllib.request\n\ndef fetch_json(url):\n    data = urllib.request.urlopen(url).read()\n    return json.loads(data)""",
    },
    {
        "label": "tiny_web_app",
        "model": "qwen3.5:4b",
        "prompt": """単一HTMLファイルで動く小さなタイマーアプリを作ってください。条件: 開始/停止/リセット、残り秒数表示、localStorage不要、CSS込み。出力はHTML全体だけ。""",
    },
    {
        "label": "model_router_plan",
        "model": "qwen3.5-9b-nothink:latest",
        "prompt": """RTX 4080 16GBで、次の3用途をローカルLLMに分担させる構成案を短く作ってください。用途: 1) 軽い下書き 2) コード修正 3) RAG回答。モデル候補: qwen3.5:4b, qwen3.5:9b, deepseek-coder-v2:16b, vLLMのOpenAI互換API。""",
    },
]


def generate(task: dict) -> dict:
    payload = {
        "model": task["model"],
        "prompt": task["prompt"],
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
    return {
        "label": task["label"],
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
    out_path = Path(__file__).resolve().parents[1] / "experiments" / "downstream_tasks_ollama.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
