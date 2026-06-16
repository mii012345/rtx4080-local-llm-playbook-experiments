#!/usr/bin/env python3
"""Generate and sanity-check a tiny single-file web app with Ollama."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:4b"
PROMPT = "単一HTMLファイルで動く小さなタイマーアプリを作ってください。条件: 開始/停止/リセット、残り秒数表示、localStorage不要、CSS込み。出力はHTML全体だけ。"


def strip_fence(text: str) -> str:
    m = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()


def main() -> None:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2400},
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
    html = strip_fence(response)
    checks = {
        "has_doctype_or_html": "<!DOCTYPE" in html.upper() or "<html" in html.lower(),
        "has_script": "<script" in html.lower(),
        "has_start_label": "開始" in html or "start" in html.lower(),
        "has_stop_label": "停止" in html or "stop" in html.lower(),
        "has_reset_label": "リセット" in html or "reset" in html.lower(),
        "has_set_interval": "setInterval" in html,
        "has_clear_interval": "clearInterval" in html,
        "looks_closed": "</html>" in html.lower(),
        "had_markdown_fence": response.lstrip().startswith("````"[:3]),
    }
    app_path = BOOK_DIR / "experiments" / "tiny_timer_app.html"
    app_path.parent.mkdir(parents=True, exist_ok=True)
    app_path.write_text(html, encoding="utf-8")
    eval_count = body.get("eval_count") or 0
    eval_sec = (body.get("eval_duration") or 0) / 1e9
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": "tiny_web_app_checked",
        "model": MODEL,
        "wall_sec": round(wall, 3),
        "eval_count": eval_count,
        "eval_sec": round(eval_sec, 3) if eval_sec else None,
        "tok_per_sec_eval": round(eval_count / eval_sec, 2) if eval_sec else None,
        "html_path": str(app_path.relative_to(BOOK_DIR)),
        "checks": checks,
        "response_preview": response[:700],
    }
    out_path = BOOK_DIR / "experiments" / "tiny_web_app_check.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
