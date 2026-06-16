#!/usr/bin/env python3
"""Make a tiny EEG-like SVG from personality probe JSONL.

This is not real neuroscience. It maps observable generation traces to waves:
- speed: tokens/sec
- verbosity: response length
- structure: markdown/list/code markers
- uncertainty: explicit "分からない" style refusal
"""
from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
IN = ROOT / "experiments" / "personality_probe_ollama.jsonl"
OUT = ROOT / "assets" / "llm-eeg-personality.svg"
PNG = REPO / "images" / "rtx4080-local-llm-playbook" / "llm-eeg-personality.png"

rows = [json.loads(line) for line in IN.read_text(encoding="utf-8").splitlines()]
by_model = defaultdict(list)
for row in rows:
    by_model[row["model"]].append(row)


def score_model(rs: list[dict]) -> dict[str, float]:
    responses = [r.get("response", "") for r in rs]
    joined = "\n".join(responses)
    avg_speed = sum(r.get("tok_per_sec") or 0 for r in rs) / max(len(rs), 1)
    avg_len = sum(len(x) for x in responses) / max(len(responses), 1)
    structure = sum(x.count("###") + x.count("- ") + x.count("```") * 2 for x in responses) / max(len(responses), 1)
    uncertainty = sum(("分から" in x or "知ることはでき" in x or "答えることはでき" in x) for x in responses) / max(len(responses), 1)
    hallucination = 1.0 if "国際宇宙ステーション" in joined or "24GB VRAM" in joined else 0.0
    empty = sum(not x.strip() for x in responses) / max(len(responses), 1)
    return {
        "speed": min(avg_speed / 300, 1),
        "verbosity": min(avg_len / 700, 1),
        "structure": min(structure / 8, 1),
        "uncertainty": uncertainty,
        "hallucination": hallucination,
        "empty": empty,
    }

metrics = {m: score_model(rs) for m, rs in by_model.items()}
width = 1100
row_h = 110
height = 90 + row_h * len(metrics)
colors = {
    "speed": "#29b6f6",
    "verbosity": "#ab47bc",
    "structure": "#66bb6a",
    "uncertainty": "#ffa726",
    "hallucination": "#ef5350",
    "empty": "#78909c",
}
labels = list(colors)

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
parts.append('<rect width="100%" height="100%" fill="#101318"/>')
parts.append('<style>text{font-family:Inter,system-ui,sans-serif;fill:#e8eaed}.muted{fill:#9aa0a6;font-size:13px}.label{font-size:15px;font-weight:700}.metric{font-size:12px}</style>')
parts.append('<text x="32" y="34" font-size="24" font-weight="800">LLM personality EEG (observable traces)</text>')
parts.append('<text x="32" y="58" class="muted">speed / verbosity / structure / uncertainty / hallucination / empty-output を波形っぽく並べた実験図</text>')
for i, lab in enumerate(labels):
    x = 680 + i * 66
    parts.append(f'<rect x="{x}" y="24" width="12" height="12" fill="{colors[lab]}" rx="3"/>')
    parts.append(f'<text x="{x+17}" y="35" class="metric">{lab}</text>')

for row_idx, (model, vals) in enumerate(metrics.items()):
    y0 = 90 + row_idx * row_h
    parts.append(f'<line x1="32" x2="1068" y1="{y0-20}" y2="{y0-20}" stroke="#2a2f37"/>')
    parts.append(f'<text x="32" y="{y0+8}" class="label">{html.escape(model)}</text>')
    base_y = y0 + 55
    x_start = 260
    x_step = 115
    for j, lab in enumerate(labels):
        v = vals[lab]
        cx = x_start + j * x_step
        amp = 8 + v * 28
        points = []
        for k in range(36):
            x = cx + k * 2.2
            yy = base_y + math.sin(k / 35 * math.tau * (1 + j * 0.25)) * amp
            points.append(f'{x:.1f},{yy:.1f}')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[lab]}" stroke-width="2.5" opacity="0.95"/>')
        parts.append(f'<text x="{cx}" y="{base_y+43}" class="metric" fill="{colors[lab]}">{lab} {v:.2f}</text>')
parts.append('</svg>')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(parts), encoding='utf-8')
print(OUT)

chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
if chrome:
    PNG.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        tmp.write(
            "<!doctype html>"
            "<meta charset='utf-8'>"
            "<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#101318}</style>"
            f"<img src='{OUT.resolve().as_uri()}' width='{width}' height='{height}' style='display:block'>"
        )
        html_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--screenshot={PNG}",
                f"--window-size={width},{height}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        html_path.unlink(missing_ok=True)
    print(PNG)
else:
    print("PNG skipped: google-chrome/chromium not found")
