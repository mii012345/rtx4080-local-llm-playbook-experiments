#!/usr/bin/env python3
"""Minimal local RAG experiment for chapter 6.

Public repo note:
This script intentionally reads synthetic fixture documents under
fixtures/rag_docs/ instead of the Zenn book manuscript. The goal is to make the
retrieval pipeline reproducible without publishing chapter source text.

Pipeline:
1. read fixture Markdown documents,
2. split them into heading/paragraph chunks,
3. embed chunks with Ollama's nomic-embed-text,
4. retrieve top-k chunks by cosine similarity,
5. pass retrieved chunks to a generation model.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = BOOK_DIR / "fixtures" / "rag_docs"
OLLAMA = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text:latest"
GENERATE_MODEL = "qwen3.5:4b"
QUESTION = "RTX 4080でRAGの回答生成を動かすなら、OllamaとvLLMをどう使い分けるべき？"
TOP_K = 4


def post_json(path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def iter_chunks() -> list[dict]:
    chunks: list[dict] = []
    for md_path in sorted(FIXTURE_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        # Drop frontmatter if present, keep headings as lightweight metadata.
        text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
        heading = ""
        buf: list[str] = []
        for line in text.splitlines():
            if line.startswith("#"):
                if buf:
                    chunk_text = "\n".join(buf).strip()
                    if len(chunk_text) > 80:
                        chunks.append({"path": md_path.name, "heading": heading, "text": chunk_text})
                    buf = []
                heading = line.lstrip("#").strip()
            elif line.strip():
                buf.append(line)
            else:
                if buf:
                    chunk_text = "\n".join(buf).strip()
                    if len(chunk_text) > 80:
                        chunks.append({"path": md_path.name, "heading": heading, "text": chunk_text})
                    buf = []
        if buf:
            chunk_text = "\n".join(buf).strip()
            if len(chunk_text) > 80:
                chunks.append({"path": md_path.name, "heading": heading, "text": chunk_text})
    if not chunks:
        raise RuntimeError(f"No fixture markdown chunks found under {FIXTURE_DIR}")
    return chunks


def embed(text: str) -> list[float]:
    body = post_json("/api/embed", {"model": EMBED_MODEL, "input": text})
    embeddings = body.get("embeddings") or []
    if not embeddings:
        raise RuntimeError(f"No embedding returned: {body}")
    return embeddings[0]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def generate(context: str) -> dict:
    prompt = f"""あなたはローカルLLM実験ログのRAG回答器です。次のcontextだけを根拠に、日本語で短く答えてください。contextにないことは推測しないでください。

[context]
{context}

[question]
{QUESTION}
"""
    start = time.perf_counter()
    body = post_json(
        "/api/generate",
        {
            "model": GENERATE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        },
    )
    wall = time.perf_counter() - start
    eval_count = body.get("eval_count") or 0
    eval_sec = (body.get("eval_duration") or 0) / 1e9
    response = body.get("response", "")
    return {
        "wall_sec": round(wall, 3),
        "eval_count": eval_count,
        "eval_sec": round(eval_sec, 3) if eval_sec else None,
        "tok_per_sec_eval": round(eval_count / eval_sec, 2) if eval_sec else None,
        "empty": not bool(response.strip()),
        "response": response,
        "response_preview": response[:700],
    }


def main() -> None:
    start = time.perf_counter()
    chunks = iter_chunks()
    query_embedding = embed(QUESTION)
    scored = []
    for i, chunk in enumerate(chunks):
        emb = embed(f"{chunk['heading']}\n{chunk['text']}")
        scored.append((cosine(query_embedding, emb), i, chunk))
    top = sorted(scored, reverse=True)[:TOP_K]
    retrieved = [
        {
            "score": round(score, 4),
            "path": chunk["path"],
            "heading": chunk["heading"],
            "text_preview": chunk["text"][:500],
        }
        for score, _, chunk in top
    ]
    context = "\n\n".join(
        f"[{item['path']} / {item['heading']} / score={item['score']}]\n{item['text_preview']}"
        for item in retrieved
    )
    generation = generate(context)
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": "rag_from_public_fixture_retrieval",
        "source_dir": str(FIXTURE_DIR.relative_to(BOOK_DIR)),
        "source_note": "Synthetic public fixtures, not book manuscript text.",
        "embed_model": EMBED_MODEL,
        "generate_model": GENERATE_MODEL,
        "question": QUESTION,
        "chunk_count": len(chunks),
        "top_k": TOP_K,
        "retrieved": retrieved,
        "generation": generation,
        "total_wall_sec": round(time.perf_counter() - start, 3),
    }
    out_path = BOOK_DIR / "experiments" / "local_rag_ollama.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
