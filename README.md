# RTX 4080 Local LLM Experiment Scripts

Runnable scripts, public fixtures, and small recorded outputs for the RTX 4080 local LLM experiments.

This repository intentionally contains only reproducible code/log artifacts, not the Zenn book manuscript. The RAG fixture documents under `fixtures/rag_docs/` are synthetic public samples.

## Quick start

```bash
git clone https://github.com/mii012345/rtx4080-local-llm-playbook-experiments.git
cd rtx4080-local-llm-playbook-experiments
ollama list
ollama pull qwen3.5:0.8b
ollama pull qwen3.5:4b
ollama pull nomic-embed-text:latest
python3 books/rtx4080-local-llm-playbook/scripts/ollama_tiny_bench.py
```

Ollama must be installed and reachable at `http://127.0.0.1:11434`. Some scripts use additional local model names; adjust them to your Ollama environment if needed.

See `books/rtx4080-local-llm-playbook/scripts/README.md` for the script/output map.
