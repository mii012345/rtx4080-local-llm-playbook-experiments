# RTX 4080 local LLM playbook scripts

このディレクトリのスクリプトは、本文中の表や観察の根拠ログを作るための小さな実験用です。

## 前提

リポジトリルートから実行します。

```bash
cd /path/to/rtx4080-local-llm-playbook-experiments
```

Ollamaが起動していることを確認します。

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags >/dev/null
```

本文で使う主なモデルは以下です。手元にない場合は `ollama pull` してください。

```bash
ollama pull qwen3.5:0.8b
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama pull qwen3.5-9b-nothink:latest
ollama pull deepseek-coder-v2:16b
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
```

## 実験対応表

以下は `books/rtx4080-local-llm-playbook/` を基準にした相対パスです。

| 章 | スクリプト | 出力 |
|---|---|---|
| 第2章 | `scripts/ollama_tiny_bench.py` | `experiments/ollama_tiny_bench.json` |
| 第3章 | `scripts/personality_probe_ollama.py` | `experiments/personality_probe_ollama.jsonl` |
| 第3章の図 | `scripts/render_personality_eeg.py` | `assets/llm-eeg-personality.svg` |
| 第4章 | `scripts/vllm_openai_bench.py` | `experiments/vllm_openai_bench.json` |
| 第6章 | `scripts/local_rag_ollama.py` | `experiments/local_rag_ollama.json` |
| 第6章の初期プローブ | `scripts/rag_probe_ollama.py` | `experiments/rag_probe_ollama.json` |
| 第7章 | `scripts/code_api_probe.py` | `experiments/code_api_probe.json` |
| 第8章 | `scripts/tiny_web_app_probe.py` | `experiments/tiny_web_app_check.json`, `experiments/tiny_timer_app.html` |
| 第9章 | `scripts/model_router_probe.py` | `experiments/manual_model_router.json` |
| 第7〜9章の初回まとめ | `scripts/downstream_tasks_ollama.py` | `experiments/downstream_tasks_ollama.json` |

## 実行例

```bash
python3 books/rtx4080-local-llm-playbook/scripts/ollama_tiny_bench.py
python3 books/rtx4080-local-llm-playbook/scripts/local_rag_ollama.py
python3 books/rtx4080-local-llm-playbook/scripts/code_api_probe.py
python3 books/rtx4080-local-llm-playbook/scripts/tiny_web_app_probe.py
python3 books/rtx4080-local-llm-playbook/scripts/model_router_probe.py
```

## 注意

- 多くの表は `n=1` の小さな実験です。厳密なベンチマークではありません。
- `wall_sec` はモデルロードや切り替えの影響を含みます。
- `tok_per_sec_eval` はOllamaの `eval_duration` から計算します。
- 実験後にVRAMを空けたい場合は `ollama ps` で常駐モデルを確認し、`ollama stop <model>` してください。
