# Synthetic local LLM notes for public RAG reproduction

These notes are synthetic fixtures for the public experiment repository. They are not copied from the book manuscript.

Ollama is convenient when trying many local models quickly. It is easy to pull models, list them, and run small probes. For exploratory work on an RTX 4080, this makes Ollama a good first step.

vLLM is better treated as a fixed API server. It exposes an OpenAI-compatible API and is useful when a RAG service or multiple agents need to send repeated requests to the same model.

On a 16GB GPU, model size is not the only constraint. vLLM can reserve substantial VRAM for KV cache, so context length and concurrency settings must be checked.