# Synthetic task routing notes

These notes are synthetic fixtures for reproducing the retrieval pipeline without publishing the book manuscript.

For lightweight drafts and small HTML prototypes, a 4B-class model can be enough. The important check is not only speed but also whether the response is empty or malformed.

For code patch suggestions, a code-oriented local model can respond quickly, but the output still needs human review. Fast code generation can also produce over-engineered or incorrect patches.

For RAG answers, the model should answer only from retrieved context. Empty output is a quality failure even if retrieval succeeded.