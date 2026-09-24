# Enterprise RAG Knowledge Assistant — Complete Documentation

Everything about this project: what it does, how
every file works, the concepts behind it, why it was built this way, and the
interview questions it is likely to attract.

## Read in this order

| Document | What it covers |
|---|---|
| [01-architecture.md](01-architecture.md) | The system end to end, both pipelines, and the exact lifecycle of one question |
| [02-repository-map.md](02-repository-map.md) | Every folder and every file, with its role and dependencies |
| [03-code-walkthrough-indexing.md](03-code-walkthrough-indexing.md) | `config.py`, `documents.py`, `chunking.py`, `store.py`, `ingest.py` — line by line |
| [04-code-walkthrough-retrieval.md](04-code-walkthrough-retrieval.md) | `retrieval.py`, `assistant.py`, `app.py` — line by line |
| [05-code-walkthrough-evaluation.md](05-code-walkthrough-evaluation.md) | `dataset.py`, `metrics.py`, `judge.py`, `harness.py`, `evaluate.py` — line by line |
| [06-concepts.md](06-concepts.md) | RAG, embeddings, vector search, chunking, RRF, cross-encoders, IR metrics with worked arithmetic |
| [07-design-decisions.md](07-design-decisions.md) | Every non-obvious choice, the alternatives rejected, and the bugs found while building |
| [08-interview-questions.md](08-interview-questions.md) | 30 questions with full answers, grouped by difficulty |

## The project in one paragraph

A question-answering assistant over a private knowledge base of 76 markdown
documents describing a fictional insurance-technology company: 4 company
documents, 8 product sheets, 32 signed contracts and 32 employee records. A
question is optionally rewritten into a better search query, embedded, matched
against a Chroma vector store, the candidates are fused and re-ranked, and the
top extracts are passed to an LLM that must answer only from them. The UI shows
those extracts next to the answer. Retrieval quality is measured with MRR, nDCG,
hit@k and precision@k on a held-out question set, and answer quality is graded
separately by an LLM judge.

## The headline result

On 60 held-out questions with `all-MiniLM-L6-v2` embeddings and markdown-aware
chunking (659 chunks), measured on this machine:

| configuration | hit@k | MRR | nDCG | P@k | keyword coverage |
|---|---|---|---|---|---|
| vector search | 0.983 | 0.941 | 0.923 | 0.572 | 0.890 |
| + query rewriting | 1.000 | 0.914 | 0.918 | 0.553 | 0.907 |
| + re-ranking | **1.000** | **0.975** | **0.962** | 0.617 | **0.935** |
| + both | 1.000 | 0.958 | 0.953 | **0.622** | 0.887 |

The interesting part is not that re-ranking wins. It is *why* the numbers move
the way they do, which [06-concepts.md](06-concepts.md) and
[07-design-decisions.md](07-design-decisions.md) explain in full.
