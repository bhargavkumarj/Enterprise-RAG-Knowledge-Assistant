# Enterprise RAG Knowledge Assistant

A question-answering assistant over a private, multi-document company knowledge
base — company records, product sheets, signed contracts and employee files.
Answers are grounded in retrieved extracts, and the UI shows the extracts next to
the answer so anyone reading it can check where it came from.

The retrieval side is treated as an engineering problem rather than a demo: every
change to chunking, query handling or ranking is measured with IR metrics on a
held-out question set, and answer quality is graded separately by an LLM judge.

```
knowledge base ──> chunking ──> embeddings ──> Chroma
                                                 │
question ──> rewrite ──> vector search ──> fuse ──> re-rank ──> grounded answer
                                                                     │
                                          MRR · nDCG · hit@k · LLM judge
```

## What is in here

| Path | Purpose |
|---|---|
| `rag/documents.py` | Loads the markdown knowledge base, tagging each file with its type |
| `rag/chunking.py` | Three strategies: recursive, markdown-header aware, and LLM-contextualised |
| `rag/store.py` | Chroma persistence, with a fingerprint check so queries cannot hit a store built with a different embedding model |
| `rag/retrieval.py` | Query rewriting, dual retrieval, reciprocal rank fusion, cross-encoder or LLM re-ranking |
| `rag/assistant.py` | Prompt assembly, citations, streaming |
| `rag/evaluation/` | Question set, IR metrics, LLM judge, and the harness that runs ablations |
| `ingest.py` / `evaluate.py` / `app.py` | Entry points |

## Documentation

A full documentation set lives in [`docs/`](docs/): architecture, a file-by-file
code walkthrough, the concepts behind the implementation, the reasoning for every
non-obvious decision, and 30 interview questions with worked answers. Start with
[`docs/README.md`](docs/README.md) for the reading order.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Embeddings default to a local sentence-transformers model, so indexing and
retrieval evaluation run with no API key at all. The chat model defaults to
OpenAI; set `CHAT_BACKEND=ollama` to run the whole thing locally.

## Usage

```bash
python ingest.py --strategy markdown        # build the vector store
python app.py                               # chat UI with a context panel
python evaluate.py --ablation               # compare retrieval configurations
python evaluate.py --answers                # add answer grading by the LLM judge
```

`ingest.py` takes `--strategy {recursive,markdown,contextual}` and chunk size
flags. `contextual` asks an LLM to write a situating sentence for every chunk
before it is embedded, which costs one call per chunk but helps on questions
whose answer depends on which document the text came from.

## Measured results

60 held-out questions, `all-MiniLM-L6-v2` embeddings, markdown chunking (659
chunks from 76 documents), k=6. Re-ranking by `ms-marco-MiniLM-L-6-v2`, query
rewriting by a local llama3.2.

| configuration | hit@k | MRR | nDCG | P@k | keyword coverage |
|---|---|---|---|---|---|
| vector search | 0.983 | 0.941 | 0.923 | 0.572 | 0.890 |
| + query rewriting | 1.000 | 0.914 | 0.918 | 0.553 | 0.907 |
| + re-ranking | **1.000** | **0.975** | **0.962** | 0.617 | **0.935** |
| + both | 1.000 | 0.958 | 0.953 | **0.622** | 0.887 |

Re-ranking is where the gain is: it finds the same documents as plain vector
search but puts the right one first far more often (MRR 0.941 → 0.975), and it
raises the share of returned chunks that are actually relevant. Query rewriting
on its own closed the last retrieval misses — hit rate reaches 1.000 — but with a
3B local model it also rewrote some precise questions into vaguer ones, which is
why MRR drops slightly. With a stronger rewriter the two should compose better;
as measured here, re-ranking alone is the configuration to ship.

Per-category nDCG for that configuration shows where the remaining difficulty is:

```
holistic 0.997 · relationship 0.996 · numerical 0.996 · direct_fact 0.978
comparative 0.948 · spanning 0.930 · temporal 0.891
```

Questions needing a specific date remain hardest, which is expected — dates are
weak embedding signals and appear in many chunks that are otherwise unrelated.

Reports land in `reports/` as JSON with per-question detail, so a regression in
one category is visible rather than averaged away.

## Design notes

**The held-out split is real.** `split_questions` splits the 150-question set
60/40, stratified by category, with a fixed seed. Chunking and prompt decisions
were made against the dev half; every number above comes from the other half.

**Retrieval and generation are scored separately.** A wrong answer can come from
bad retrieval or from a model ignoring good context, and one average hides which.
IR metrics score the retriever against gold keywords; the judge scores the answer
against a reference on accuracy, completeness and groundedness.

**Fusion instead of concatenation.** The original and rewritten queries each
retrieve a candidate list, and the lists are merged by reciprocal rank fusion, so
a chunk ranked reasonably by both beats one ranked first by a single query.

**The store refuses to answer with the wrong embeddings.** A manifest records the
embedding model, and opening a store built with a different one raises instead of
silently returning nonsense from an incompatible vector space.

**Two re-rankers.** `RERANKER=cross-encoder` scores every query/chunk pair with a
small local model — fast, free, no API. `RERANKER=llm` asks a chat model to rank
the candidates, which handles instructions the cross-encoder cannot but costs a
call per question.

## Verified

Indexing, retrieval, the full ablation above, the answer path and the LLM judge
were all run on this machine. The ablation numbers come from a real run against
the 60 held-out questions, not an estimate. Answers and judging were exercised
with a local llama3.2; with a frontier chat model the answer scores will be
higher than anything that small model produces.
