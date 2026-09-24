# Repository Map

Every folder and every file in this repository.

```
enterprise-rag-assistant/
├── README.md                     project-facing readme (setup, usage, results)
├── requirements.txt              12 pinned-minimum dependencies
├── .env.example                  every environment variable, documented
├── .gitignore                    excludes .env, vector_store/, reports/, caches
│
├── ingest.py                     ENTRY POINT — build the vector store
├── evaluate.py                   ENTRY POINT — measure retrieval and answers
├── app.py                        ENTRY POINT — Gradio chat UI
│
├── data/
│   ├── knowledge_base/           76 markdown documents, the private corpus
│   │   ├── company/              4 files: about, careers, culture, overview
│   │   ├── products/             8 files: one per insurance product
│   │   ├── contracts/            32 files: signed customer contracts
│   │   └── employees/            32 files: employee records and reviews
│   └── questions.jsonl           150 evaluation questions
│
├── rag/                          the library
│   ├── __init__.py               re-exports Settings and the settings singleton
│   ├── config.py                 all configuration, model factories, fingerprint
│   ├── documents.py              loading markdown into LangChain Documents
│   ├── chunking.py               three chunking strategies
│   ├── store.py                  Chroma build/open/stats, manifest safety check
│   ├── retrieval.py              rewrite, search, fuse, re-rank
│   ├── assistant.py              prompt assembly, answering, streaming
│   └── evaluation/
│       ├── __init__.py           re-exports the public evaluation API
│       ├── dataset.py            question model, loading, stratified split
│       ├── metrics.py            hit@k, MRR, nDCG, P@k, keyword coverage
│       ├── judge.py              LLM-as-judge with a structured verdict
│       └── harness.py            runs configurations, aggregates, reports
│
└── (generated, git-ignored)
    ├── vector_store/             Chroma SQLite + manifest.json
    └── reports/                  timestamped JSON evaluation reports
```

## What each file is responsible for

### Entry points

**`ingest.py`** — The only thing that writes to the vector store. Takes
`--strategy`, `--chunk-size`, `--chunk-overlap`, applies them to the settings
singleton, loads documents, chunks them, prints the chunk count and median chunk
length, and builds the store. Prints the embedding fingerprint so you know what
the store can be queried with.

**`evaluate.py`** — Loads and splits the question set, decides which
configurations to run (one, or four in `--ablation` mode), runs them through the
harness, prints a comparison table and per-category nDCG for the best
configuration, and saves a JSON report. Never writes to the vector store.

**`app.py`** — Gradio UI. Two columns: chat on the left, retrieved context on the
right. Reads `store.stats()` to show an honest status line and degrades to a
warning if the store is missing rather than crashing on launch.

### The library

**`rag/config.py`** — Single source of truth. Everything tunable is an
environment variable with a default. Provides three factories (`embeddings()`,
`chat_model()`, `judge()`) so no other module imports a provider SDK directly,
and `fingerprint()`, which identifies the embedding space.

**`rag/documents.py`** — Turns the filesystem into `Document` objects. One
document per file — chunking happens later and needs the whole document as
context. Records `source` (filename stem), `path` (relative, used to join chunks
back to their parent) and `doc_type` (parent folder).

**`rag/chunking.py`** — Three strategies behind one dispatch function, registered
in a `STRATEGIES` dict that `ingest.py` reads to build its `--strategy` choices.
Adding a strategy means adding one function and one dict entry.

**`rag/store.py`** — Owns Chroma. `build()` wipes and rebuilds, writing a
manifest. `open_store()` is `lru_cache`d and validates the manifest fingerprint
before returning a handle. `stats()` merges the manifest with the live vector
count for the UI.

**`rag/retrieval.py`** — The interesting module. Query rewriting, vector search,
reciprocal rank fusion, two re-rankers, and `retrieve()` which sequences them.
Defines `Chunk` (a provider-neutral result) and `RetrievalTrace` (what happened,
not just what was returned).

**`rag/assistant.py`** — Prompt construction and the two generation modes
(`ask` blocking, `stream` incremental). Defines `Answer` with a `sources`
property that de-duplicates while preserving rank order.

**`rag/evaluation/dataset.py`** — `Question` pydantic model and the stratified
dev/holdout split. The split is deterministic (`random.Random(42)`), so the
holdout set is the same on every machine.

**`rag/evaluation/metrics.py`** — Pure functions, no I/O, no model calls. Given
chunks and gold keywords, return numbers. Trivially unit-testable.

**`rag/evaluation/judge.py`** — One prompt, one pydantic `Verdict`, one function.
Uses `with_structured_output` so the scores come back typed and range-validated
rather than parsed out of prose.

**`rag/evaluation/harness.py`** — `Case` (one question's result), `RunResult`
(one configuration's results with aggregation helpers), `run()` (thread pool),
`compare()` (aligned text table), `save()` (JSON report).

## Dependency direction

```
        config.py  ◄── everything (no module-level provider imports)
            ▲
   ┌────────┼────────┐
documents  chunking  store
                       ▲
                   retrieval
                       ▲
                   assistant
                       ▲
            ┌──────────┴──────────┐
        harness                 app.py
       (+ metrics, judge, dataset)
            ▲
        evaluate.py
```

Nothing imports upward. `metrics.py` imports `Chunk` from `retrieval` only for
the type annotation. The one deliberate quirk is `harness.py` calling
`retrieval.store.open_store()` — reaching through `retrieval` to the `store`
module it imported — to warm the embedding model before the thread pool starts.

## External dependencies and why each is there

| Package | Why |
|---|---|
| `langchain-core` | `Document` and the message types |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter` |
| `langchain-chroma` | Chroma vector store integration |
| `langchain-huggingface` | Local sentence-transformers embeddings |
| `langchain-openai` | OpenAI chat and embeddings |
| `langchain-ollama` | Local chat models |
| `chromadb` | The vector database itself |
| `sentence-transformers` | Embedding models and the `CrossEncoder` re-ranker |
| `gradio` | The UI |
| `pydantic` | `Question`, `Verdict`, `Ranking` — typed structured output |
| `python-dotenv` | Loads `.env` |
| `tqdm` | Progress bars during evaluation and contextual chunking |
