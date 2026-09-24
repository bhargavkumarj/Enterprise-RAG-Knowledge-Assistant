# Code Walkthrough — Indexing

Covers `rag/config.py`, `rag/documents.py`, `rag/chunking.py`, `rag/store.py` and
`ingest.py`.

---

## `rag/config.py`

### Module setup

```python
load_dotenv(override=True)
ROOT = Path(__file__).resolve().parent.parent
```

`override=True` means values in `.env` win over variables already exported in the
shell. That is the behaviour you want for a project file: the `.env` is the
project's declared configuration, and a stale exported variable in a long-lived
terminal should not silently change what runs. It also makes the observed
behaviour match what the file says, which matters when you are debugging.

`ROOT` resolves to the project directory: `config.py` is at
`<project>/rag/config.py`, so `.parent.parent` is `<project>`. `.resolve()` makes
it absolute and collapses symlinks, so every derived path works regardless of the
current working directory.

### `_flag`

```python
def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}
```

Environment variables are always strings. `bool("false")` is `True` in Python, so
a naive conversion silently inverts the meaning of `REWRITE_QUERY=false`. This
accepts the five spellings people actually type and treats everything else as
false. `str(default)` makes the fallback go through the same parsing path.

### The `Settings` dataclass

Grouped into six blocks:

**Paths.** `knowledge_base`, `questions`, `vector_store`, `collection`. All
derived from `ROOT`, so nothing is machine-specific.

**Embeddings.** `embedding_backend` picks `huggingface` (default, local, free) or
`openai`. The default matters: the entire indexing and retrieval evaluation runs
with no API key, so anyone can clone this and get real numbers.

**Chat.** `chat_backend` picks `openai` (default) or `ollama`. `judge_model` is
separate from `openai_chat_model` so the grader can be a different, typically
stronger, model than the one being graded.

**Chunking.** `chunk_strategy`, `chunk_size` (800 chars), `chunk_overlap` (150).

**Retrieval.** `top_k` (6, what reaches the prompt), `candidate_k` (20, what is
fetched before re-ranking), `rewrite_query`, `rerank`, `reranker`
(`llm` or `cross-encoder`), `cross_encoder_model`.

**Identity.** `organisation`, injected into every prompt.

### The private fields

```python
_embeddings: object | None = field(default=None, repr=False)
_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
```

`field(default_factory=...)` is required for the lock: a plain default would
share one `Lock` object across every `Settings` instance created, because
dataclass defaults are evaluated once at class definition. `repr=False` keeps
both out of the generated `__repr__`, so printing settings does not dump a model
object.

### `embeddings()`

```python
def embeddings(self):
    with self._lock:
        if self._embeddings is None:
            ...
        return self._embeddings
```

Three things happen here.

**Lazy import.** The provider package is imported inside the branch. Nothing
imports `langchain_openai` unless the OpenAI backend is actually selected, which
keeps startup fast and means a missing optional dependency does not break an
unrelated code path.

**Caching.** A sentence-transformers model takes several seconds to load and
hundreds of megabytes of memory. Building one per call would make retrieval
unusable.

**Locking.** This is the fix for a real crash. When `harness.run` fanned out to 4
threads and each called `embeddings()`, several tried to materialise the same
model onto the device simultaneously and PyTorch raised
`NotImplementedError: Cannot copy out of meta tensor; no data!`. The lock
serialises construction; after the first call every thread takes the fast path.

Returning inside the `with` block is intentional — it keeps the check and the
return atomic.

### `chat_model()` and `judge()`

```python
def chat_model(self, temperature: float = 0.0, model: str | None = None):
```

`temperature=0.0` by default. For a grounded assistant, sampling diversity is
not a feature: the same question against the same extracts should give the same
answer, and evaluation is meaningless if the system is a random variable.

The `model` override is what lets `judge()` reuse this factory with a different
model name, and what `contextual_chunks` could use to chunk with a cheap model
while answering with an expensive one.

Note that chat models are **not** cached. They are thin HTTP clients — cheap to
construct, and no shared mutable state to protect.

### `fingerprint()`

```python
return f"{self.embedding_backend}:{model}"
```

This identifies the **embedding space** and nothing else. It deliberately
excludes chunk strategy and size.

That distinction was a bug. The fingerprint originally included
`chunk_strategy:chunk_size`, and the store validated it on open. But the query
side has no idea what strategy built the store — it just reads `settings`, which
carries the default. So indexing with `--strategy markdown` and then querying
raised `RuntimeError: built with a different embedding/chunking setup` for a
store that was perfectly valid.

The principle: **validate what makes queries wrong, record the rest.** A
different embedding model makes queries meaningless — vectors from two models are
not comparable, and Chroma will happily return nearest neighbours in the wrong
space. A different chunk size just changes what the chunks look like. So the
fingerprint covers the model, and the manifest records strategy, size and overlap
for reporting.

---

## `rag/documents.py`

### `load_knowledge_base()`

```python
for path in sorted(root.rglob("*.md")):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        continue
```

`sorted()` makes ingestion deterministic. Without it, filesystem ordering varies
between machines, chunk ids shift, and two "identical" stores differ.

`rglob` recurses, so the folder structure can be nested arbitrarily. Empty files
are skipped — an empty chunk embeds to a near-meaningless vector that can still
be returned as a nearest neighbour.

`encoding="utf-8"` is explicit because the platform default differs on Windows
and these documents contain typographic punctuation.

The metadata:

| Key | Value | Used by |
|---|---|---|
| `source` | `path.stem`, e.g. `Contract with Apex Reinsurance for Rellm` | Citations in prompts, UI labels |
| `path` | relative path, e.g. `contracts/Apex.md` | `contextual_chunks` joins chunks to parents |
| `doc_type` | parent folder name: `company`, `products`, `contracts`, `employees` | Per-type reporting, potential filtering |

One `Document` per file, not per section. Chunking happens later, and
`contextual_chunks` needs the *whole* document to describe a chunk's place in it.

### `summarise()`

A plain counter producing `{'company': 4, 'contracts': 32, 'employees': 32,
'products': 8}`. Printed by `ingest.py` — if a folder is empty or a glob is
wrong, it shows up immediately instead of surfacing later as unexplained
retrieval misses.

---

## `rag/chunking.py`

### Why chunk at all

Two independent reasons. **Context limits** — you cannot paste 76 documents into
a prompt. **Retrieval precision** — a vector is a fixed-size summary of whatever
you embed. Embed a 4,000-word contract and you get a vector meaning "a contract,
generally". Embed a 500-character clause and you get a vector meaning that
clause. Smaller chunks are sharper but risk cutting an answer in half.

### Strategy 1 — `recursive_chunks`

```python
separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
```

`RecursiveCharacterTextSplitter` tries separators in order. It splits on the
first one that produces pieces under `chunk_size`, recursing into any piece still
too large. The custom list is ordered by semantic strength: markdown headings
first, then paragraphs, then lines, then sentences, then words. The default list
starts at `\n\n`, so passing headings explicitly keeps sections together when
they fit.

The 150-character overlap means consecutive chunks share their boundary text, so
a sentence split across the boundary still appears whole in one of them.

### Strategy 2 — `markdown_chunks` (used for the reported results)

```python
header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
```

Two passes. First split on `#`, `##`, `###` — semantic boundaries the author
wrote. Then size-limit each section, because a single section can still be long.

`strip_headers=False` is important. The heading text ("Termination", "Annual
Performance History") is often the most retrievable part of the section. Dropping
it and moving it to metadata would remove it from the embedded text.

```python
section.metadata = {**doc.metadata, **section.metadata}
```

Merge order matters: parent metadata first, section metadata second, so the
header keys (`h1`, `h2`, `h3`) are added without the splitter's fresh metadata
dict wiping out `source`, `path` and `doc_type`.

On this corpus: 76 documents → 659 chunks, median 532 characters.

### Strategy 3 — `contextual_chunks`

Implements Anthropic's contextual-retrieval idea. The problem it solves: a chunk
reading *"The agreement may be terminated with 30 days notice"* is unretrievable
by "how do I get out of the Apex contract?" — it never says Apex, or contract, or
which agreement.

```python
def _describe(args) -> Document:
    chunk, parent = args
    ...
    context = model.invoke(prompt).content.strip()
    return Document(page_content=f"{context}\n\n{chunk.page_content}", ...)
```

For each chunk, the LLM sees the parent document (first 6,000 chars) and the
chunk, and writes one or two sentences placing it. That prefix is prepended
*before embedding*, so the chunk's vector now carries the entity names.

```python
by_source = {doc.metadata["path"]: doc for doc in documents}
pairs = [(chunk, by_source[chunk.metadata["path"]]) for chunk in base]
```

This is what `path` exists for. It is the stable key that survives chunking and
lets each chunk find its parent.

`_describe` takes a single tuple argument because `ThreadPoolExecutor.map` passes
one item per call. Threads, not processes: the work is waiting on HTTP.

Cost: one LLM call per chunk. 659 chunks on this corpus. That is the trade — real
retrieval gains on reference-heavy documents, paid for once at index time.

### `chunk_documents()`

```python
if strategy not in STRATEGIES:
    raise ValueError(f"Unknown chunk strategy {strategy!r}; pick one of {list(STRATEGIES)}")
chunks = STRATEGIES[strategy](documents)
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = index
```

Dispatch through a dict, validate with a message that lists the valid options,
and stamp a sequential `chunk_id`. The id is assigned after strategy selection so
it reflects the final ordering regardless of which strategy ran.

`STRATEGIES` is also what `ingest.py` reads for its `--strategy` choices, so the
CLI cannot drift out of sync with the implementation.

---

## `rag/store.py`

### `build()`

```python
if reset and settings.vector_store.exists():
    shutil.rmtree(settings.vector_store)
```

Full wipe, not upsert. Re-ingesting is meant to be idempotent: run it twice, get
the same store. Chroma's `add` would append, so a second run with the same
documents would double every chunk and skew every similarity search.

```python
store = Chroma.from_documents(documents=chunks, embedding=settings.embeddings(), ...)
```

`from_documents` creates the collection, embeds every chunk in batches, and
persists. Chroma writes SQLite plus a binary index under `persist_directory` —
nothing else needs to run, which is why this project has no database service.

Then the manifest:

```json
{
  "fingerprint": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
  "chunks": 659,
  "chunk_strategy": "markdown",
  "chunk_size": 800,
  "chunk_overlap": 150
}
```

One field is a safety check, the rest are provenance.

### `open_store()`

```python
@lru_cache(maxsize=1)
def open_store() -> Chroma:
```

Cached because opening a Chroma client and attaching the embedding function is
not free and every retrieval call needs it.

Two guards:

```python
if not manifest_path.exists():
    raise FileNotFoundError("No vector store yet — run `python ingest.py` first.")
```

An error that tells you the fix. Without it you get an empty collection and
silently empty results, which look like a retrieval bug.

```python
if manifest.get("fingerprint") != settings.fingerprint():
    raise RuntimeError("The vector store was built with a different embedding model ...")
```

The critical one. Query vectors from model A compared against document vectors
from model B produce confident nonsense — Chroma returns nearest neighbours, and
"nearest" in a mismatched space is arbitrary. This fails loudly instead, naming
both fingerprints.

Note `lru_cache` is not itself thread-safe against concurrent first calls — two
threads can both miss and both construct. That is exactly why the lock lives in
`Settings.embeddings()`, where the expensive, unsafe construction actually
happens, and why `harness.run` warms the store before starting its pool.

### `stats()`

```python
manifest["vectors"] = store._collection.count()
```

Merges the recorded manifest with the live count. `_collection` is a private
LangChain attribute — the public wrapper has no count method. It is used in one
place, for a display string, and the UI wraps the whole call in `try/except`.

---

## `ingest.py`

```python
settings.chunk_strategy = args.strategy
settings.chunk_size = args.chunk_size
settings.chunk_overlap = args.chunk_overlap
```

CLI flags mutate the settings singleton, so the whole library sees them without
threading parameters through every function. `settings` is a plain dataclass
instance, not frozen, precisely to allow this.

```python
lengths = sorted(len(chunk.page_content) for chunk in chunks)
median = lengths[len(lengths) // 2]
```

The median, not the mean. Chunk length distributions are skewed by a few very
long or very short pieces; the median tells you what a typical chunk looks like.
If you ask for 800 and get a median of 130, your separators are firing too
aggressively and retrieval will be full of fragments — that is worth seeing at
index time rather than discovering later.

The final line prints the fingerprint, so the store's contract with the query
side is visible in the terminal.
