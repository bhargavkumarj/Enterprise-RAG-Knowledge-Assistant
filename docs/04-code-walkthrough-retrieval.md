# Code Walkthrough — Retrieval and Answering

Covers `rag/retrieval.py`, `rag/assistant.py` and `app.py`.

---

## `rag/retrieval.py`

The module with the most engineering in it. Everything between "a user typed
something" and "here are six extracts" lives here.

### `Chunk`

```python
@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str
    score: float = 0.0
```

A deliberately plain result type. LangChain's `Document` has a nested
`metadata` dict, so every consumer would write
`doc.metadata.get("doc_type", "unknown")`. `Chunk` flattens the three fields that
matter and makes them required, so the UI, the metrics and the prompt builder all
read `chunk.doc_type` and cannot typo a metadata key.

`from_document` is the single conversion point, and the only place the `.get(...,
"unknown")` defaults appear.

`score` is mutable and means different things at different stages: a Chroma
relevance score after `vector_search`, a cross-encoder logit after re-ranking.
That is intentional — it always means "how good this chunk is, according to
whatever last ranked it", which is what the UI should display.

### `RetrievalTrace`

```python
@dataclass
class RetrievalTrace:
    query: str
    rewritten: str | None
    candidates: int
    chunks: list[Chunk]
```

Returning a trace rather than a bare list is a debuggability decision. When an
answer is wrong you need to know *which* stage failed: did the rewrite mangle the
question (`rewritten`), did the search return too few candidates
(`candidates`), or did the re-ranker pick badly (`chunks`)? A bare list answers
none of those.

### `cross_encoder()`

```python
_cross_encoder_lock = threading.Lock()
_cross_encoder_model = None

def cross_encoder():
    global _cross_encoder_model
    with _cross_encoder_lock:
        if _cross_encoder_model is None:
            from sentence_transformers import CrossEncoder
            _cross_encoder_model = CrossEncoder(settings.cross_encoder_model)
        return _cross_encoder_model
```

This was originally `@lru_cache(maxsize=1)` and it crashed in the ablation run
with the same meta-tensor error as the embeddings. `lru_cache` guarantees *cache
correctness*, not *single construction*: four threads can all miss, all
construct, and one wins. For a pure function that is wasteful; for a PyTorch
model being moved onto an accelerator it is a crash.

The explicit lock guarantees exactly one construction. Same fix as
`Settings.embeddings()`, same reason.

### `rewrite()`

```python
turns = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-6:]) or "(none)"
```

Last 6 turns only — enough to resolve "he", "that contract", "the same year",
without spending the whole prompt on history. The `or "(none)"` handles the empty
case so the model sees an explicit marker instead of a blank region, which models
sometimes fill with invented context.

```python
query = settings.chat_model(temperature=0.0).invoke(prompt).content.strip()
return query.strip('"').strip() or question
```

Two guards, both from observed behaviour:

**Quote stripping.** The first local run returned `"Insurellm knowledge base:
IIOTY award winner 2023"` — with the quotes. Those quote characters become
tokens and shift the embedding for no semantic reason. `.strip('"')` removes
them, and the second `.strip()` cleans up whitespace revealed underneath.

**Fallback.** `or question` means an empty rewrite falls back to the original.
An empty query would otherwise embed to a meaningless vector and return
effectively random chunks.

The prompt tells the model to *keep every proper noun*. The failure mode of query
rewriting is generalisation — turning "What does the Apex Reinsurance contract
say about termination?" into "contract termination terms", which matches all 32
contracts equally. Proper nouns are the highest-signal tokens in this corpus.

### `vector_search()`

```python
hits = store.open_store().similarity_search_with_relevance_scores(query, k=k)
return [Chunk.from_document(doc, score) for doc, score in hits]
```

`_with_relevance_scores` rather than plain `similarity_search` because the UI
displays scores and the fusion step benefits from having them available. Chroma
normalises these to roughly 0–1, higher is better — unlike raw distance, where
lower is better, which is a classic sign-flip bug.

### `merge()` — reciprocal rank fusion

```python
for group in groups:
    for rank, chunk in enumerate(group, start=1):
        scores[chunk.text] = scores.get(chunk.text, 0.0) + 1.0 / (60 + rank)
        keep.setdefault(chunk.text, chunk)
```

RRF combines ranked lists using **ranks, not scores**. That is the whole point:
the original query and the rewritten query produce scores on different scales
(different query vectors, different distance distributions), so averaging the
scores compares incomparable numbers. Ranks are comparable by construction.

The constant 60 is the standard value from the original RRF paper. It damps the
difference between top ranks: `1/61 = 0.0164` versus `1/62 = 0.0161` is a 2%
gap, whereas without the constant `1/1` versus `1/2` is a 50% gap. Damping means
a chunk must appear well across *several* lists to beat a chunk that appears
first in one.

Worked example — chunk A at rank 3 in both lists, chunk B at rank 1 in one:

```
A: 1/63 + 1/63 = 0.03175
B: 1/61         = 0.01639
```

A wins. Two queries agreeing on a moderately good chunk is stronger evidence than
one query loving a chunk.

`keep.setdefault(chunk.text, chunk)` keeps the first `Chunk` object seen for each
text, so the surviving object carries the score from the list it ranked highest
in. Keying on the text also de-duplicates the overlap between the two lists for
free — the same chunk retrieved by both queries appears once.

The function is variadic (`*groups`), so adding a third query source — an HyDE
expansion, a keyword search, a metadata-filtered search — needs no change here.

### `rerank_with_llm()`

```python
extracts = "\n\n".join(
    f"[{i}] ({chunk.doc_type}/{chunk.source})\n{chunk.text[:1200]}"
    for i, chunk in enumerate(chunks)
)
model = settings.chat_model(temperature=0.0).with_structured_output(Ranking)
```

Chunks are numbered and truncated to 1,200 characters each. Truncation bounds the
prompt: 30 candidates × 1,200 chars ≈ 36,000 characters. The leading part of a
chunk is usually enough to judge relevance.

`with_structured_output(Ranking)` binds the pydantic schema so the reply is a
validated `Ranking` object, not prose to parse. `Ranking.ids` is
`list[int]` — the model returns indices, not text, which is far cheaper and
cannot corrupt the chunk content.

```python
try:
    ranking = model.invoke(...)
except Exception:
    return chunks[:keep]
```

Broad catch, deliberately. If the model refuses, times out, or emits invalid
JSON, the fused order is already a reasonable ranking. Degrading to it is better
than failing the whole question. This is a *ranking refinement* — an optional
improvement over a working baseline — so it is exactly where a broad catch is
justified.

```python
seen, ordered = set(), []
for index in ranking.ids:
    if 0 <= index < len(chunks) and index not in seen:
        seen.add(index)
        ordered.append(chunks[index])
return (ordered or chunks)[:keep]
```

Three validations, because LLMs returning index lists get them wrong in
predictable ways:

1. **Bounds check** — a hallucinated index like 47 when there are 30 candidates
   would raise `IndexError` on a raw lookup.
2. **Duplicate check** — models repeat indices; without `seen`, the same chunk
   appears twice in the context, wasting a slot.
3. **Empty fallback** — if every id was invalid, `ordered` is empty and `or
   chunks` falls back to the fused order.

### `rerank_with_cross_encoder()`

```python
scores = cross_encoder().predict([(question, chunk.text) for chunk in chunks])
for chunk, score in zip(chunks, scores):
    chunk.score = float(score)
return sorted(chunks, key=lambda c: c.score, reverse=True)[:keep]
```

The conceptual difference from vector search, which is the single most important
idea in this module:

- **Bi-encoder** (the embedding model): encodes the question and each chunk
  *separately* into vectors, then compares with cosine similarity. Chunks can be
  encoded once at index time, which is what makes search over millions of
  documents feasible. The cost is that the two encodings never see each other.
- **Cross-encoder**: puts the question and one chunk into the *same* forward pass
  and outputs a single relevance score. The transformer attends across both, so
  it can tell that "2023" in the question matches "2023" in the chunk in a way
  two independent vectors cannot represent.

Cross-encoders are far more accurate and far more expensive — one forward pass
per pair, nothing precomputable. Hence the two-stage pattern: a cheap bi-encoder
narrows millions to 20–30, an expensive cross-encoder orders those precisely.
This is why re-ranking produced the biggest measured gain (MRR 0.941 → 0.975).

`predict` takes the whole batch, so this is one batched forward pass. `float()`
converts numpy float32 to a Python float so the dataclass field and JSON
serialisation behave.

Note the scores are raw logits — the measured example gave `+6.50` for a relevant
pair and `-11.09` for an irrelevant one. They are not probabilities and are not
comparable to the Chroma relevance scores they overwrite, which is fine because
they are only ever used for ordering and display.

### `retrieve()` — the orchestrator

```python
k = k or settings.top_k
use_rewrite = settings.rewrite_query if use_rewrite is None else use_rewrite
use_rerank = settings.rerank if use_rerank is None else use_rerank
```

The `is None` checks are load-bearing. Writing `use_rewrite or settings.rewrite_query`
would make an explicit `use_rewrite=False` fall through to the setting, because
`False or True` is `True`. The ablation harness passes explicit `False` values —
with the sloppy version, three of the four ablation rows would silently be the
same configuration and the whole experiment would be invalid.

```python
depth = settings.candidate_k if use_rerank else k
```

Fetch 20 when re-ranking, 6 when not. Without re-ranking, extra candidates would
just be truncated away — pure waste. With it, they are the pool the re-ranker
reorders. This is also what makes the ablation fair: each configuration does the
minimum work it needs.

```python
groups = [vector_search(question, depth)]
if rewritten and rewritten.lower() != question.lower():
    groups.append(vector_search(rewritten, depth))
```

The original query is *always* searched, even when rewriting is on. If the
rewrite drops something important, the original still contributes candidates and
RRF can recover them. The equality check avoids paying for an identical second
search.

```python
if use_rerank and len(candidates) > k:
```

Skip re-ranking when there is nothing to re-rank — with 6 or fewer candidates for
`k=6`, every candidate is going into the prompt regardless of order. Saves a call.

---

## `rag/assistant.py`

### The system prompt

```
Answer only from the extracts below. ... If the extracts do not contain the
answer, say you do not have that information — never guess a name, date, figure
or contract term. Cite the extracts you used as [source] ...
```

Four instructions, each targeting a specific failure:

1. **"Answer only from the extracts"** — blocks the model answering from
   parametric memory. For a fictional company any such answer is fabricated.
2. **"say you do not have that information"** — gives an explicit escape hatch.
   Without one, models invent something, because producing text is the default
   behaviour.
3. **"never guess a name, date, figure or contract term"** — names the categories
   that actually matter. Generic "do not hallucinate" is weaker than naming the
   fields where a hallucination does damage.
4. **"Cite ... as [source]"** — makes the answer checkable, and the citation
   format matches the `[source]` labels `build_context` produces.

The length instruction ("two or three sentences unless asked for detail") exists
because verbose answers score *worse* on the judge's completeness/groundedness
axes — padding introduces claims the extracts do not support.

### `Answer.sources`

```python
seen = []
for chunk in self.chunks:
    label = f"{chunk.doc_type}/{chunk.source}"
    if label not in seen:
        seen.append(label)
return seen
```

A list, not a `set`, because **order is information**. The chunks arrive ranked,
so the first source is the most relevant one. `set` would discard that. The
`in` check on a list is O(n), which is irrelevant for 6 items and keeps the code
obvious.

### `build_context()`

```python
f"[{chunk.source}] ({chunk.doc_type})\n{chunk.text}"
```

Each extract is labelled with the name the model is asked to cite. The
`[source]` bracket form is deliberately the same shape as the citation
instruction, so the model copies rather than invents a format.

### `to_messages()`

```python
role, content = turn.get("role"), turn.get("content", "")
if role == "user":
    messages.append(HumanMessage(content=content))
elif role == "assistant":
    messages.append(AIMessage(content=content))
```

Converts Gradio's `{"role", "content"}` dicts into LangChain message objects.
Anything that is neither user nor assistant — a tool message, a stray system
entry — is silently dropped, so a history containing an unexpected role cannot
inject instructions into the prompt.

### `ask()` and `stream()`

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT.format(...)),
    *to_messages(history),
    HumanMessage(content=question),
]
```

Order is the design:

- **System message first**, carrying both the rules and the extracts. Putting
  context in the system message rather than the user turn keeps it outside the
  conversation, so the model treats it as ground truth rather than as something
  the user said.
- **History next**, so pronouns and follow-ups resolve.
- **Question last**, the most recent instruction.

`ask` and `stream` are near-identical by design. `stream` differs only in
yielding `(partial, chunks)` per token instead of returning once. The chunks are
yielded on *every* iteration including the first, so the UI can paint the
evidence panel before a single answer token has been generated — the retrieval is
already done by then.

```python
partial += piece.content or ""
```

The `or ""` handles chunks whose `content` is `None`, which some providers emit
for metadata-only frames.

`**retrieval_kwargs` passes through to `retrieve()`, which is how the evaluation
harness runs `ask()` under four different retrieval configurations without the
assistant knowing anything about ablations.

---

## `app.py`

### `context_panel()`

```python
body = chunk.text.strip()
if len(body) > 900:
    body = body[:900] + " …"
lines.append(
    f"**{position}. {chunk.source}** · `{chunk.doc_type}` · score {chunk.score:.3f}\n\n"
    f"> {body.replace(chr(10), chr(10) + '> ')}"
)
```

Each chunk renders as a markdown blockquote with rank, source, type and score.

`chr(10)` instead of a literal `\n` because the expression is inside an f-string,
and backslashes were not permitted in f-string expressions before Python 3.12.
`chr(10)` works on every version.

The replace prefixes *every* line with `> `, since markdown blockquotes need the
marker per line — without it, a multi-line chunk breaks out of the quote after
the first line.

Truncation at 900 characters keeps the panel scannable. The full text is still in
the prompt; this is a display limit only.

### `respond()`

```python
history = history + [{"role": "user", "content": message}]
for partial, retrieved in assistant.stream(message, history=history[:-1]):
    yield history + [{"role": "assistant", "content": partial}], context_panel(...)
```

`history + [...]` builds a new list rather than mutating the one Gradio holds in
state. Mutating shared UI state is how you get a chat that duplicates messages on
re-render.

`history[:-1]` drops the message just appended — `assistant.stream` adds the
question itself as the final `HumanMessage`, so passing it in the history too
would duplicate it.

The generator yields a *new* assistant message each time rather than editing one,
which is what Gradio's `type="messages"` chatbot expects for streaming.

### `build()`

```python
try:
    info = store.stats()
    status = f"{info['vectors']:,} chunks · {info['chunk_strategy']} chunking · ..."
except Exception as error:
    status = f"⚠️ {error}"
```

The status line is live truth, read from the manifest: how many vectors, which
strategy built them, which embeddings, what `k` is, whether re-ranking is on. Not
a hardcoded description that drifts.

The `try/except` is why running `app.py` before `ingest.py` shows a warning
banner instead of a stack trace — and the message is `store.open_store()`'s
"run `python ingest.py` first", which tells the user exactly what to do.

```python
box.submit(respond, [box, chatbot], [chatbot, context]).then(lambda: "", None, box)
```

`.then()` chains after the generator finishes and clears the input box. Chaining
rather than clearing inside `respond` means the box empties only on success — if
generation fails, the typed question is still there.
