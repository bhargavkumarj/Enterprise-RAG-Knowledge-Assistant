# Design Decisions

Every non-obvious choice, the alternatives that were rejected, and the bugs found
while building.

---

## Decision 1 — Embeddings default to local, chat defaults to hosted

**Choice.** `EMBEDDING_BACKEND=huggingface`, `CHAT_BACKEND=openai`.

**Why the asymmetry.** They are different kinds of task. Embedding is a narrow,
well-solved problem where an 80MB model performs close to a hosted one, and you
need to run it 659 times at index time and twice per query — so local is both
adequate and much cheaper. Generation is where model quality actually shows, and
a small local model produces visibly worse grounded answers.

**The consequence that matters.** Indexing and the entire retrieval evaluation
run with **no API key**. Someone can clone this, run `ingest.py` and
`evaluate.py --ablation`, and get the real published numbers for free. An
evaluation that requires a credit card does not get reproduced.

---

## Decision 2 — The fingerprint covers the embedding model only

**Original implementation.** `fingerprint()` returned
`backend:model:chunk_strategy:chunk_size`, and `open_store()` rejected any
mismatch.

**What broke.** Indexing with `--strategy markdown` then querying raised:

```
RuntimeError: The vector store was built with a different embedding/chunking setup
(huggingface:...:markdown:800 vs huggingface:...:recursive:800)
```

The store was perfectly valid. The query side simply has no way to know which
strategy built it — it reads `settings`, which carries the default.

**The fix and the principle.** *Validate what makes queries wrong; record the
rest.* A different embedding model makes queries meaningless — vectors from two
models are not comparable and Chroma returns confident nonsense. A different
chunk strategy just changes what the chunks look like; queries still work.

So the fingerprint covers the model, and the manifest separately records
strategy, size and overlap for provenance and for the UI status line.

**Generalisation.** An over-strict invariant that fires on valid states is worse
than no invariant, because people work around it and then it protects nothing.

---

## Decision 3 — `Chunk` instead of LangChain's `Document`

**Choice.** A four-field dataclass, converted once in `Chunk.from_document`.

**Why.** `Document` stores everything in a `metadata` dict, so every consumer
writes `doc.metadata.get("doc_type", "unknown")` — a typo produces `"unknown"`
silently rather than an error. Flattening the three fields that matter into
required attributes means the UI, the metrics and the prompt builder all read
`chunk.doc_type`, and a typo is an `AttributeError`.

It also decouples the rest of the system from LangChain. `metrics.py`,
`assistant.py` and `app.py` never import LangChain types — swapping the vector
store would touch `store.py` and `retrieval.vector_search` only.

**Cost.** One conversion step and a second type to understand. Worth it at this
size; more so as consumers multiply.

---

## Decision 4 — Return a trace, not a list

**Choice.** `retrieve()` returns `RetrievalTrace(query, rewritten, candidates,
chunks)`.

**Why.** When an answer is wrong, "which stage failed?" is the first question.
The trace answers it: a mangled `rewritten` is a rewriting problem, a low
`candidates` count is a search problem, good candidates with bad `chunks` is a
re-ranking problem.

This is also what lets the UI display the search query that was actually used,
which is genuinely useful — users can see that "who won it" became "IIOTY award
winner 2023" and understand why they got what they got.

---

## Decision 5 — `is None` rather than `or` for the option defaults

```python
use_rewrite = settings.rewrite_query if use_rewrite is None else use_rewrite
```

**The bug this avoids.** `use_rewrite or settings.rewrite_query` evaluates
`False or True` → `True`. An explicit `use_rewrite=False` would be silently
overridden by the setting.

**Why it is load-bearing here.** The ablation passes explicit `False` values for
three of its four configurations. With the sloppy version, all four rows would
have run with rewriting on, every row would show nearly identical numbers, and
the experiment would be meaningless — while looking completely fine.

This class of bug is dangerous precisely because it produces plausible output.

---

## Decision 6 — Always search the original query

**Choice.** Even with rewriting on, `retrieve()` searches the original question
and fuses both result lists.

**Why.** Rewriting is lossy. It can drop the proper noun that was the highest
signal in the query. Searching only the rewrite makes the system strictly worse
whenever the rewrite is bad, with no recovery path.

Fusing both means a bad rewrite costs one extra search and RRF still surfaces
what the original found. Combined with the prompt's "keep every proper noun"
instruction, this bounds the downside of a feature that measurably has one.

---

## Decision 7 — RRF rather than score averaging

**Rejected.** Average the Chroma relevance scores from both searches.

**Why rejected.** The scores come from different query vectors, so their
distributions differ. Averaging compares numbers that are not on the same scale.

**Chosen.** Reciprocal rank fusion, which uses ranks — comparable by
construction — with K=60 damping so consensus across lists beats a single list's
favourite. Also the standard method for hybrid dense/sparse search, so extending
to BM25 later needs no change to `merge()`, which is already variadic.

---

## Decision 8 — Two re-rankers, cross-encoder for the reported results

**Cross-encoder** (`ms-marco-MiniLM-L-6-v2`): local, free, one batched forward
pass, ~50ms for 30 candidates. No API dependency.

**LLM**: can follow instructions a cross-encoder cannot ("prefer the most recent
contract"), but costs a call per question and adds latency.

The published ablation used the cross-encoder — it makes the numbers reproducible
without an API key, which is the same argument as Decision 1.

**The LLM re-ranker's three validation guards** exist because models returning
index lists get them wrong predictably: out-of-range indices (`IndexError`),
repeated indices (a wasted context slot), and occasionally nothing usable
(fall back to fused order). All three were written defensively from the start
rather than after a crash, because the failure modes are well known.

---

## Decision 9 — Broad `except` in the LLM re-ranker

```python
try:
    ranking = model.invoke(...)
except Exception:
    return chunks[:keep]
```

**Normally a smell.** Justified here by what the function *is*: an optional
refinement on top of an already-working ranking. The fused order is a reasonable
answer. Failing the entire question because a ranking refinement failed trades a
good outcome for no outcome.

**Where this would be wrong.** Anywhere the fallback is not already correct —
`open_store()` deliberately raises rather than returning an empty store, because
there is no sensible fallback for "no index exists".

---

## Decision 10 — Locks around model construction

**The bug, twice.** Both the embedding model and the cross-encoder crashed with:

```
NotImplementedError: Cannot copy out of meta tensor; no data!
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

when four evaluation threads tried to construct them simultaneously.

**Why `lru_cache` was not enough.** `lru_cache` guarantees *cache correctness*,
not *single construction*. Four threads can all miss, all construct, and one
wins. For a pure function that is wasted work; for a PyTorch model being
materialised onto a device it is a crash.

**The fix.** An explicit `threading.Lock` around the check-and-construct in both
places, plus a warm-up call in `harness.run()` before the pool starts so the
model is loaded once, on the main thread, before any worker needs it.

**The lesson.** "Cached" and "constructed once" are different guarantees. If
construction has side effects — device allocation, file locks, network setup —
you need the second one, and only an explicit lock gives it to you.

---

## Decision 11 — Quote-stripping the rewritten query

**Observed.** The first local rewrite returned, verbatim:

```
"Insurellm knowledge base: IIOTY award winner 2023"
```

including the quotation marks, which the model added because the prompt asked for
"the search query only" and it interpreted that as "quote the query".

**Why it matters.** Those quote characters tokenise and shift the embedding.
Small, but free to fix and pure downside otherwise.

**Fix.** `query.strip('"').strip() or question` — strip quotes, strip whitespace
revealed underneath, and fall back to the original if the result is empty.

**The general point.** LLM output needs normalising at the boundary. Every place
a model's text becomes a machine input — a search query, an index, a number — is
a place to validate.

---

## Decision 12 — Five metrics, not one composite

**Rejected.** A single "retrieval score".

**Why rejected.** In the measured ablation, query rewriting moved hit rate from
0.983 to 1.000 and MRR from 0.941 to 0.914 — in opposite directions. Any weighted
composite would have collapsed that into a small number and hidden the actual
finding, which is that rewriting rescues total misses while blunting precise
queries.

**The principle.** Composite metrics are for dashboards where someone needs one
number. Engineering decisions need the components, because the components are
where the mechanism is visible.

---

## Decision 13 — Keyword-proxy relevance

**Rejected.** Human relevance labels — the gold standard.

**Why rejected.** 659 chunks × 150 questions ≈ 98,000 judgements, and they would
need redoing whenever the chunking changed.

**Chosen.** A chunk is relevant if it contains a gold keyword.

**Honest about the limits.** False positives (wrong "Thompson") and false
negatives (a paraphrase without the keyword) both exist, so the *absolute* values
are approximate.

**Why it is still valid.** The proxy is applied identically to every
configuration. Systematic error cancels when comparing configurations, which is
the only thing these numbers are used for. An approximate but unbiased comparison
beats an exact measurement nobody can afford to produce.

---

## Decision 14 — Retrieval metrics computed from the answering retrieval

**Choice.** In `_evaluate_one`, the `--answers` branch scores retrieval using
`answer.chunks` — the chunks that actually went into the prompt — rather than
making a separate `retrieve()` call.

**Why.** Two independent retrievals would produce two different chunk sets, so
the retrieval score and the judge score would describe different runs and could
not be correlated. As written, a case with high retrieval metrics and a low
accuracy verdict is unambiguous: retrieval worked, generation did not.

---

## Decision 15 — A 2×2 factorial ablation

**Rejected.** Baseline versus everything-on.

**Why rejected.** That comparison shows a net improvement and tells you nothing
about which feature caused it — or that one of them is hurting.

**Chosen.** Four rows: neither, rewriting only, re-ranking only, both. That is
what revealed re-ranking as the clear win and rewriting as a mixed bag, which in
turn produced an actionable recommendation: ship re-ranking now, revisit
rewriting with a stronger model.

Every row runs on the same questions, in the same process, against the same
store, so the only difference is the retrieval options.

---

## Decision 16 — Full wipe on re-ingest

**Choice.** `build()` does `shutil.rmtree` before rebuilding.

**Why.** Chroma's `add` appends. Re-running `ingest.py` without a wipe would
double every chunk, and duplicate chunks would occupy multiple slots in the top-k
— quietly degrading precision in a way that looks like a model problem.

**Cost.** No incremental indexing. At 659 chunks a full rebuild takes seconds. At
a million it would need a real strategy (content hashing, upserts by id), which
is a fair thing to name in an interview as the first thing to change at scale.

---

## Decision 17 — Context in the system message

**Choice.** Extracts go in the `SystemMessage` with the rules, not in the user
turn.

**Why.** Position signals authority. Content in the system message reads as
configuration the model was given; content in a user turn reads as something the
user asserted — and user assertions are exactly what a model should be sceptical
of. It also keeps the extracts pinned at the front as the conversation grows,
rather than drifting into the middle of a long history.

---

## Decision 18 — Streaming yields chunks on every iteration

**Choice.** `stream()` yields `(partial_text, chunks)` on every token, with the
same chunks object each time.

**Why.** Retrieval completes before the first token is generated. Yielding the
chunks immediately lets the UI paint the evidence panel while the answer is still
being written, so the user sees *what the system found* before they see what it
concluded — which is the right order for a tool whose selling point is
verifiability.

---

## Summary of bugs found by running the code

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| Over-strict fingerprint | Valid store rejected at query time | Fingerprint included chunk strategy, unknowable from the query side | Fingerprint covers the embedding model only |
| Embedding model race | `Cannot copy out of meta tensor` under 4 threads | Concurrent construction of a PyTorch model | `threading.Lock` in `Settings.embeddings()` |
| Cross-encoder race | Same crash in the ablation run | `lru_cache` does not prevent concurrent construction | Explicit lock + module-level singleton |
| Quoted rewrite | Search query wrapped in `"` | Model quoting its output | `.strip('"')` with an empty-result fallback |
| Corrupt HF cache | `Can't load the model for cross-encoder/...` | Partially downloaded cache entry | Cleared the cache entry; model then loaded and scored correctly |

None of these would have been found by reading the code. They were found by
running it on the real corpus, which is the argument for the evaluation harness
existing at all.
