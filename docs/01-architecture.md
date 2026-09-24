# Architecture

## The problem RAG solves here

The knowledge base is private. No language model has ever seen these contracts or
employee records, and fine-tuning one on 76 documents would be expensive, slow to
update, and would still produce confident guesses about clauses that do not
exist. RAG sidesteps all of that: keep the documents in a searchable index, fetch
the relevant ones at question time, and let the model read them.

That turns "what does the model know?" into "can we find the right paragraph?",
which is a measurable engineering problem — and measuring it is most of what this
project is about.

## Two pipelines

The system has two distinct pipelines that touch the same vector store but run at
different times and for different reasons.

### Pipeline 1 — indexing (offline, run once per configuration change)

```
data/knowledge_base/**/*.md
        │
        ▼
 load_knowledge_base()        76 Documents, each tagged {source, path, doc_type}
        │
        ▼
 chunk_documents(strategy)    659 chunks (markdown strategy, median 532 chars)
        │                     strategies: recursive | markdown | contextual
        ▼
 settings.embeddings()        each chunk -> a 384-dim vector
        │
        ▼
 store.build()                Chroma collection + manifest.json
```

The manifest records the embedding fingerprint so the query side can refuse to
run against a store built with a different model.

### Pipeline 2 — answering (online, per question)

```
question ─────────────────────────────────────────────┐
   │                                                  │
   ▼  (optional)                                      │
rewrite()          "who won it in 2023?"              │
   │               + history                          │
   │               -> "Insurellm IIOTY award 2023"    │
   ▼                                                  ▼
vector_search(rewritten, 20)          vector_search(original, 20)
   │                                                  │
   └──────────────────► merge() ◄─────────────────────┘
                      reciprocal rank fusion
                      de-duplicated candidate list
                            │
                            ▼  (optional)
                      rerank_with_cross_encoder()  or  rerank_with_llm()
                      rescore every (question, chunk) pair
                            │
                            ▼
                      top k = 6 chunks
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     build_context()               RetrievalTrace
     system prompt with            (returned to the UI so the
     extracts + citation rule       user can see the evidence)
              │
              ▼
     chat model .invoke() or .stream()
              │
              ▼
          Answer(text, chunks, rewritten)
```

### Pipeline 3 — evaluation (offline, on demand)

Evaluation reuses pipeline 2 exactly. It does not reimplement retrieval, which
matters: a harness that runs different code to the product measures the wrong
thing.

```
data/questions.jsonl  (150 questions)
        │
        ▼
 split_questions(holdout=0.4, seed=42, stratified by category)
        │
        ├──► dev (90)      used while tuning
        └──► holdout (60)  used for every reported number
                 │
                 ▼
        harness.run(questions, **options)   4 threads
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
 retrieval.retrieve()   assistant.ask()   (only with --answers)
       │                    │
       ▼                    ▼
 retrieval_metrics()    judge()  -> Verdict(accuracy, completeness, groundedness)
       │                    │
       └────────┬───────────┘
                ▼
          RunResult (per-case + aggregate)
                │
       ┌────────┴────────┐
       ▼                 ▼
   compare() table   save() JSON in reports/
```

## Lifecycle of one question, in full detail

Take `"Who won the IIOTY award in 2023?"` with rewriting and cross-encoder
re-ranking both on.

**1. Entry.** `app.respond()` appends the message to the chat history and calls
`assistant.stream(message, history=history[:-1])`. The slice drops the message
just appended so it is not duplicated — it goes in separately as the final
`HumanMessage`.

**2. Retrieval begins.** `assistant.stream` immediately calls
`retrieval.retrieve(question, history=history)`. Nothing is sent to the chat
model yet.

**3. Rewriting.** `settings.rewrite_query` is true, so `rewrite()` builds a prompt
containing the last 6 turns of conversation and the question, and asks the chat
model for a search query. It returns something like
`Insurellm IIOTY award winner 2023`. The result is stripped of surrounding quotes
— models like to wrap a query in them, and a leading `"` becomes a token that
shifts the embedding for no reason.

**4. Candidate retrieval.** Because re-ranking is on, `depth` is
`settings.candidate_k` = 20, not `k` = 6. Two searches run: one on the original
question, one on the rewrite. Each returns 20 `(Document, score)` pairs, which
become `Chunk` objects. If the rewrite is identical to the question
(case-insensitively) the second search is skipped.

**5. Fusion.** `merge()` combines the two lists with reciprocal rank fusion. A
chunk's fused score is the sum over lists of `1 / (60 + rank)`. A chunk appearing
at rank 3 in both lists scores `2/63 = 0.0317`; a chunk at rank 1 in only one
scores `1/61 = 0.0164`. Agreement across queries beats a single strong hit. The
dictionary is keyed on chunk text, which de-duplicates the overlap between the
two lists for free.

**6. Re-ranking.** The ~25–35 fused candidates go to
`rerank_with_cross_encoder()`. The cross-encoder scores every
`(question, chunk_text)` pair jointly — it reads both at once rather than
comparing two independently-computed vectors — and the chunks are sorted by that
score. The top 6 survive. Each chunk's `.score` is overwritten with the
cross-encoder logit, which is what the UI displays.

**7. Context assembly.** `build_context()` renders the 6 chunks as
`[source] (doc_type)\ntext`, separated by blank lines. That label is what the
model cites.

**8. Prompt assembly.** A `SystemMessage` carrying the instructions and the
extracts, then the prior conversation as alternating `HumanMessage`/`AIMessage`,
then the current question as a `HumanMessage`. Context goes in the system message
so the conversation history cannot push it out of position or be mistaken for
user input.

**9. Generation.** `.stream()` yields tokens. After each one, `stream()` yields
`(partial_text, chunks)`. The chunks are the same object every time — the UI can
render the evidence panel on the first token, before the answer exists.

**10. Display.** `context_panel()` renders each chunk as a markdown blockquote
with its source, type and score.

## Where the extension points are

| You want to | Change |
|---|---|
| Use a different embedding model | `EMBEDDING_BACKEND` / `HF_EMBEDDING_MODEL`, then re-run `ingest.py` |
| Add a chunking strategy | Add a function to `chunking.STRATEGIES` |
| Swap the re-ranker | `RERANKER=llm` or `cross-encoder`, or add a branch in `retrieve()` |
| Run entirely locally | `CHAT_BACKEND=ollama` (embeddings are already local) |
| Point at a different corpus | Replace `data/knowledge_base/`, set `ORGANISATION` |
| Add a retrieval metric | Add a function to `metrics.py` and a key to `retrieval_metrics()` |

## Threading model

Three places use concurrency, and each had to be made safe:

- **`chunking.contextual_chunks`** — 8 threads, one LLM call per chunk. Network
  bound, so threads are the right tool.
- **`harness.run`** — 4 threads, one question each. Also network bound.
- **`config.Settings.embeddings`** — *not* concurrent, but called from threads. A
  `threading.Lock` guards a single shared instance.

The lock is not decoration. Loading `sentence-transformers` from several threads
simultaneously raised `NotImplementedError: Cannot copy out of meta tensor` on a
real run, because each thread tried to materialise the same model onto the device
at once. The same failure hit the cross-encoder and is guarded the same way in
`retrieval.cross_encoder()`. `harness.run` additionally calls `open_store()` once
before the pool starts, so the model is warm before any thread needs it.
