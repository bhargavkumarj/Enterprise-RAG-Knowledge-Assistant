# Concepts Explained

The theory behind the project, connected back to the code that implements it.

---

## 1. Retrieval-Augmented Generation

### The problem

A language model's knowledge is frozen in its weights at training time. For a
private corpus it has three failure modes:

1. **It never saw the data.** Nothing about these contracts is in any model.
2. **It cannot be corrected.** A contract amended today cannot reach the weights.
3. **It does not know that it does not know.** Asked about a company it has never
   heard of, a model produces plausible text anyway.

### The three options

| Approach | How | Cost | Freshness | Traceability |
|---|---|---|---|---|
| Fine-tuning | Retrain on the corpus | High, repeated per update | Stale immediately | None |
| Long context | Paste everything in | Very high per query | Live | Poor |
| **RAG** | Index, retrieve, inject | Low | Live | Per-chunk citations |

RAG wins here on all three axes that matter. Update a document, re-run
`ingest.py`, done. And because the extracts are known, the answer is checkable —
which is exactly what the UI's context panel exists for.

### The reframing

RAG converts "does the model know this?" into "can we find the right paragraph?"
The second question has decades of information-retrieval research behind it and,
more importantly, is **measurable**. That is why this project has an evaluation
package at all.

---

## 2. Embeddings

### What they are

An embedding model maps text to a fixed-length vector of floats.
`all-MiniLM-L6-v2` produces 384 dimensions. The model is trained so that texts
with similar meaning land near each other.

The key property is that it is **not** keyword matching. "How do I cancel?" and
"termination provisions" share no words but embed close together, because the
model was trained on pairs of semantically related text.

### Similarity

Cosine similarity — the cosine of the angle between two vectors:

```
cos(a, b) = (a · b) / (|a| × |b|)
```

Ranges from -1 to 1; 1 means the same direction. Angle rather than distance
means the *magnitude* of the vectors is ignored, so a long chunk and a short
chunk about the same topic still score as similar.

Chroma exposes `similarity_search_with_relevance_scores`, which normalises to
roughly 0–1 with higher meaning better. Raw distance is the opposite convention
— a classic source of inverted-ranking bugs, avoided by using the relevance form.

### Why the embedding model is a hard contract

A vector only means something relative to the model that produced it. Dimension
142 of MiniLM and dimension 142 of `text-embedding-3-small` are unrelated
numbers. Comparing them produces nearest neighbours that are nearest in a
meaningless sense — and nothing errors, because the arithmetic is valid.

That is what `Settings.fingerprint()` and the manifest check in `store.py`
prevent. It is the single most consequential invariant in the system.

### The model used here

`sentence-transformers/all-MiniLM-L6-v2`: 6 transformer layers, 384 dimensions,
~80MB, runs on CPU in milliseconds. Chosen because it is free, local, fast and
good enough that the measured hit rate is 0.983 before any re-ranking. Bigger
models (`all-mpnet-base-v2`, OpenAI's `text-embedding-3-large`) do better on
hard queries; switching is one environment variable and a re-ingest.

---

## 3. Vector databases and ANN search

### What Chroma does

Stores vectors alongside their text and metadata, and answers "give me the k
nearest vectors to this one".

Exact nearest-neighbour search is O(n) per query — compare against every stored
vector. At 659 chunks that is instant. At 10 million it is not, so vector
databases use **approximate** nearest neighbour indexes (HNSW is the common one:
a navigable small-world graph you greedily descend). ANN trades a small chance of
missing a true neighbour for orders-of-magnitude speedup.

At this corpus size the approximation is irrelevant. It matters the moment the
corpus grows, and the code does not change — which is the argument for using a
vector store rather than a numpy array from the start.

### Persistence

Chroma writes SQLite plus index files to `persist_directory`. No server, no
container, no connection string. The whole store is a folder you can delete, and
`.gitignore` excludes it because it is derived data — reproducible from
`ingest.py` in seconds.

---

## 4. Chunking

### The fundamental tension

A vector is a fixed-size summary of whatever you embed.

- **Too large** — the vector averages several topics. A 4,000-word contract
  embeds to "a contract, generally", and matches every contract query equally.
- **Too small** — chunks lose the context that makes them interpretable. "The fee
  is $1,200 annually" is useless without knowing whose fee.

There is no universally correct size. It depends on document structure, query
type, and the embedding model's effective context.

### The three strategies here

**Recursive** — split on a priority list of separators, recursing into anything
still too big. General purpose, structure-agnostic. The custom separator list
(`\n## `, `\n### `, `\n\n`, `\n`, `. `, ` `) prefers semantic boundaries over
arbitrary ones.

**Markdown-aware** (used for the reported results) — split on headings first,
then size-limit. Works well here because contracts and employee records are
written with meaningful section headings, which are natural semantic boundaries
that a human author already identified.

**Contextual** — the interesting one. Anthropic published the observation that
chunks lose their referents. A clause reading *"The agreement may be terminated
with 30 days notice"* cannot be retrieved by "how do I exit the Apex contract?" —
it never says Apex. The fix: have an LLM write a sentence situating each chunk in
its parent document, prepend it, and embed the combination. The chunk's vector
now carries the entity names. Costs one LLM call per chunk at index time, and
nothing at query time.

### Overlap

150 characters shared between consecutive chunks. A sentence straddling a
boundary appears whole in at least one chunk. Overlap costs storage and creates
near-duplicate retrievals — which the RRF de-duplication in `merge()` partly
absorbs, since it keys on chunk text.

---

## 5. Query rewriting

### Two problems it solves

**Conversational references.** "Who won it in 2023?" embeds to something about
winning and 2023, with no idea what "it" is. The rewrite sees the history and
produces "Insurellm IIOTY award winner 2023".

**Vocabulary mismatch.** Users write questions; documents contain statements.
"How do I get out of this contract?" versus "Termination provisions". A rewrite
can move the query toward document vocabulary.

### The failure mode, and what the measurement showed

Rewriting can *generalise*. "What does the Apex Reinsurance contract say about
termination?" becoming "contract termination terms" now matches all 32 contracts
equally well. The specific proper noun was the highest-signal token and it is
gone.

This is exactly what the ablation caught. With a local llama3.2 as the rewriter:

```
vector search       hit@k 0.983   MRR 0.941
+ query rewriting   hit@k 1.000   MRR 0.914
```

Hit rate went **up** — rewriting rescued the questions plain search missed
entirely. MRR went **down** — it also blunted some precise questions. A single
metric would have shown one of those and hidden the other.

Two mitigations are in the code: the prompt explicitly instructs "keep every
proper noun", and `retrieve()` always searches the original query as well, so RRF
can recover anything the rewrite dropped.

---

## 6. Reciprocal Rank Fusion

### The problem

Two searches return two ranked lists with scores on different scales. The query
vectors differ, so the distance distributions differ. Averaging the scores
compares incomparable numbers.

### The solution

Ignore the scores; use the ranks.

```
RRF(chunk) = Σ over lists  1 / (K + rank_in_that_list)
```

with K = 60, the value from the original paper.

### What K does

It damps the difference between top ranks.

| ranks | without K | with K=60 |
|---|---|---|
| 1 vs 2 | 1.000 vs 0.500 (50% gap) | 0.0164 vs 0.0161 (2% gap) |
| 1 vs 10 | 1.000 vs 0.100 | 0.0164 vs 0.0143 |

Small K makes rank 1 dominate, so a single list's favourite wins outright. Large
K flattens everything toward equality, so *consensus across lists* decides. K=60
sits where a chunk ranked decently by several queries beats a chunk ranked first
by one.

### Worked example from this system

```
original query list:  [A, B, C, D, ...]
rewritten query list: [E, F, A, B, ...]

A: 1/61 + 1/63 = 0.01639 + 0.01587 = 0.03226   ← both lists
B: 1/62 + 1/64 = 0.01613 + 0.01563 = 0.03176   ← both lists
E: 1/61                            = 0.01639   ← one list only
```

A and B, found by both queries, outrank E which only one query liked. That is the
behaviour you want: agreement is evidence.

### Why not just concatenate

Concatenating and truncating gives the first list's results priority purely
because it ran first. RRF has no such bias and is the standard approach in hybrid
search, where it fuses dense (vector) and sparse (BM25) results.

---

## 7. Bi-encoders vs cross-encoders

The most important idea in the retrieval module.

### Bi-encoder — what the embedding model is

```
question ──► encoder ──► vector_q  ┐
                                    ├──► cosine similarity ──► score
chunk    ──► encoder ──► vector_c  ┘
```

The two texts are encoded **independently**. That is what makes search scalable:
chunk vectors are computed once at index time and reused for every query forever.
At query time you encode one thing and do fast vector math.

The cost: the encodings never see each other. The chunk's vector was computed
without knowing the question, so it must be a general-purpose summary. Fine
distinctions — this chunk says 2023 and the question asks about 2023 — are hard
to represent in a single fixed vector.

### Cross-encoder — what the re-ranker is

```
[question, chunk] ──► transformer ──► relevance score
```

Both texts go through **one** forward pass. Self-attention runs across the
concatenation, so the model can directly compare tokens in the question with
tokens in the chunk.

Far more accurate. Also unscalable: nothing can be precomputed, and scoring a
million chunks means a million forward passes.

### The two-stage pattern

```
659 chunks ──bi-encoder──► 20 per query ──RRF──► ~30 ──cross-encoder──► 6
   cheap, approximate, high recall          expensive, precise, high precision
```

This is how essentially all production search works. Recall first, precision
second.

### The measured payoff

```
vector search   hit@k 0.983   MRR 0.941   nDCG 0.923   P@k 0.572
+ re-ranking    hit@k 1.000   MRR 0.975   nDCG 0.962   P@k 0.617
```

Read that carefully. Hit rate barely moved — the right documents were already
being found. MRR and nDCG jumped — they were being found but **ranked badly**.
That is the precise signature of a re-ranking win, and it is why the
recommendation is to ship re-ranking rather than a bigger embedding model.

The cross-encoder used, `ms-marco-MiniLM-L-6-v2`, outputs raw logits. Measured on
a real pair: `+6.50` for relevant, `-11.09` for irrelevant. Not probabilities,
but sharply separated, which is all ordering needs.

---

## 8. Prompt grounding

### The structure

```
SystemMessage:  rules + extracts
HumanMessage:   (history turn 1)
AIMessage:      (history turn 1)
...
HumanMessage:   current question
```

Extracts live in the **system** message, not the user turn. That positions them
as ground truth the model was configured with, rather than as something the user
claimed. It also keeps them fixed at the front of the context as the conversation
grows.

### The four instructions and their failure modes

| Instruction | Prevents |
|---|---|
| "Answer only from the extracts" | Answering from parametric memory — always fabrication here |
| "say you do not have that information" | Inventing when evidence is absent; gives an explicit alternative to guessing |
| "never guess a name, date, figure or contract term" | Names the fields where a hallucination causes real damage |
| "Cite the extracts you used as [source]" | Unverifiable answers; the format matches `build_context`'s labels |

The length cap exists because verbose answers score *worse* on groundedness —
padding is where unsupported claims appear.

### Temperature 0

Every model call in this project uses `temperature=0.0`. For grounded QA,
sampling diversity is not a feature; and an evaluation harness measuring a random
variable produces differences that are noise.

---

## 9. Information retrieval metrics

Full definitions, worked arithmetic and the reasoning for each are in
[05-code-walkthrough-evaluation.md](05-code-walkthrough-evaluation.md). The
summary:

| Metric | Formula | Rank-sensitive | Answers |
|---|---|---|---|
| hit@k | `any(relevant)` | No | Was it retrievable at all? |
| MRR | `1 / rank_of_first_relevant` | Strongly | Is the best evidence at the top? |
| nDCG | `DCG / IDCG`, discount `1/log2(rank+1)` | Yes, gently | Is the whole ordering good? |
| P@k | `relevant / k` | No | How much of the context is useful? |
| coverage | `keywords_found / keywords_total` | No | For multi-part answers, did we get all parts? |

### Why proxy relevance works here

Gold relevance labels would require a human to judge all 659 chunks against all
150 questions — about 98,000 judgements. Instead, a chunk is relevant if it
contains a gold keyword.

This is imperfect. It has false positives (a chunk mentioning "Thompson" about a
different Thompson) and false negatives (a chunk that paraphrases without using
the keyword). But it is **cheap, deterministic, and unbiased across
configurations** — the same proxy scores every configuration, so differences
between configurations are real even if the absolute level is approximate. Since
the entire purpose is ranking configurations against each other, that is the
property that matters.

---

## 10. LLM-as-a-judge

### Why it works

Grading an answer against a reference is a much easier task than producing the
answer. The judge is not asked to recall who won the award in 2023; it is asked
whether two pieces of text agree. Comparison is far more reliable than recall.

### Known biases

| Bias | Mitigation here |
|---|---|
| Length bias — longer answers score higher | System prompt caps answer length; groundedness explicitly penalises unsupported extras |
| Self-preference — models favour their own output | `JUDGE_MODEL` is configurable separately from the answering model |
| Score clustering at 4 | Explicit anchors: "a wrong fact scores 1", "reserve 5 for expert sign-off" |
| Position bias | Not applicable — single answer, not pairwise comparison |

### Why three separate axes

Averaging them into one score destroys the diagnostic value. "Correct but
incomplete" and "complete but partly wrong" are different bugs with different
fixes — the first is usually retrieval, the second usually the prompt. Three
numbers tell you which; one number tells you neither.

### Structured output

`with_structured_output(Verdict)` binds a pydantic schema so scores come back as
validated integers in 1–5, not prose to regex. A model returning 7 fails
validation rather than quietly skewing the average.

---

## 11. Held-out evaluation

### The problem

Chunk size, overlap, separators, `k`, `candidate_k`, prompt wording — every one
was chosen by looking at results. If the final numbers come from the same
questions, they measure how well the configuration was fitted to those questions.
That is the same overfitting problem as reporting training accuracy in supervised
learning.

### The solution

```python
dev, holdout = split_questions(questions, holdout=0.4, seed=42)
```

90 dev questions for tuning, 60 holdout for reporting. Deterministic seed,
stratified by category so both halves contain the same difficulty mix.

Every number in the project README comes from the holdout half.

### Why stratification matters here

Measured nDCG by category ranges from 0.891 (`temporal`) to 0.997 (`holistic`).
An unstratified split could easily put most temporal questions on one side,
making the two halves not comparable and the holdout number an artefact of the
split rather than a property of the system.
