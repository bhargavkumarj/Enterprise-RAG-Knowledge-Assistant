# Code Walkthrough — Evaluation

Covers `rag/evaluation/dataset.py`, `metrics.py`, `judge.py`, `harness.py` and
`evaluate.py`.

The evaluation package is the part that separates this from a RAG demo. A demo
shows you one good answer. This tells you, with numbers, whether a change made
retrieval better or worse.

---

## `rag/evaluation/dataset.py`

### `Question`

```python
class Question(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str
```

One line of `data/questions.jsonl`:

```json
{"question": "Who won the prestigious IIOTY award in 2023?",
 "keywords": ["Maxine", "Thompson", "IIOTY"],
 "reference_answer": "Maxine Thompson won the prestigious Insurellm Innovator of the Year (IIOTY) award in 2023.",
 "category": "direct_fact"}
```

Each field feeds a different consumer:

- **`question`** — the input to both retrieval and answering.
- **`keywords`** — the *retrieval* ground truth. A chunk is judged relevant if it
  contains one of these. This is what makes retrieval measurable without a human
  labelling all 659 chunks against all 150 questions.
- **`reference_answer`** — the *answering* ground truth, used only by the judge.
- **`category`** — enables per-category breakdowns and stratified splitting.

The seven categories in the set: `direct_fact`, `numerical`, `temporal`,
`relationship`, `comparative`, `spanning`, `holistic`. They span genuinely
different retrieval difficulties. `direct_fact` needs one chunk. `spanning` needs
several. `comparative` needs several *and* the model to relate them.

Using pydantic rather than dicts means a malformed line fails at load with a
field-level error, not 200 lines later with a `KeyError`.

### `split_questions()`

```python
by_category: dict[str, list[Question]] = {}
for question in questions:
    by_category.setdefault(question.category, []).append(question)

rng = random.Random(seed)
for category in sorted(by_category):
    group = by_category[category][:]
    rng.shuffle(group)
    cut = int(len(group) * (1 - holdout))
    dev.extend(group[:cut])
    held_out.extend(group[cut:])
```

Four decisions in nine lines.

**Stratified by category.** A plain random 60/40 split could put most `temporal`
questions in dev and most `holistic` in holdout. Since categories have genuinely
different difficulty — measured nDCG ranges from 0.891 for `temporal` to 0.997
for `holistic` — an unbalanced split makes the two halves incomparable and the
holdout number an artefact of the split. Splitting within each category keeps the
category mix identical in both halves.

**`random.Random(seed)`, not `random.shuffle`.** A local generator instance does
not touch the global random state, so calling this cannot perturb anything else
that uses randomness, and nothing else can perturb it. Combined with the fixed
seed, the holdout set is byte-identical on every machine and every run.

**`sorted(by_category)`.** Dict iteration order follows insertion order, which
follows file order. Sorting the keys means the shuffle sequence is consumed in a
fixed order, so the split does not change if someone reorders the JSONL file.

**`group[:]`.** Shuffles a copy. `rng.shuffle` is in-place, and mutating the
caller's list would be a surprising side effect.

**Why hold out at all.** Chunk size, overlap, the separator list, `k`, the prompt
wording — every one of those was chosen by looking at results. If those results
came from the same questions used for the final numbers, the numbers measure how
well the configuration was fitted to those questions, not how well it retrieves.
The holdout half was never looked at while tuning.

---

## `rag/evaluation/metrics.py`

Pure functions. No I/O, no model calls, no global state. Every one is
deterministic given chunks and keywords, which makes them trivial to test and
impossible to make flaky.

### `_relevance()` — the foundation

```python
lowered = [chunk.text.lower() for chunk in chunks]
wanted = [keyword.lower() for keyword in keywords]
return [int(any(keyword in text for keyword in wanted)) for text in lowered]
```

Returns a binary relevance vector aligned to the ranked chunk list — e.g.
`[0, 1, 1, 0, 0, 1]` means chunks 2, 3 and 6 are relevant.

**Any, not all.** One keyword is enough. Requiring all of `["Maxine",
"Thompson", "IIOTY"]` in a single chunk would mark a chunk saying "Maxine
Thompson received the award" as irrelevant, which is wrong.

**Substring, not token.** `"Maxine" in text` matches inside words. The
false-positive risk is real but small on this corpus, and the alternative
(tokenising, stemming) adds machinery and its own errors.

**Lowercased once.** Both sides normalised, so `"IIOTY"` matches `"iioty"`.

This is *proxy* relevance, not human judgement. That trade is discussed in
[07-design-decisions.md](07-design-decisions.md) — the short version is that it
is cheap, deterministic, unbiased across configurations, and good enough to rank
configurations against each other, which is what it is used for.

### `hit_rate()`

```python
return float(any(_relevance(chunks, keywords)))
```

1.0 if any retrieved chunk is relevant, else 0.0. Averaged over questions this is
**recall@k**: the fraction of questions where the answer was retrievable at all.

It is the *floor* metric. If hit@k is 0.9, then 10% of questions cannot be
answered correctly no matter how good the generator is, because the evidence
never reached the prompt. Measured: 0.983 for plain vector search, 1.000 once
rewriting or re-ranking is added.

### `keyword_coverage()`

```python
blob = " ".join(chunk.text.lower() for chunk in chunks)
found = sum(1 for keyword in keywords if keyword.lower() in blob)
return found / len(keywords)
```

Concatenates *all* retrieved chunks and asks what fraction of keywords appear
anywhere in that union. Unlike the other metrics it is rank-insensitive by
design.

This is the metric for **spanning** questions. "Which customers are signed up to
Markellm and what do they pay?" has its answer distributed across several
contracts; no single chunk contains every keyword. Hit rate would say 1.0 after
finding one contract. Coverage says 0.4, which is the truth.

### `mrr()`

```python
for rank, relevant in enumerate(_relevance(chunks, keywords), start=1):
    if relevant:
        return 1.0 / rank
return 0.0
```

Reciprocal rank of the *first* relevant chunk. First position → 1.0, second →
0.5, third → 0.333, tenth → 0.1, none → 0.0.

Averaged over questions this is Mean Reciprocal Rank. It measures whether the
best evidence is at the *top*, which matters more than it looks: LLMs weight
earlier context more heavily, and with `k=6` the sixth extract competes with five
others for attention.

The steep drop from 1.0 to 0.5 is the point. Moving the right chunk from rank 2
to rank 1 is worth as much as moving it from rank 2 to nowhere-past-10. MRR is
the metric that made re-ranking's value visible: 0.941 → 0.975 while hit rate
barely moved, i.e. re-ranking found the *same* documents but ordered them better.

### `dcg()` and `ndcg()`

```python
def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))
```

Discounted Cumulative Gain. Each relevant chunk contributes `1 / log2(rank + 1)`:

| rank | discount `1/log2(rank+1)` |
|---|---|
| 1 | 1.000 |
| 2 | 0.631 |
| 3 | 0.500 |
| 4 | 0.431 |
| 5 | 0.387 |
| 6 | 0.356 |

The logarithm is a gentler penalty than MRR's `1/rank`, which reflects how people
actually read a ranked list — position 2 is nearly as good as position 1, but
position 20 is much worse than position 10.

```python
def ndcg(chunks, keywords) -> float:
    relevances = _relevance(chunks, keywords)
    ideal = dcg(sorted(relevances, reverse=True))
    return dcg(relevances) / ideal if ideal else 0.0
```

Normalised DCG divides by the best achievable DCG **for this question's actual
relevance vector**. `sorted(relevances, reverse=True)` pushes all the 1s to the
front, which is the ideal ranking of the chunks that were retrieved.

Worked example. Retrieved relevance `[0, 1, 1, 0, 0, 1]`:

```
relevance = [0, 1, 1, 0, 0, 1]

DCG   = 0(1.000) + 1(0.631) + 1(0.500) + 0(0.431) + 0(0.387) + 1(0.356)
      = 1.487

ideal     = [1, 1, 1, 0, 0, 0]      the same three hits, ranked perfectly
IDCG  = 1(1.000) + 1(0.631) + 1(0.500)
      = 2.131

nDCG  = 1.487 / 2.131 = 0.698
```

Normalising makes questions with different numbers of relevant chunks
comparable — a question with 1 relevant chunk and a question with 4 both score on
a 0–1 scale, so averaging across the question set is meaningful.

The `if ideal else 0.0` guard handles a question where nothing relevant was
retrieved: `ideal` is 0 and the division would raise.

### `precision_at_k()`

```python
relevances = _relevance(chunks, keywords)
return sum(relevances) / len(relevances) if relevances else 0.0
```

What fraction of the k returned chunks are relevant. Rank-insensitive.

This is the **context efficiency** metric. Precision 0.572 at k=6 means about 3.4
of the 6 extracts are useful and 2.6 are noise the model has to read past. Noise
is not free: it costs tokens, and irrelevant text measurably increases the chance
of a model latching onto the wrong fact.

Precision rose from 0.572 to 0.617 with re-ranking and 0.622 with both — the
clearest evidence that re-ranking improves what actually lands in the prompt.

### `retrieval_metrics()`

Bundles all five into a dict. The harness stores the dict per case, so adding a
metric here automatically appears in every stored report without touching the
harness.

### Why five metrics and not one

They disagree, and the disagreements are the finding:

| Metric | Answers |
|---|---|
| hit@k | Was the answer retrievable at all? |
| MRR | Was the best evidence at the top? |
| nDCG | Was the whole ordering good? |
| P@k | How much of the context is useful? |
| coverage | For multi-part answers, did we get all the parts? |

In the measured ablation, query rewriting *improved* hit rate (0.983 → 1.000)
while *reducing* MRR (0.941 → 0.914). A single composite score would have hidden
that completely. Two metrics moving in opposite directions told the real story:
rewriting rescued the questions plain search missed entirely, but blunted some
precise questions into vaguer ones.

---

## `rag/evaluation/judge.py`

### Why an LLM judge at all

Retrieval metrics cannot tell you whether the *answer* is good. A system can
retrieve perfectly and still produce a wrong answer by misreading the extracts,
and exact-match scoring against a reference fails on correct paraphrases. Human
grading is the gold standard and does not scale to 60 questions per
configuration × 4 configurations.

An LLM grading a generated answer against a *reference answer* is a much easier
task than answering the question — it is comparison, not recall.

### The three axes

```
- accuracy: does it contradict the reference or invent facts? A wrong fact scores 1.
- completeness: does it cover everything the reference covers?
- groundedness: is every claim supported by the reference, with no unsupported detail?
```

These are separable failures:

- High accuracy, low completeness → correct but partial. Usually a retrieval
  problem: the model only got some of the evidence.
- High accuracy, low groundedness → correct *and* padded with unsupported
  extras. A prompt problem, and the reason the system prompt caps answer length.
- Low accuracy → wrong. Could be either; the retrieval metrics on the same case
  disambiguate.

"A wrong fact scores 1" gives the model an anchor. Without a concrete rule, LLM
judges cluster everything at 4 and the metric stops discriminating. "Reserve 5
for answers a domain expert would sign off on unchanged" anchors the top end the
same way.

### `Verdict`

```python
class Verdict(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    comment: str = Field(description="One sentence explaining the scores")
```

`ge=1, le=5` are enforced by pydantic and, with `with_structured_output`, are
usually exported into the JSON schema the provider enforces. A model returning 7
fails validation loudly rather than silently skewing an average.

`comment` is required, and it is not decoration. When a configuration scores 2.8
on accuracy you need to know *why*, and 60 one-line comments in the JSON report
are how you find out. Asking for the comment also tends to improve the scores
themselves — the model commits to a reason.

`mean` is a convenience property, not used by the harness, which reports the
three axes separately on purpose.

### `judge()`

```python
model = settings.judge().with_structured_output(Verdict)
return model.invoke(PROMPT.format(question=question, reference=reference, answer=answer))
```

Three lines. `settings.judge()` is `chat_model(temperature=0.0, model=judge_model)`
— zero temperature because a grader that returns different scores for the same
input makes every comparison noise, and a separate model name so you can grade
with a stronger model than you answer with.

Note the judge sees the **reference answer, not the retrieved context**. It is
grading answer quality against ground truth, not faithfulness to the extracts.
That keeps the axes clean — faithfulness to the *wrong* extracts is a retrieval
failure, and the retrieval metrics on the same case already measure it.

---

## `rag/evaluation/harness.py`

### `Case`

```python
@dataclass
class Case:
    question: str
    category: str
    metrics: dict[str, float]
    sources: list[str]
    answer: str | None = None
    verdict: dict | None = None
```

One question's full result. `answer` and `verdict` are optional because
retrieval-only runs (the default, no API key needed) do not produce them.

`sources` records *which documents* were retrieved. When a question scores 0, the
sources tell you whether the retriever found plausible-but-wrong documents or
nothing related at all — two very different bugs.

`verdict` is a plain dict rather than the `Verdict` model so `dataclasses.asdict`
can serialise the whole `Case` without a custom encoder.

### `RunResult`

```python
def mean(self, key: str) -> float:
    values = [case.metrics[key] for case in self.cases if key in case.metrics]
    return statistics.fmean(values) if values else 0.0
```

`statistics.fmean` rather than `sum/len`: it is the float-optimised version and
handles the empty case explicitly above.

The `if key in case.metrics` filter means adding a new metric does not break
reports generated before it existed.

```python
def by_category(self, key: str) -> dict[str, float]:
```

Groups cases by category and averages within each. This is what surfaced the
finding that `temporal` questions are the hardest (nDCG 0.891) while `holistic`
are nearly solved (0.997) — invisible in the aggregate 0.962.

```python
def summary(self) -> dict:
    data = {...}
    if any(case.verdict for case in self.cases):
        data.update(accuracy=..., completeness=..., groundedness=...)
    return data
```

The judge columns appear only when there are verdicts, so retrieval-only runs
produce a clean five-column summary instead of a table full of zeros.

### `_evaluate_one()`

```python
if with_answers:
    answer = assistant.ask(question.question, **options)
    chunks = answer.chunks
    verdict = judging.judge(question.question, answer.text, question.reference_answer)
    return Case(..., metrics=retrieval_metrics(chunks, question.keywords), ...)

trace = retrieval.retrieve(question.question, **options)
return Case(..., metrics=retrieval_metrics(trace.chunks, question.keywords), ...)
```

Two modes, one crucial property: **retrieval metrics are computed from the chunks
that actually went into the answer**, not from a separate retrieval call. If the
two branches retrieved independently, the retrieval score and the answer score
would describe different retrievals and could not be correlated.

The retrieval-only branch skips generation entirely. That is what makes
`evaluate.py` runnable with no API key and in seconds rather than minutes — the
full 60-question retrieval ablation ran in about 8 seconds per configuration.

### `run()`

```python
result = RunResult(name=name, config=dict(options))
retrieval.store.open_store()  # load the model once before fanning out
with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    jobs = [pool.submit(_evaluate_one, q, with_answers, options) for q in questions]
    for job in tqdm(futures.as_completed(jobs), total=len(jobs), desc=name):
        result.cases.append(job.result())
```

**The warm-up line is a bug fix.** Without it, four threads race to load the
embedding model. The `Settings` lock now prevents the crash, but the warm-up also
avoids three threads blocking on the lock for the model load, and makes the
progress bar meaningful from the first tick.

**Threads, not processes.** Every task is I/O bound — an HTTP call, or a Chroma
query that releases the GIL in native code. Processes would duplicate the
embedding model in memory per worker.

**`as_completed`, not `map`.** Results arrive in completion order, so the
progress bar advances as work finishes rather than stalling behind one slow
question. Order does not matter because every `Case` carries its own question.

**`config=dict(options)`** snapshots the retrieval options into the result, so a
saved report records *what was run*, not just what came out. Six months later
that is the difference between a usable report and a set of unexplained numbers.

`job.result()` re-raises exceptions from the worker, so a failure surfaces
instead of silently producing a short result set.

### `compare()`

```python
judged = any("accuracy" in r.summary() for r in results)
headers = ["configuration", "hit@k", "MRR", "nDCG", "P@k", "coverage"]
if judged:
    headers += ["accuracy", "complete", "grounded"]
```

Columns adapt to the data present.

```python
widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]

def line(cells):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
```

`zip(headers, *rows)` transposes: each `column` is the header plus every value in
that column, so `max(len(...))` gives the width needed. `.rstrip()` removes
trailing padding so the output has no invisible whitespace.

Plain text rather than a dataframe: no pandas dependency, and it pastes cleanly
into a terminal, a commit message or a README.

### `save()`

```python
stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
path = REPORTS / f"{label}-{stamp}.json"
```

UTC and a sortable timestamp, so reports from different machines interleave
correctly and `ls` sorts chronologically. Never overwrites.

```python
"summaries": [r.summary() for r in results],
"runs": [{"name": r.name, "cases": [asdict(case) for case in r.cases]} for r in results],
```

Both levels. Summaries are for scanning; per-case detail is for diagnosis —
which questions regressed, what was retrieved for them, what the judge said. An
average that moves from 0.962 to 0.940 is a number; the three cases that caused
it are actionable.

---

## `evaluate.py`

```python
dev, holdout = dataset.split_questions(dataset.load_questions())
questions = {"dev": dev, "holdout": holdout, "all": dev + holdout}[args.split]
```

Dict dispatch on the validated `--split` choice. `argparse` has already
restricted the values, so a `KeyError` is impossible.

```python
if args.ablation:
    configurations = [
        ("vector search", dict(use_rewrite=False, use_rerank=False)),
        ("+ query rewriting", dict(use_rewrite=True, use_rerank=False)),
        ("+ re-ranking", dict(use_rewrite=False, use_rerank=True)),
        ("+ both", dict(use_rewrite=True, use_rerank=True)),
    ]
```

A proper 2×2 factorial ablation: baseline, each feature alone, both together.
Testing only "baseline vs everything on" would have shown a net improvement and
completely hidden that query rewriting *reduces* MRR on its own. Isolating each
factor is what produced the actual recommendation — ship re-ranking, hold
rewriting until the rewriter is stronger.

Every configuration runs on the **same question list in the same process**
against the **same vector store**, so the only difference between rows is the
retrieval options.

```python
best = max(results, key=lambda r: r.mean("ndcg"))
print(f"\nnDCG by category for '{best.name}':")
```

Picks the winner by nDCG and breaks it down by category, surfacing where the best
configuration still struggles.

```python
path = harness.save(results, "ablation" if args.ablation else args.split)
```

Labels the report by what it was, so `reports/` stays readable as it fills up.
