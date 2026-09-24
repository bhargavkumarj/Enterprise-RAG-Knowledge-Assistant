import concurrent.futures as futures
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from rag import assistant, retrieval
from rag.config import ROOT
from rag.evaluation import judge as judging
from rag.evaluation.dataset import Question
from rag.evaluation.metrics import retrieval_metrics

REPORTS = ROOT / "reports"


@dataclass
class Case:
    question: str
    category: str
    metrics: dict[str, float]
    sources: list[str]
    answer: str | None = None
    verdict: dict | None = None


@dataclass
class RunResult:
    name: str
    config: dict
    cases: list[Case] = field(default_factory=list)

    def mean(self, key: str) -> float:
        values = [case.metrics[key] for case in self.cases if key in case.metrics]
        return statistics.fmean(values) if values else 0.0

    def judged_mean(self, key: str) -> float:
        values = [case.verdict[key] for case in self.cases if case.verdict]
        return statistics.fmean(values) if values else 0.0

    def by_category(self, key: str) -> dict[str, float]:
        groups: dict[str, list[float]] = {}
        for case in self.cases:
            groups.setdefault(case.category, []).append(case.metrics.get(key, 0.0))
        return {name: statistics.fmean(values) for name, values in sorted(groups.items())}

    def summary(self) -> dict:
        data = {
            "name": self.name,
            "config": self.config,
            "questions": len(self.cases),
            "hit_rate": self.mean("hit_rate"),
            "mrr": self.mean("mrr"),
            "ndcg": self.mean("ndcg"),
            "precision": self.mean("precision"),
            "keyword_coverage": self.mean("keyword_coverage"),
        }
        if any(case.verdict for case in self.cases):
            data.update(
                accuracy=self.judged_mean("accuracy"),
                completeness=self.judged_mean("completeness"),
                groundedness=self.judged_mean("groundedness"),
            )
        return data


def _evaluate_one(question: Question, with_answers: bool, options: dict) -> Case:
    if with_answers:
        answer = assistant.ask(question.question, **options)
        chunks = answer.chunks
        verdict = judging.judge(question.question, answer.text, question.reference_answer)
        return Case(
            question=question.question,
            category=question.category,
            metrics=retrieval_metrics(chunks, question.keywords),
            sources=[f"{c.doc_type}/{c.source}" for c in chunks],
            answer=answer.text,
            verdict={
                "accuracy": verdict.accuracy,
                "completeness": verdict.completeness,
                "groundedness": verdict.groundedness,
                "comment": verdict.comment,
            },
        )

    trace = retrieval.retrieve(question.question, **options)
    return Case(
        question=question.question,
        category=question.category,
        metrics=retrieval_metrics(trace.chunks, question.keywords),
        sources=[f"{c.doc_type}/{c.source}" for c in trace.chunks],
    )


def run(
    questions: list[Question],
    name: str,
    with_answers: bool = False,
    workers: int = 4,
    **options,
) -> RunResult:
    result = RunResult(name=name, config=dict(options))
    retrieval.store.open_store()  # load the model once before fanning out
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_evaluate_one, q, with_answers, options) for q in questions]
        for job in tqdm(futures.as_completed(jobs), total=len(jobs), desc=name):
            result.cases.append(job.result())
    return result


def compare(results: list[RunResult]) -> str:
    judged = any("accuracy" in r.summary() for r in results)
    headers = ["configuration", "hit@k", "MRR", "nDCG", "P@k", "coverage"]
    if judged:
        headers += ["accuracy", "complete", "grounded"]

    rows = []
    for result in results:
        summary = result.summary()
        row = [
            result.name,
            f"{summary['hit_rate']:.3f}",
            f"{summary['mrr']:.3f}",
            f"{summary['ndcg']:.3f}",
            f"{summary['precision']:.3f}",
            f"{summary['keyword_coverage']:.3f}",
        ]
        if judged:
            row += [
                f"{summary.get('accuracy', 0):.2f}",
                f"{summary.get('completeness', 0):.2f}",
                f"{summary.get('groundedness', 0):.2f}",
            ]
        rows.append(row)

    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]

    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()

    return "\n".join([line(headers), line("-" * w for w in widths), *(line(r) for r in rows)])


def save(results: list[RunResult], label: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = REPORTS / f"{label}-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated": stamp,
                "summaries": [r.summary() for r in results],
                "runs": [
                    {"name": r.name, "cases": [asdict(case) for case in r.cases]}
                    for r in results
                ],
            },
            indent=2,
        )
    )
    return path
