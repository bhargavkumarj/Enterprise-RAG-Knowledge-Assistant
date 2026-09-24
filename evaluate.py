"""Score retrieval and answer quality on the held-out question set."""

import argparse

from rag.config import settings
from rag.evaluation import dataset, harness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="holdout")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate the first N questions")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Compare plain vector search against query rewriting and re-ranking",
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="Also generate answers and grade them with the LLM judge",
    )
    args = parser.parse_args()

    dev, holdout = dataset.split_questions(dataset.load_questions())
    questions = {"dev": dev, "holdout": holdout, "all": dev + holdout}[args.split]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Evaluating {len(questions)} {args.split} questions at k={args.k}\n")

    if args.ablation:
        configurations = [
            ("vector search", dict(use_rewrite=False, use_rerank=False)),
            ("+ query rewriting", dict(use_rewrite=True, use_rerank=False)),
            ("+ re-ranking", dict(use_rewrite=False, use_rerank=True)),
            ("+ both", dict(use_rewrite=True, use_rerank=True)),
        ]
    else:
        configurations = [
            (
                f"rewrite={settings.rewrite_query} rerank={settings.rerank}",
                dict(use_rewrite=settings.rewrite_query, use_rerank=settings.rerank),
            )
        ]

    results = [
        harness.run(
            questions,
            name=name,
            with_answers=args.answers,
            workers=args.workers,
            k=args.k,
            **options,
        )
        for name, options in configurations
    ]

    print()
    print(harness.compare(results))

    best = max(results, key=lambda r: r.mean("ndcg"))
    print(f"\nnDCG by category for '{best.name}':")
    for category, score in best.by_category("ndcg").items():
        print(f"  {category:<16} {score:.3f}")

    path = harness.save(results, "ablation" if args.ablation else args.split)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
