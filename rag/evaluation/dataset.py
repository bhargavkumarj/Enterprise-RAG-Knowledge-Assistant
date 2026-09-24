import json
import random
from pathlib import Path

from pydantic import BaseModel

from rag.config import settings


class Question(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or settings.questions
    with open(path, encoding="utf-8") as handle:
        return [Question(**json.loads(line)) for line in handle if line.strip()]


def split_questions(
    questions: list[Question], holdout: float = 0.4, seed: int = 42
) -> tuple[list[Question], list[Question]]:
    """Deterministic dev/held-out split, stratified by category.

    Tuning decisions are made on `dev`; the numbers that get reported come from
    the held-out half, which no chunking or prompt change was fitted against.
    """
    by_category: dict[str, list[Question]] = {}
    for question in questions:
        by_category.setdefault(question.category, []).append(question)

    rng = random.Random(seed)
    dev, held_out = [], []
    for category in sorted(by_category):
        group = by_category[category][:]
        rng.shuffle(group)
        cut = int(len(group) * (1 - holdout))
        dev.extend(group[:cut])
        held_out.extend(group[cut:])
    return dev, held_out
