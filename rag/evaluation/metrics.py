import math

from rag.retrieval import Chunk


def _relevance(chunks: list[Chunk], keywords: list[str]) -> list[int]:
    """A chunk counts as relevant when it mentions at least one gold keyword."""
    lowered = [chunk.text.lower() for chunk in chunks]
    wanted = [keyword.lower() for keyword in keywords]
    return [int(any(keyword in text for keyword in wanted)) for text in lowered]


def hit_rate(chunks: list[Chunk], keywords: list[str]) -> float:
    return float(any(_relevance(chunks, keywords)))


def keyword_coverage(chunks: list[Chunk], keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    blob = " ".join(chunk.text.lower() for chunk in chunks)
    found = sum(1 for keyword in keywords if keyword.lower() in blob)
    return found / len(keywords)


def mrr(chunks: list[Chunk], keywords: list[str]) -> float:
    for rank, relevant in enumerate(_relevance(chunks, keywords), start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def ndcg(chunks: list[Chunk], keywords: list[str]) -> float:
    relevances = _relevance(chunks, keywords)
    ideal = dcg(sorted(relevances, reverse=True))
    return dcg(relevances) / ideal if ideal else 0.0


def precision_at_k(chunks: list[Chunk], keywords: list[str]) -> float:
    relevances = _relevance(chunks, keywords)
    return sum(relevances) / len(relevances) if relevances else 0.0


def retrieval_metrics(chunks: list[Chunk], keywords: list[str]) -> dict[str, float]:
    return {
        "hit_rate": hit_rate(chunks, keywords),
        "mrr": mrr(chunks, keywords),
        "ndcg": ndcg(chunks, keywords),
        "precision": precision_at_k(chunks, keywords),
        "keyword_coverage": keyword_coverage(chunks, keywords),
    }
