import threading
from dataclasses import dataclass

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag import store
from rag.config import settings

REWRITE_PROMPT = """You turn a user's message into a search query for the {organisation}
knowledge base, which holds company overviews, product sheets, signed contracts and
employee records.

Conversation so far:
{history}

User's message: {question}

Resolve pronouns and references using the conversation, keep every proper noun, and
expand obvious abbreviations. Respond with the search query only."""

RERANK_PROMPT = """Rank these knowledge base extracts by how useful they are for
answering the question. Return the ids of the {keep} most useful extracts, best first.
Discard anything irrelevant.

Question: {question}

{extracts}"""


class Ranking(BaseModel):
    ids: list[int] = Field(description="Extract ids, most relevant first")


@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str
    score: float = 0.0

    @classmethod
    def from_document(cls, document: Document, score: float = 0.0) -> "Chunk":
        return cls(
            text=document.page_content,
            source=document.metadata.get("source", "unknown"),
            doc_type=document.metadata.get("doc_type", "unknown"),
            score=score,
        )


@dataclass
class RetrievalTrace:
    query: str
    rewritten: str | None
    candidates: int
    chunks: list[Chunk]


_cross_encoder_lock = threading.Lock()
_cross_encoder_model = None


def cross_encoder():
    global _cross_encoder_model
    with _cross_encoder_lock:
        if _cross_encoder_model is None:
            from sentence_transformers import CrossEncoder

            _cross_encoder_model = CrossEncoder(settings.cross_encoder_model)
        return _cross_encoder_model


def rewrite(question: str, history: list[dict] | None = None) -> str:
    turns = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-6:]) or "(none)"
    prompt = REWRITE_PROMPT.format(
        organisation=settings.organisation, history=turns, question=question
    )
    query = settings.chat_model(temperature=0.0).invoke(prompt).content.strip()
    return query.strip('"').strip() or question


def vector_search(query: str, k: int) -> list[Chunk]:
    hits = store.open_store().similarity_search_with_relevance_scores(query, k=k)
    return [Chunk.from_document(doc, score) for doc, score in hits]


def merge(*groups: list[Chunk]) -> list[Chunk]:
    """Reciprocal rank fusion over several result lists."""
    scores: dict[str, float] = {}
    keep: dict[str, Chunk] = {}
    for group in groups:
        for rank, chunk in enumerate(group, start=1):
            scores[chunk.text] = scores.get(chunk.text, 0.0) + 1.0 / (60 + rank)
            keep.setdefault(chunk.text, chunk)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [keep[text] for text, _ in ordered]


def rerank_with_llm(question: str, chunks: list[Chunk], keep: int) -> list[Chunk]:
    extracts = "\n\n".join(
        f"[{i}] ({chunk.doc_type}/{chunk.source})\n{chunk.text[:1200]}"
        for i, chunk in enumerate(chunks)
    )
    model = settings.chat_model(temperature=0.0).with_structured_output(Ranking)
    try:
        ranking = model.invoke(
            RERANK_PROMPT.format(question=question, keep=keep, extracts=extracts)
        )
    except Exception:
        return chunks[:keep]

    seen, ordered = set(), []
    for index in ranking.ids:
        if 0 <= index < len(chunks) and index not in seen:
            seen.add(index)
            ordered.append(chunks[index])
    return (ordered or chunks)[:keep]


def rerank_with_cross_encoder(question: str, chunks: list[Chunk], keep: int) -> list[Chunk]:
    scores = cross_encoder().predict([(question, chunk.text) for chunk in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    return sorted(chunks, key=lambda c: c.score, reverse=True)[:keep]


def retrieve(
    question: str,
    history: list[dict] | None = None,
    k: int | None = None,
    use_rewrite: bool | None = None,
    use_rerank: bool | None = None,
) -> RetrievalTrace:
    k = k or settings.top_k
    use_rewrite = settings.rewrite_query if use_rewrite is None else use_rewrite
    use_rerank = settings.rerank if use_rerank is None else use_rerank

    rewritten = rewrite(question, history) if use_rewrite else None
    depth = settings.candidate_k if use_rerank else k

    groups = [vector_search(question, depth)]
    if rewritten and rewritten.lower() != question.lower():
        groups.append(vector_search(rewritten, depth))
    candidates = merge(*groups)

    if use_rerank and len(candidates) > k:
        if settings.reranker == "cross-encoder":
            chunks = rerank_with_cross_encoder(rewritten or question, candidates, k)
        else:
            chunks = rerank_with_llm(rewritten or question, candidates, k)
    else:
        chunks = candidates[:k]

    return RetrievalTrace(
        query=question, rewritten=rewritten, candidates=len(candidates), chunks=chunks
    )
