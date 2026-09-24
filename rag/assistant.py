from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag import retrieval
from rag.config import settings

SYSTEM_PROMPT = """You are the internal knowledge assistant for {organisation}.

Answer only from the extracts below. They come from company overviews, product
sheets, signed contracts and employee records. If the extracts do not contain the
answer, say you do not have that information — never guess a name, date, figure or
contract term. Cite the extracts you used as [source] at the end of the sentence
they support. Keep answers tight: two or three sentences unless asked for detail.

Extracts:
{context}"""


@dataclass
class Answer:
    text: str
    chunks: list[retrieval.Chunk]
    rewritten: str | None = None

    @property
    def sources(self) -> list[str]:
        seen = []
        for chunk in self.chunks:
            label = f"{chunk.doc_type}/{chunk.source}"
            if label not in seen:
                seen.append(label)
        return seen


def build_context(chunks: list[retrieval.Chunk]) -> str:
    return "\n\n".join(
        f"[{chunk.source}] ({chunk.doc_type})\n{chunk.text}" for chunk in chunks
    )


def to_messages(history: list[dict] | None):
    messages = []
    for turn in history or []:
        role, content = turn.get("role"), turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def ask(question: str, history: list[dict] | None = None, **retrieval_kwargs) -> Answer:
    trace = retrieval.retrieve(question, history=history, **retrieval_kwargs)
    context = build_context(trace.chunks) or "(no relevant extracts found)"

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT.format(organisation=settings.organisation, context=context)
        ),
        *to_messages(history),
        HumanMessage(content=question),
    ]
    response = settings.chat_model(temperature=0.0).invoke(messages)
    return Answer(text=response.content, chunks=trace.chunks, rewritten=trace.rewritten)


def stream(question: str, history: list[dict] | None = None, **retrieval_kwargs):
    """Yield (partial_answer, chunks) so the UI can render context immediately."""
    trace = retrieval.retrieve(question, history=history, **retrieval_kwargs)
    context = build_context(trace.chunks) or "(no relevant extracts found)"

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT.format(organisation=settings.organisation, context=context)
        ),
        *to_messages(history),
        HumanMessage(content=question),
    ]
    partial = ""
    for piece in settings.chat_model(temperature=0.0).stream(messages):
        partial += piece.content or ""
        yield partial, trace.chunks
