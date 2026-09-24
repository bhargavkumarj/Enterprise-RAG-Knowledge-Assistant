from pydantic import BaseModel, Field

from rag.config import settings

PROMPT = """You are grading an internal knowledge assistant against a reference answer.

Question: {question}

Reference answer: {reference}

Assistant's answer: {answer}

Score each dimension from 1 to 5:
- accuracy: does it contradict the reference or invent facts? A wrong fact scores 1.
- completeness: does it cover everything the reference covers?
- groundedness: is every claim supported by the reference, with no unsupported detail?

Be strict. Reserve 5 for answers a domain expert would sign off on unchanged."""


class Verdict(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    comment: str = Field(description="One sentence explaining the scores")

    @property
    def mean(self) -> float:
        return (self.accuracy + self.completeness + self.groundedness) / 3


def judge(question: str, answer: str, reference: str) -> Verdict:
    model = settings.judge().with_structured_output(Verdict)
    return model.invoke(PROMPT.format(question=question, reference=reference, answer=answer))
