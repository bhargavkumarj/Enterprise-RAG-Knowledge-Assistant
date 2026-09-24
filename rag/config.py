import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    knowledge_base: Path = ROOT / "data" / "knowledge_base"
    questions: Path = ROOT / "data" / "questions.jsonl"
    vector_store: Path = ROOT / "vector_store"
    collection: str = "knowledge_base"

    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "huggingface")
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    chat_backend: str = os.getenv("CHAT_BACKEND", "openai")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    judge_model: str = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")

    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    top_k: int = int(os.getenv("TOP_K", "6"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "20"))
    rewrite_query: bool = _flag("REWRITE_QUERY", True)
    rerank: bool = _flag("RERANK", True)
    reranker: str = os.getenv("RERANKER", "llm")
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    organisation: str = os.getenv("ORGANISATION", "Insurellm")

    _embeddings: object | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def embeddings(self):
        # A local sentence-transformers model takes seconds to load and is not safe
        # to build from several threads at once, so it is loaded once and shared.
        with self._lock:
            if self._embeddings is None:
                if self.embedding_backend == "openai":
                    from langchain_openai import OpenAIEmbeddings

                    self._embeddings = OpenAIEmbeddings(model=self.openai_embedding_model)
                else:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self._embeddings = HuggingFaceEmbeddings(model_name=self.hf_embedding_model)
            return self._embeddings

    def chat_model(self, temperature: float = 0.0, model: str | None = None):
        if self.chat_backend == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model or self.ollama_chat_model, temperature=temperature)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or self.openai_chat_model, temperature=temperature)

    def judge(self):
        return self.chat_model(temperature=0.0, model=self.judge_model)

    def fingerprint(self) -> str:
        """Identifies the embedding space. Queries can only be compared against a
        store built with the same one, so this is checked when the store opens."""
        model = (
            self.hf_embedding_model
            if self.embedding_backend == "huggingface"
            else self.openai_embedding_model
        )
        return f"{self.embedding_backend}:{model}"


settings = Settings()
