import json
import shutil
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import settings

MANIFEST = "manifest.json"


def build(chunks: list[Document], reset: bool = True) -> Chroma:
    if reset and settings.vector_store.exists():
        shutil.rmtree(settings.vector_store)
    settings.vector_store.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=settings.embeddings(),
        collection_name=settings.collection,
        persist_directory=str(settings.vector_store),
    )
    (settings.vector_store / MANIFEST).write_text(
        json.dumps(
            {
                "fingerprint": settings.fingerprint(),
                "chunks": len(chunks),
                "chunk_strategy": settings.chunk_strategy,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            indent=2,
        )
    )
    return store


@lru_cache(maxsize=1)
def open_store() -> Chroma:
    manifest_path = settings.vector_store / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError("No vector store yet — run `python ingest.py` first.")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("fingerprint") != settings.fingerprint():
        raise RuntimeError(
            "The vector store was built with a different embedding model "
            f"({manifest.get('fingerprint')} vs {settings.fingerprint()}). Re-run ingest.py."
        )

    return Chroma(
        collection_name=settings.collection,
        persist_directory=str(settings.vector_store),
        embedding_function=settings.embeddings(),
    )


def stats() -> dict:
    store = open_store()
    manifest = json.loads((settings.vector_store / MANIFEST).read_text())
    manifest["vectors"] = store._collection.count()
    return manifest
