"""Build the vector store from the knowledge base."""

import argparse

from rag import store
from rag.chunking import STRATEGIES, chunk_documents
from rag.config import settings
from rag.documents import load_knowledge_base, summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=settings.chunk_strategy)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    settings.chunk_strategy = args.strategy
    settings.chunk_size = args.chunk_size
    settings.chunk_overlap = args.chunk_overlap

    documents = load_knowledge_base()
    print(f"Loaded {len(documents)} documents: {summarise(documents)}")

    chunks = chunk_documents(documents, args.strategy)
    lengths = sorted(len(chunk.page_content) for chunk in chunks)
    median = lengths[len(lengths) // 2]
    print(f"{args.strategy} chunking produced {len(chunks)} chunks (median {median} chars)")

    store.build(chunks)
    print(f"Embedded with {settings.fingerprint()} into {settings.vector_store}")


if __name__ == "__main__":
    main()
