import concurrent.futures as futures

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.config import settings

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

CONTEXT_PROMPT = """You are indexing a document for a company knowledge base.

Document: {source} ({doc_type})

<document>
{document}
</document>

Here is one chunk taken from that document:

<chunk>
{chunk}
</chunk>

Write one or two sentences that situate this chunk inside the wider document, so
that it can be retrieved on its own. Mention the entities the chunk refers to by
name. Respond with the sentences only."""


def recursive_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documents)


def markdown_chunks(documents: list[Document]) -> list[Document]:
    """Split on markdown headings first, then size-limit the sections.

    Contracts and employee records are heavily sectioned, so heading-aware
    splitting keeps clauses intact instead of cutting them mid-sentence.
    """
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    sections = []
    for doc in documents:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata = {**doc.metadata, **section.metadata}
            sections.append(section)
    return size_splitter.split_documents(sections)


def _describe(args) -> Document:
    chunk, parent = args
    model = settings.chat_model(temperature=0.0)
    prompt = CONTEXT_PROMPT.format(
        source=parent.metadata.get("source", "unknown"),
        doc_type=parent.metadata.get("doc_type", "unknown"),
        document=parent.page_content[:6000],
        chunk=chunk.page_content,
    )
    context = model.invoke(prompt).content.strip()
    return Document(
        page_content=f"{context}\n\n{chunk.page_content}",
        metadata={**chunk.metadata, "contextualised": True},
    )


def contextual_chunks(documents: list[Document], workers: int = 8) -> list[Document]:
    """Anthropic-style contextual retrieval: prepend an LLM-written situating
    sentence to every chunk before it is embedded. Costs one LLM call per chunk."""
    by_source = {doc.metadata["path"]: doc for doc in documents}
    base = markdown_chunks(documents)
    pairs = [(chunk, by_source[chunk.metadata["path"]]) for chunk in base]

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_describe, pairs))


STRATEGIES = {
    "recursive": recursive_chunks,
    "markdown": markdown_chunks,
    "contextual": contextual_chunks,
}


def chunk_documents(documents: list[Document], strategy: str | None = None) -> list[Document]:
    strategy = strategy or settings.chunk_strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunk strategy {strategy!r}; pick one of {list(STRATEGIES)}")
    chunks = STRATEGIES[strategy](documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
