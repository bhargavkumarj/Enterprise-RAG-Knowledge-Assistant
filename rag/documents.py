from pathlib import Path

from langchain_core.documents import Document

from rag.config import settings


def load_knowledge_base(root: Path | None = None) -> list[Document]:
    """Read every markdown file under the knowledge base, one Document per file.

    The immediate parent folder becomes `doc_type` (company, products,
    contracts, employees) so retrieval can be filtered or reported per source.
    """
    root = root or settings.knowledge_base
    if not root.exists():
        raise FileNotFoundError(f"Knowledge base not found at {root}")

    documents = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.stem,
                    "path": str(path.relative_to(root)),
                    "doc_type": path.parent.name,
                },
            )
        )
    return documents


def summarise(documents: list[Document]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in documents:
        doc_type = doc.metadata.get("doc_type", "unknown")
        counts[doc_type] = counts.get(doc_type, 0) + 1
    return counts
