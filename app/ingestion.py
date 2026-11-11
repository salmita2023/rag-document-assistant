"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> Qdrant."""
import uuid
from pathlib import Path

from pypdf import PdfReader
from qdrant_client.models import PointStruct

from app import config
from app.embeddings import embed_texts
from app.vector_store import ensure_collection, get_qdrant_client


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Read a PDF and return a list of (page number, text)."""
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))
    return pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a text into overlapping chunks of chunk_size characters."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def ingest_pdf(path: Path) -> int:
    """Index a PDF in Qdrant and return the number of chunks created."""
    ensure_collection()
    chunks, payloads = [], []
    for page_number, page_text in load_pdf(path):
        for index, chunk in enumerate(split_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(chunk)
            payloads.append({"source": path.name, "page": page_number, "chunk": index, "text": chunk})

    if not chunks:
        return 0

    vectors = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
    points = [
        PointStruct(
            # Stable id: re-indexing the same PDF overwrites its chunks instead of duplicating them
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{p['source']}-{p['page']}-{p['chunk']}")),
            vector=vector,
            payload=p,
        )
        for vector, p in zip(vectors, payloads)
    ]
    get_qdrant_client().upsert(collection_name=config.COLLECTION_NAME, points=points)
    return len(points)


if __name__ == "__main__":
    pdf_files = sorted(Path(config.DATA_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF found in {config.DATA_DIR}/")
    for pdf in pdf_files:
        print(f"{pdf.name}: {ingest_pdf(pdf)} chunks indexed")
