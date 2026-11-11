"""Retrieval + generation: find the relevant chunks, then ask the LLM to answer with citations."""
from google.genai import types

from app import config
from app.embeddings import embed_query
from app.gemini_client import get_gemini_client
from app.vector_store import get_qdrant_client

SYSTEM_PROMPT = """You are a document assistant.
Answer the question using ONLY the numbered context passages below.
- Cite the passages you use with their number, e.g. [1] or [2][3].
- If the answer is not in the context, say clearly that you could not find it in the documents.
- Answer in the same language as the question, concisely."""


def retrieve(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Return the top_k most similar chunks with their metadata and similarity score."""
    result = get_qdrant_client().query_points(
        collection_name=config.COLLECTION_NAME,
        query=embed_query(question),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "source": point.payload["source"],
            "page": point.payload["page"],
            "text": point.payload["text"],
            "score": round(point.score, 4),
        }
        for point in result.points
    ]


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {p['source']}, page {p['page']})\n{p['text']}" for i, p in enumerate(passages, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(question: str, passages: list[dict]) -> str:
    response = get_gemini_client().models.generate_content(
        model=config.LLM_MODEL,
        # Instructions are sent inside the prompt (system_instruction is not supported on every API version)
        contents=f"{SYSTEM_PROMPT}\n\n{build_prompt(question, passages)}",
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text or ""


def answer(question: str, top_k: int = config.TOP_K) -> dict:
    passages = retrieve(question, top_k)
    if not passages:
        return {"answer": "No documents are indexed yet. Upload a PDF first.", "sources": []}
    return {"answer": generate(question, passages), "sources": passages}
