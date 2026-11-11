"""Turns texts into vectors (embeddings) with the Gemini API."""
from google.genai import types

from app import config
from app.gemini_client import get_gemini_client

BATCH_SIZE = 50  # number of texts sent per request


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Return one vector per text.

    task_type = "RETRIEVAL_DOCUMENT" for document chunks,
                "RETRIEVAL_QUERY" for user questions.
    """
    client = get_gemini_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=config.EMBEDDING_DIM,
            ),
        )
        vectors.extend(embedding.values for embedding in response.embeddings)
    return vectors


def embed_query(question: str) -> list[float]:
    return embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
