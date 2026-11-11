"""REST API (FastAPI): upload PDFs and ask questions."""
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import config, rag
from app.ingestion import ingest_pdf
from app.vector_store import ensure_collection


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_collection()
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    yield


app = FastAPI(title="RAG Document Assistant", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What are the obligations for high-risk AI systems?"])
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class Source(BaseModel):
    source: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile) -> dict:
    filename = Path(file.filename or "").name  # drop any folder path sent by the client
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    destination = Path(config.DATA_DIR) / filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    chunks = ingest_pdf(destination)
    if chunks == 0:
        raise HTTPException(status_code=422, detail="No text could be extracted (scanned PDF?).")
    return {"filename": filename, "chunks_indexed": chunks}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    return rag.answer(request.question, request.top_k)
