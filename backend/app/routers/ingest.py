from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.services.extraction import extract_text
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.qdrant_store import upsert_chunks

router = APIRouter()


@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """
    Full ingestion pipeline:
    file -> extract text -> chunk -> embed -> store in Qdrant
    """
    try:
        raw_bytes = await file.read()
        text = extract_text(file.filename, raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file.")

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text produced zero chunks.")

    vectors = embed_texts(chunks)
    count = upsert_chunks(chunks, vectors, source=file.filename)

    return {
        "filename": file.filename,
        "chunks_created": count,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
