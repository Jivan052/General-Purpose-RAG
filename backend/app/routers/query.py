import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.services.embeddings import embed_query
from app.services.qdrant_store import search
from app.services.llm import generate_answer

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


@router.post("/query")
async def query(req: QueryRequest):
    """
    Full RAG pipeline:
    question -> embed -> Qdrant search -> top-k chunks -> LLM -> answer
    """
    top_k = req.top_k or settings.top_k
    retrieval_start = time.perf_counter()

    query_vector = embed_query(req.question)
    retrieved = search(query_vector, top_k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    if not retrieved:
        return {
            "answer": "No documents have been ingested yet — upload one first.",
            "retrieved_chunks": [],
            "retrieval_ms": round(retrieval_ms, 2),
            "llm_response_ms": 0,
        }

    context_chunks = [r["text"] for r in retrieved]
    llm_start = time.perf_counter()
    answer = generate_answer(req.question, context_chunks)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    return {
        "answer": answer,
        "retrieved_chunks": retrieved,
        "retrieval_ms": round(retrieval_ms, 2),
        "llm_response_ms": round(llm_ms, 2),
    }
