from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import apply_runtime_settings, settings
from app.routers import ingest, query
from app.services.embeddings import clear_client_cache as clear_embedding_client
from app.services.llm import clear_client_cache as clear_llm_client
from app.services.qdrant_store import clear_client_cache as clear_qdrant_client

app = FastAPI(title="RAG Lab API")

# Frontend is a plain static HTML file opened directly in the browser
# (or served separately), so we allow all origins for this local lab.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)


class RuntimeSettingsPayload(BaseModel):
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    top_k: Optional[int] = None


@app.get("/")
async def root():
    return {"Greeting": "Hello BUDDY ON GENERAL PURPOSE RAG!"}

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def read_config():
    return {
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "top_k": settings.top_k,
        "llm_api_key_present": bool(settings.llm_api_key),
    }


@app.post("/config")
async def update_config(payload: RuntimeSettingsPayload):
    try:
        updates = payload.model_dump(exclude_none=True)
        config = apply_runtime_settings(updates)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    clear_embedding_client()
    clear_llm_client()
    clear_qdrant_client()

    return {
        "status": "updated",
        "config": config,
    }
