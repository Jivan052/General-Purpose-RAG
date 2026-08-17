from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant — embedded mode, no server/Docker needed.
    qdrant_path: str = "./qdrant_data"
    qdrant_collection: str = "rag_documents"

    # Embeddings
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100

    # LLM + embeddings both go through OpenRouter's OpenAI-compatible API.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b:free"

    # Retrieval
    top_k: int = 5

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()


def apply_runtime_settings(values: dict) -> dict:
    """Apply a small set of runtime overrides and return the effective config."""
    if "llm_base_url" in values and values["llm_base_url"]:
        settings.llm_base_url = str(values["llm_base_url"]).strip()
    if "llm_api_key" in values and values["llm_api_key"] is not None:
        settings.llm_api_key = str(values["llm_api_key"]).strip()
    if "llm_model" in values and values["llm_model"]:
        settings.llm_model = str(values["llm_model"]).strip()
    if "embedding_model" in values and values["embedding_model"]:
        settings.embedding_model = str(values["embedding_model"]).strip()
    if "embedding_dim" in values and values["embedding_dim"] is not None:
        settings.embedding_dim = int(values["embedding_dim"])
    if "top_k" in values and values["top_k"] is not None:
        settings.top_k = int(values["top_k"])

    return {
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "top_k": settings.top_k,
    }
