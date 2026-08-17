from functools import lru_cache
from openai import OpenAI

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Same OpenRouter account/key used for the LLM, reused for embeddings."""
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def clear_client_cache() -> None:
    get_client.cache_clear()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Stage 1: text -> vector, for a batch of texts, via OpenRouter."""
    client = get_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
