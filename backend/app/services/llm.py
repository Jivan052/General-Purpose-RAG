from openai import OpenAI
from functools import lru_cache

from app.config import settings

RAG_SYSTEM_PROMPT = (
    "You answer using only the provided context. Keep the response brief, clear, "
    "and to the point. Prefer bullet points when helpful. Do not add information "
    "not in the context. If the answer is not present, say so briefly."
)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def clear_client_cache() -> None:
    get_client.cache_clear()


def sanitize_answer(answer: str) -> str:
    """Keep the LLM output compact and free of repeated lines."""
    cleaned_lines: list[str] = []
    seen: set[str] = set()

    for raw in answer.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = " ".join(line.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines[:5]).strip()
    if len(result) > 700:
        result = " ".join(result.split())[:700]
    return result


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Stage 4: retrieved chunks + question -> LLM -> answer."""
    context = "\n\n---\n\n".join(context_chunks)
    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer in 2-3 concise lines or up to 3 bullet points. Keep each line short and factual. "
        "No long explanation, no repeated points, no generic filler."
    )

    client = get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return sanitize_answer(response.choices[0].message.content or "")
