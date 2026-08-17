"""
Stage 1 building block: turn raw text into overlapping chunks.

Why overlap? If a sentence that answers the user's question gets cut
in half at a chunk boundary, neither chunk alone carries full meaning.
Overlap reduces the chance a key sentence is orphaned.
"""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Simple sliding-window character chunker.

    chunk_size: max characters per chunk
    chunk_overlap: characters shared between consecutive chunks
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - chunk_overlap

    return chunks
