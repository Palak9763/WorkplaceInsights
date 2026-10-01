"""
Real embedding via Ollama's nomic-embed-text model.
Replaces the fake random vector used in the earlier connection test script.
"""
import ollama
from config import OLLAMA_HOST, EMBEDDING_MODEL

_CACHE: dict[str, list[float]] = {}
_CACHE_MAX = 512


def embed(text: str) -> list[float]:
    if text in _CACHE:
        return _CACHE[text]
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    vec = response["embedding"]
    if len(_CACHE) >= _CACHE_MAX:
        # evict oldest entry
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[text] = vec
    return vec
