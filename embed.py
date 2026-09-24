"""
Real embedding via Ollama's nomic-embed-text model.
Replaces the fake random vector used in the earlier connection test script.
"""
import ollama
from config import OLLAMA_HOST, EMBEDDING_MODEL


def embed(text: str) -> list[float]:
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]
