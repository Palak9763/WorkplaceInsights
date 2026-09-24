"""
Shared config loader.
All scripts import from here so connection details live in one place (.env).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Neo4j (Neo4j Desktop, local) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-password-here")

# --- Qdrant Cloud ---
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "graphrag_chunks")

# --- Ollama ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "qwen2.5:3b-instruct")