"""
Shared config loader.
All scripts import from here so connection details live in one place (.env).

This version assumes LOCAL setup:
- Neo4j Desktop running on bolt://localhost:7687
- Qdrant running locally on localhost:6333
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Neo4j (Neo4j Desktop, local) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-password-here")

# --- Qdrant (local) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "graphrag_chunks")

# --- Ollama ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "qwen2.5:3b-instruct")
