"""
Shared config loader.
All scripts import from here so connection details live in one place (.env).
Query pipeline tunables also live here so nothing is hardcoded inline.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-password-here")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "graphrag_chunks")

# --- Ollama ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "qwen2.5:3b-instruct")
REASONING_MODEL = os.getenv("REASONING_MODEL", EXTRACTION_MODEL)

# --- Query pipeline tunables ---
# Schema cache time-to-live (seconds)
SCHEMA_CACHE_TTL = int(os.getenv("SCHEMA_CACHE_TTL", "30"))

# Number of sample node names fetched per label for schema enrichment
SCHEMA_SAMPLE_NAMES_PER_LABEL = int(os.getenv("SCHEMA_SAMPLE_NAMES_PER_LABEL", "5"))

# Entity linker: top-N anchor nodes returned from full-text / n-gram search
ENTITY_LINK_TOP_K = int(os.getenv("ENTITY_LINK_TOP_K", "5"))

# Entity linker: minimum full-text score to accept an anchor
ENTITY_LINK_MIN_SCORE = float(os.getenv("ENTITY_LINK_MIN_SCORE", "1.0"))

# Cypher: maximum rows returned by any graph query
GRAPH_MAX_ROWS = int(os.getenv("GRAPH_MAX_ROWS", "25"))

# Cypher: number of LLM generation retries on validation failure
CYPHER_RETRIES = int(os.getenv("CYPHER_RETRIES", "1"))

# --- Retrieval upgrade tunables ---
# Anchor forces hybrid routing
ANCHOR_FORCES_HYBRID = os.getenv("ANCHOR_FORCES_HYBRID", "true").lower() == "true"

# Graph traversal tunables
GRAPH_MAX_DEPTH = int(os.getenv("GRAPH_MAX_DEPTH", "1"))
GRAPH_MAX_PATH_LEN = int(os.getenv("GRAPH_MAX_PATH_LEN", "3"))
GRAPH_MAX_FANOUT = int(os.getenv("GRAPH_MAX_FANOUT", "10"))
GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", "10"))
GRAPH_REL_CUTOFF = float(os.getenv("GRAPH_REL_CUTOFF", "0.75"))
GRAPH_STAGE_TIMEOUT = float(os.getenv("GRAPH_STAGE_TIMEOUT", "5.0"))

# Vector semantic retrieval tunables
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))
VECTOR_MIN_SCORE = float(os.getenv("VECTOR_MIN_SCORE", "0.0"))
RRF_K = int(os.getenv("RRF_K", "60"))
VECTOR_REL_CUTOFF = float(os.getenv("VECTOR_REL_CUTOFF", "0.5"))
VECTOR_MMR_LAMBDA = float(os.getenv("VECTOR_MMR_LAMBDA", "0.7"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))