# Extraction Pipeline — Step 2

Builds on your verified Neo4j Desktop + Qdrant connections. This is the
actual ingestion pipeline: real text -> LLM extraction -> both stores.

## What's in here

```
graphrag-extraction/
├── .env.example       # same values as your connection test - reuse them
├── requirements.txt
├── config.py            # connection settings (Neo4j Desktop local, Qdrant local)
├── schema.py            # the entity/relationship JSON schema (pydantic)
├── extract.py            # calls qwen2.5:3b-instruct, validates JSON, retries on failure
├── embed.py              # real embeddings via nomic-embed-text (replaces the fake test vector)
├── write_graph.py         # writes extracted entities/relationships into Neo4j
├── write_vector.py        # writes chunks + embeddings into Qdrant
├── run_ingestion.py       # the main script - runs everything end to end
└── README.md
```

## Prereqs (you already have these)

- Neo4j Desktop running, database "Active"
- Qdrant running locally on port 6333
- Ollama running, with both models pulled:
  ```
  ollama pull qwen2.5:3b-instruct
  ollama pull nomic-embed-text
  ```

## Setup

```bash
cd graphrag-extraction
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` — same `NEO4J_PASSWORD` you used for the connection test.

## Run it

```bash
python run_ingestion.py
```

This runs 3 sample text chunks (the Priya/Atlas example, plus a new
September outage chunk) through the full pipeline:

1. **Extraction** — each chunk goes to `qwen2.5:3b-instruct`, gets
   parsed into entities + relationships, retries up to 3 times if the
   JSON is malformed
2. **Graph write** — entities/relationships get `MERGE`d into Neo4j
   (not `CREATE`d — so running this twice doesn't duplicate nodes)
3. **Vector write** — each chunk's raw text gets a real embedding via
   `nomic-embed-text` and is upserted into Qdrant

Watch the terminal output — you'll see exactly what each chunk
extracted, and whether any attempt needed a retry.

## Verify the results

**In Neo4j Desktop**, open the Browser tab and run:
```cypher
MATCH (n) RETURN n LIMIT 50
```
You should see Person/Project/Technology/Service/Incident nodes,
connected by WORKS_ON/USES/CAUSED_DELAY_TO/INVOLVED relationships —
including the Auth Service node now having TWO incidents connected to
it (August and September), since chunk 3 added a second outage.

**In Qdrant**, open http://localhost:6333/dashboard and check the
`graphrag_chunks` collection — you should see 3 points, each with real
768-dimension embeddings and the source/date metadata.

## Query Orchestration Pipeline

The query pipeline dynamically discovers graph schema and routes user questions to Cypher graph traversal, vector retrieval, or hybrid synthesis with anti-hallucination fact checking.

### Architecture Overview

```
User Query
    │
    ▼
[query_classifier.py] ──> Decides route: "graph" | "vector" | "hybrid"
    │
    ├── (Graph Route) ──> [entity_linker.py] ──> Discovers anchor nodes via full-text / n-gram
    │                     [cypher_generator.py] ──> Generates Cypher using live Neo4j schema
    │                     [cypher_validator.py] ──> Validates AST/schema labels/safe read-only
    │                     [retrieval.py] ──> Executes Cypher against Neo4j with fallback
    │
    ├── (Vector Route) ──> [retrieval.py] ──> Embeds query & searches Qdrant
    │
    └── (Reasoning / Synthesizer) ──> [reasoning.py]
                                      ├── Synthesizes context into natural answer
                                      └── Anti-hallucination check (flags unsupported facts)
```

### Key Modules

- `graph_schema.py`: Dynamic schema discovery via `CALL db.labels()`, `CALL db.relationshipTypes()`, and `db.schema.visualization()`. Cached with configurable TTL.
- `entity_linker.py`: Schema-agnostic anchor discovery using Neo4j full-text indexes and n-gram substring searches.
- `query_classifier.py`: Dynamic few-shot routing based on live schema relationships and node samples.
- `cypher_generator.py`: Generates safe read-only Cypher queries strictly conforming to live schema.
- `cypher_validator.py`: Enforces read-only safety, valid labels/rel types, and row limit constraints.
- `retrieval.py`: Orchestrates dual retrieval (Neo4j graph traversal + Qdrant vector search).
- `reasoning.py`: LLM synthesis and hallucination detection against retrieved facts.
- `ingest_api.py`: FastAPI server exposing `/query`, `/ingest/raw`, and `/schema` endpoints.
- `eval.py`: Automated evaluation harness testing graph, vector, and hybrid questions.

### Running the API & Evaluation

1. **Start the API Server**:
   ```bash
   uvicorn ingest_api:app --reload --port 8000
   ```

2. **Query the API**:
   ```bash
   curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What technologies are used in our projects?"}'
   ```

3. **Run Evaluation Harness**:
   ```bash
   python eval.py --questions eval_questions.json --url http://127.0.0.1:8000
   ```

