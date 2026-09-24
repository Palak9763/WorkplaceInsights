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

## Next steps

Once this looks right on the sample chunks, swap `SAMPLE_CHUNKS` in
`run_ingestion.py` for your actual documents (or wire up a real
ingestion layer with connectors/parsers/chunker for Confluence, Slack,
etc.). After that, the next piece to build is Query Orchestration —
taking a live question and routing it to graph, vector, or both. Ask
when you're ready for that one.
