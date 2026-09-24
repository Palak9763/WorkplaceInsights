"""
Runs the full ingestion pipeline end to end:
  text chunk -> LLM extraction -> Neo4j (entities/relationships)
             -> embedding -> Qdrant (chunk + vector)

This is Layer 2+3+4+5a+5b from our architecture, in one script.
Uses the Priya/Atlas example chunks we've been using throughout, plus
one new chunk to show the graph accumulating knowledge across documents.

Run:  python run_ingestion.py
"""
from extract import extract
from write_graph import write_to_graph
from write_vector import write_to_vector_store

# Replace this with real chunks from your ingestion layer (Confluence,
# Slack, etc.) once this pipeline is proven on known-good examples.
SAMPLE_CHUNKS = [
    {
        "text": (
            "Priya Sharma (London) led the backend migration for Project Atlas. "
            "The project uses PostgreSQL as its primary database."
        ),
        "source": "Confluence/Project-Atlas-Q3",
        "date": "2026-08-15",
    },
    {
        "text": (
            "In August, the Auth Service outage delayed the launch of Project Atlas by two weeks."
        ),
        "source": "Confluence/Project-Atlas-Q3",
        "date": "2026-08-15",
    },
    {
        "text": (
            "The Auth Service had a 3-hour outage on September 18, which caused a delay "
            "to Project Atlas by pushing the launch to the following Friday."
        ),
        "source": "Slack/#eng-updates",
        "date": "2026-09-18",
    },
]


def main():
    print(f"Running ingestion on {len(SAMPLE_CHUNKS)} chunk(s)...\n")

    for i, chunk in enumerate(SAMPLE_CHUNKS, start=1):
        print(f"--- Chunk {i}/{len(SAMPLE_CHUNKS)} (source: {chunk['source']}) ---")

        # 1. Extraction (Layer 4)
        result = extract(chunk["text"], chunk_id=f"chunk-{i}")

        if result is None:
            print("  Skipping graph write - extraction failed.\n")
        else:
            # 2. Write to Neo4j (Layer 5a)
            write_to_graph(result, source=chunk["source"])

        # 3. Write to Qdrant (Layer 5b) - happens regardless of extraction
        #    success, since the raw text is still useful for vector search
        #    even if structured extraction failed on this chunk
        write_to_vector_store(chunk["text"], source=chunk["source"], date=chunk["date"])

        print()

    print("Ingestion complete.")
    print("\nCheck your results:")
    print("  Neo4j Browser (Desktop) -> run: MATCH (n) RETURN n LIMIT 50")
    print("  Qdrant Dashboard -> https://cloud.qdrant.io (open your cluster, then Dashboard tab)")


if __name__ == "__main__":
    main()
