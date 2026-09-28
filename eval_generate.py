"""
Samples the live graph (Neo4j) and Qdrant chunks to draft eval_questions.json.

Run:  python eval_generate.py
Output: eval_questions.json  (review and correct before running eval.py)

Question categories generated:
  - graph:    neighborhood / path questions anchored to real nodes
  - vector:   questions targeting specific document text
  - hybrid:   questions that need both graph and document context
  - unanswerable: questions about entities that do not exist in the data
"""
import json
import random
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    GRAPH_MAX_ROWS,
)

OUTPUT_FILE = "eval_questions.json"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neo4j_run(cypher: str, params: dict | None = None) -> list[dict]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows = []
    try:
        with driver.session() as session:
            for rec in session.run(cypher, **(params or {})):
                rows.append(dict(rec))
    finally:
        driver.close()
    return rows


def _sample_nodes(label: str, limit: int = 8) -> list[dict]:
    return _neo4j_run(
        f"MATCH (n:`{label}`) WHERE n.name IS NOT NULL "
        f"RETURN n.name AS name, labels(n) AS labels LIMIT {limit}"
    )


def _sample_edges(limit: int = 20) -> list[dict]:
    return _neo4j_run(
        f"MATCH (a)-[r]->(b) WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
        f"RETURN a.name AS a, type(r) AS rel, b.name AS b LIMIT {limit}"
    )


def _sample_chunks(limit: int = 10) -> list[dict]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=limit,
            with_payload=True,
        )
        return [dict(r.payload) for r in results]
    except Exception as exc:
        print(f"[eval_gen] Qdrant scroll error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Question drafters
# ---------------------------------------------------------------------------

def _graph_questions(edges: list[dict], node_samples: dict[str, list[dict]]) -> list[dict]:
    qs = []
    seen = set()
    for edge in edges:
        a, rel, b = edge["a"], edge["rel"], edge["b"]
        key = (a.lower(), rel)
        if key in seen:
            continue
        seen.add(key)
        q = {
            "id": f"graph_{len(qs)+1}",
            "category": "graph",
            "question": f"What is {a} connected to via {rel}?",
            "expected_facts": [f"{a} -[{rel}]-> {b}"],
            "unanswerable": False,
        }
        qs.append(q)
        if len(qs) >= 5:
            break
    return qs


def _vector_questions(chunks: list[dict]) -> list[dict]:
    qs = []
    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "")
        if len(text) < 20:
            continue
        # Draft a question from the first sentence of the chunk
        first_sentence = text.split(".")[0].strip()
        q = {
            "id": f"vector_{len(qs)+1}",
            "category": "vector",
            "question": f"According to the documents, what does the following refer to: '{first_sentence[:80]}'?",
            "expected_facts": [first_sentence[:120]],
            "source": source,
            "unanswerable": False,
        }
        qs.append(q)
        if len(qs) >= 4:
            break
    return qs


def _hybrid_questions(edges: list[dict], chunks: list[dict]) -> list[dict]:
    qs = []
    for edge in random.sample(edges, min(3, len(edges))):
        a, rel, b = edge["a"], edge["rel"], edge["b"]
        q = {
            "id": f"hybrid_{len(qs)+1}",
            "category": "hybrid",
            "question": f"What do the documents say about {a}, and what graph relationships does it have?",
            "expected_facts": [f"{a} -[{rel}]-> {b}"],
            "unanswerable": False,
        }
        qs.append(q)
    return qs


def _unanswerable_questions() -> list[dict]:
    return [
        {
            "id": "unanswerable_1",
            "category": "unanswerable",
            "question": "What is the quarterly revenue of the Martian subsidiary?",
            "expected_facts": [],
            "unanswerable": True,
        },
        {
            "id": "unanswerable_2",
            "category": "unanswerable",
            "question": "Who is the CEO of ZetaCorp International?",
            "expected_facts": [],
            "unanswerable": True,
        },
        {
            "id": "unanswerable_3",
            "category": "unanswerable",
            "question": "What happened during the XK-9 protocol breach?",
            "expected_facts": [],
            "unanswerable": True,
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[eval_gen] Sampling Neo4j graph...")
    edges = _sample_edges(limit=30)
    labels_result = _neo4j_run("CALL db.labels()")
    labels = [r.get("label") for r in labels_result if r.get("label")]
    node_samples = {}
    for lbl in labels:
        node_samples[lbl] = _sample_nodes(lbl, limit=6)

    print(f"[eval_gen] Got {len(edges)} edges, {len(labels)} labels.")

    print("[eval_gen] Sampling Qdrant chunks...")
    chunks = _sample_chunks(limit=10)
    print(f"[eval_gen] Got {len(chunks)} chunks.")

    questions = (
        _graph_questions(edges, node_samples)
        + _vector_questions(chunks)
        + _hybrid_questions(edges, chunks)
        + _unanswerable_questions()
    )

    # Number them sequentially
    for i, q in enumerate(questions, 1):
        q["id"] = f"q{i:02d}"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"[eval_gen] Wrote {len(questions)} questions to {OUTPUT_FILE}")
    print("REVIEW the file and correct expected_facts before running eval.py")
    for q in questions:
        print(f"  [{q['category']}] {q['question']}")
        if q["expected_facts"]:
            print(f"          expect: {q['expected_facts'][0][:80]}")


if __name__ == "__main__":
    main()
