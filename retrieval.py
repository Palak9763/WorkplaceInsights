"""
Graph and vector retrieval with:
  - Primary path: LLM-generated Cypher via graph_retrieve()
  - Fallback:     anchor neighborhood expansion (no LLM, no hardcoded patterns)
  - Fact ranking: graph sentences re-ranked by cosine similarity to the question
  - Vector path:  Qdrant semantic search with configurable score threshold
"""
import math
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path
from qdrant_client import QdrantClient

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    GRAPH_MAX_ROWS, GRAPH_RANK_TOP_K, GRAPH_RANK_MIN_SCORE,
    VECTOR_TOP_K, VECTOR_MIN_SCORE,
)
from embed import embed


# ---------------------------------------------------------------------------
# Neo4j value serialiser
# ---------------------------------------------------------------------------
def _serialize(val):
    if isinstance(val, Node):
        return {"id": val.id, "labels": list(val.labels), "properties": dict(val.items())}
    if isinstance(val, Relationship):
        return {"id": val.id, "type": val.type, "properties": dict(val.items())}
    if isinstance(val, Path):
        return {
            "nodes": [_serialize(n) for n in val.nodes],
            "relationships": [_serialize(r) for r in val.relationships],
        }
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_serialize(v) for v in val]
    return val


def _run_cypher(cypher: str, params: dict | None = None) -> tuple[list[dict], str | None]:
    """Runs cypher, returns (rows, error_or_None). Never silently swallows errors."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows: list[dict] = []
    error: str | None = None
    try:
        with driver.session() as session:
            result = session.run(cypher, **(params or {}))
            for record in result:
                rows.append({k: _serialize(record[k]) for k in record.keys()})
    except Exception as exc:
        error = f"Neo4j query error: {exc}"
        print(f"[retrieval] {error}")
    finally:
        driver.close()
    return rows, error


# ---------------------------------------------------------------------------
# Row → sentence (stored direction via a_is_source flag)
# ---------------------------------------------------------------------------
def row_to_sentence(row: dict) -> str:
    """
    Converts a graph result row to a directed sentence.
    Handles both the anchor-fallback schema and flat LLM-query schemas.
    """
    if "a_is_source" in row:
        # Anchor fallback row: {a_name, rel, b_name, a_is_source}
        a = row.get("a_name") or "?"
        rel = row.get("rel") or row.get("relationship") or "?"
        b = row.get("b_name") or row.get("target") or "?"
        if row["a_is_source"]:
            return f"{a} -[{rel}]-> {b}"
        else:
            return f"{b} -[{rel}]-> {a}"
    elif "relationship" in row:
        # Standard LLM query row: {name, relationship, target, ...}
        subj = row.get("name") or "?"
        rel = row.get("relationship") or "?"
        tgt = row.get("target") or "?"
        return f"{subj} -[{rel}]-> {tgt}"
    else:
        # Generic: flatten to key=value pairs
        return "  ".join(f"{k}={v}" for k, v in row.items() if v is not None)


# ---------------------------------------------------------------------------
# Cosine similarity + fact ranking
# ---------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def rank_graph_facts(
    rows: list[dict], question: str
) -> tuple[list[dict], list[float]]:
    """
    Embeds each graph row as a sentence, computes cosine similarity to the
    question embedding, and returns (ranked_rows, scores) sorted desc.
    Applies GRAPH_RANK_TOP_K and GRAPH_RANK_MIN_SCORE from config.
    """
    if not rows:
        return [], []

    sentences = [row_to_sentence(r) for r in rows]
    try:
        q_vec = embed(question)
        sent_vecs = [embed(s) for s in sentences]
        scores = [_cosine(q_vec, v) for v in sent_vecs]
    except Exception as exc:
        print(f"[retrieval] Embedding error during fact ranking: {exc}")
        # Return all rows unranked on embedding failure
        return rows, [0.0] * len(rows)

    paired = sorted(zip(scores, rows, sentences), key=lambda x: x[0], reverse=True)
    result_rows, result_scores = [], []
    for score, row, sentence in paired:
        if score < GRAPH_RANK_MIN_SCORE:
            break
        row["_sentence"] = sentence
        row["_rank_score"] = round(score, 4)
        result_rows.append(row)
        result_scores.append(score)
        if len(result_rows) >= GRAPH_RANK_TOP_K:
            break

    print(f"[retrieval] Fact ranking: {len(rows)} rows -> {len(result_rows)} kept "
          f"(top score={result_scores[0]:.4f} min_threshold={GRAPH_RANK_MIN_SCORE})")
    return result_rows, result_scores


# ---------------------------------------------------------------------------
# Primary graph retrieval (LLM Cypher)
# ---------------------------------------------------------------------------
def graph_retrieve(cypher: str) -> tuple[list[dict], str | None]:
    """Returns (rows, error_or_None)."""
    print("==================================================")
    print("GRAPH RETRIEVAL (LLM Cypher)")
    print("==================================================")
    rows, error = _run_cypher(cypher)
    print(f"Raw row count: {len(rows)}")
    if rows:
        print(f"Sample row: {rows[0]}")
    print()
    return rows, error


# ---------------------------------------------------------------------------
# Anchor-neighborhood fallback (no LLM, no hardcoded patterns)
# ---------------------------------------------------------------------------
_ANCHOR_FALLBACK_CYPHER = """
MATCH (a)-[r]-(b)
WHERE elementId(a) IN $anchor_ids
RETURN DISTINCT
    elementId(r)      AS rid,
    a.name            AS a_name,
    labels(a)         AS a_labels,
    type(r)           AS rel,
    b.name            AS b_name,
    labels(b)         AS b_labels,
    startNode(r) = a  AS a_is_source
LIMIT $max_rows
"""


def graph_fallback_retrieve(anchors: list[dict]) -> tuple[list[dict], str | None]:
    """
    Expands the neighborhood of each anchor node.
    Deduplicates by edge elementId (rid) so mirrored rows vanish.
    Returns (rows, error_or_None). Skips cleanly if no anchors.
    """
    print("==================================================")
    print("GRAPH RETRIEVAL (Anchor Fallback)")
    print("==================================================")

    if not anchors:
        print("[fallback] No anchors — skipping.")
        print()
        return [], None

    anchor_ids = [a["element_id"] for a in anchors if a.get("element_id")]
    if not anchor_ids:
        print("[fallback] Anchors have no element_ids (n-gram fallback was used). "
              "Trying name-based pattern.")
        # Names-based fallback when element_ids are unavailable
        name_conditions = " OR ".join(
            f"toLower(a.name) CONTAINS toLower('{a['name']}')"
            for a in anchors
        )
        name_cypher = f"""
MATCH (a)-[r]-(b)
WHERE {name_conditions}
RETURN DISTINCT
    elementId(r)     AS rid,
    a.name           AS a_name,
    labels(a)        AS a_labels,
    type(r)          AS rel,
    b.name           AS b_name,
    labels(b)        AS b_labels,
    startNode(r) = a AS a_is_source
LIMIT {GRAPH_MAX_ROWS}
"""
        rows, error = _run_cypher(name_cypher)
    else:
        rows, error = _run_cypher(
            _ANCHOR_FALLBACK_CYPHER,
            {"anchor_ids": anchor_ids, "max_rows": GRAPH_MAX_ROWS},
        )

    print(f"Fallback raw row count: {len(rows)}")
    if rows:
        print(f"Sample row: {rows[0]}")
    print()
    return rows, error


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------
def vector_retrieve(question: str) -> tuple[list[dict], str | None]:
    """Returns (chunks, error_or_None)."""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    results: list[dict] = []
    error: str | None = None

    try:
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            error = f"Qdrant collection '{QDRANT_COLLECTION}' not found."
            print(f"[vector] {error}")
        else:
            query_vector = embed(question)
            print(f"[vector] Embedded question, vector length={len(query_vector)}")
            hits = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=VECTOR_TOP_K,
                score_threshold=VECTOR_MIN_SCORE if VECTOR_MIN_SCORE > 0 else None,
            )
            print(f"[vector] Qdrant returned {len(hits)} hit(s) above score threshold {VECTOR_MIN_SCORE}")
            for hit in hits:
                payload = dict(hit.payload) if hit.payload else {}
                payload["score"] = hit.score
                results.append(payload)
    except Exception as exc:
        error = f"Qdrant error: {exc}"
        print(f"[vector] ERROR: {exc}")

    print("==================================================")
    print("VECTOR RETRIEVAL")
    print("==================================================")
    print(f"Chunks returned: {len(results)}")
    print()
    return results, error
