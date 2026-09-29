"""
Entity linker: matches free-text questions against real graph nodes.

Primary path: Neo4j full-text index over the `name` property of all node labels.
  - Index is created at startup and rebuilt when the label set changes.
  - Queries with the raw question; returns top-K anchors above a score threshold.

Fallback (index unavailable): n-gram matching against cached node names from
the live schema samples. No stopword list — n-grams that match nothing are dropped.

Anchor shape:
  {"element_id": str, "name": str, "labels": list[str], "score": float}
"""
import re
from neo4j import GraphDatabase
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    ENTITY_LINK_TOP_K, ENTITY_LINK_MIN_SCORE,
)

_INDEX_NAME = "graphrag_entity_name_index"
_current_label_set: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _ensure_fulltext_index(session, labels: list[str]) -> bool:
    """
    Creates (or recreates) the full-text index when the label set changes.
    Returns True if the index is available.
    """
    global _current_label_set
    label_set = frozenset(labels)

    if not label_set:
        return False

    needs_rebuild = label_set != _current_label_set

    if needs_rebuild:
        # Drop old index if it exists
        try:
            session.run(f"DROP INDEX {_INDEX_NAME} IF EXISTS")
        except Exception as exc:
            print(f"[linker] Could not drop old index: {exc}")

        # Build label string for the index definition
        label_str = "|".join(f"`{lbl}`" for lbl in sorted(label_set))
        try:
            session.run(
                f"CREATE FULLTEXT INDEX {_INDEX_NAME} "
                f"IF NOT EXISTS FOR (n:{label_str}) ON EACH [n.name]"
            )
            _current_label_set = label_set
            print(f"[linker] Full-text index built for labels: {sorted(label_set)}")
            return True
        except Exception as exc:
            print(f"[linker] Could not create full-text index: {exc}")
            _current_label_set = frozenset()
            return False

    return True


# ---------------------------------------------------------------------------
# N-gram fallback
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _ngram_match(question: str, label_samples: dict[str, list[str]]) -> list[dict]:
    """
    Matches question n-grams (1-3) against sample node names from the schema.
    Returns anchors with a synthetic score = ngram length (longer = better).
    """
    q_lower = question.lower()
    seen_names: set[str] = set()
    hits: list[dict] = []

    # Try progressively shorter n-grams
    for n in range(3, 0, -1):
        grams = _ngrams(question, n)
        for lbl, samples in label_samples.items():
            for sample in samples:
                sample_lower = sample.lower()
                for gram in grams:
                    if gram in sample_lower or sample_lower in q_lower:
                        if sample_lower not in seen_names:
                            seen_names.add(sample_lower)
                            hits.append({
                                "element_id": None,
                                "name": sample,
                                "labels": [lbl],
                                "score": float(n),  # n-gram length as proxy score
                            })

    # Sort by score desc, cap at top-K above threshold
    hits.sort(key=lambda h: h["score"], reverse=True)
    return [h for h in hits if h["score"] >= ENTITY_LINK_MIN_SCORE][:ENTITY_LINK_TOP_K]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _sanitize_lucene_query(text: str) -> str:
    """Strips special Lucene characters that break full-text search."""
    cleaned = re.sub(r'[\+\-\&\|\!\(\)\{\}\[\]\^\"~*?:\\/]', ' ', text)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    return " ".join(tokens)


def link_entities(question: str, schema: dict) -> list[dict]:
    """
    Links the question text to real graph nodes.
    Returns a list of anchor dicts: {element_id, name, labels, score}.
    Logs everything; never raises — returns [] on total failure.
    """
    print("==================================================")
    print("ENTITY LINKING")
    print("==================================================")
    print(f"Question: {question}")

    labels = schema.get("labels", [])
    label_samples = schema.get("label_samples", {})
    anchors: list[dict] = []

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    index_available = False
    try:
        with driver.session() as session:
            index_available = _ensure_fulltext_index(session, labels)

            if index_available:
                lucene_q = _sanitize_lucene_query(question)
                if lucene_q:
                    try:
                        result = session.run(
                            f"""
                            CALL db.index.fulltext.queryNodes($idx, $q)
                            YIELD node, score
                            WHERE score >= $min_score
                            RETURN elementId(node) AS element_id,
                                   node.name        AS name,
                                   labels(node)     AS labels,
                                   score
                            ORDER BY score DESC
                            LIMIT $top_k
                            """,
                            idx=_INDEX_NAME,
                            q=lucene_q,
                            min_score=ENTITY_LINK_MIN_SCORE,
                            top_k=ENTITY_LINK_TOP_K,
                        )
                        for rec in result:
                            anchors.append({
                                "element_id": rec["element_id"],
                                "name": rec["name"],
                                "labels": list(rec["labels"]),
                                "score": rec["score"],
                            })
                    except Exception as exc:
                        print(f"[linker] Full-text query failed: {exc}")

            # Fall back to n-gram if index gave nothing
            if not anchors and label_samples:
                print("[linker] Full-text gave no results; using n-gram fallback.")
                raw_anchors = _ngram_match(question, label_samples)
                for a in raw_anchors:
                    try:
                        res = session.run(
                            "MATCH (n) WHERE toLower(n.name) = toLower($name) RETURN elementId(n) AS element_id, labels(n) AS labels LIMIT 1",
                            name=a["name"],
                        ).single()
                        if res:
                            a["element_id"] = res["element_id"]
                            a["labels"] = list(res["labels"])
                    except Exception as r_exc:
                        print(f"[linker] Could not resolve element_id for '{a['name']}': {r_exc}")
                    anchors.append(a)

    except Exception as exc:
        print(f"[linker] Neo4j connection error: {exc}")
    finally:
        driver.close()

    print(f"Anchors found: {len(anchors)}")
    for a in anchors:
        print(f"  element_id={a.get('element_id')}  {a['name']} {a['labels']}  score={a['score']:.3f}")
    print()
    return anchors

