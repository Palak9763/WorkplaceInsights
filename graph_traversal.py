"""
Graph traversal module implementing:
1. ElementId-based parameterized variable-length anchor expansion (1-hop or 2-hop).
2. Connecting path search between multi-anchors (allShortestPaths).
3. Hub protection (fanout capping with similarity ranking).
4. Stored-direction edge extraction and deduplication by elementId(r).
5. Fact ranking against question embeddings with relative cutoff.
6. Graph path building.
7. Conditional LLM Cypher execution for aggregations or empty traversals.
8. Stage timeout enforcement (GRAPH_STAGE_TIMEOUT).
"""
import time
import math
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GRAPH_MAX_ROWS, GRAPH_MAX_DEPTH, GRAPH_MAX_PATH_LEN,
    GRAPH_MAX_FANOUT, GRAPH_TOP_K, GRAPH_REL_CUTOFF,
    GRAPH_STAGE_TIMEOUT
)
from embed import embed


def _serialize(val):
    if isinstance(val, Node):
        return {"id": val.id, "element_id": getattr(val, "element_id", str(val.id)),
                "labels": list(val.labels), "properties": dict(val.items())}
    if isinstance(val, Relationship):
        return {"id": val.id, "element_id": getattr(val, "element_id", str(val.id)),
                "type": val.type, "properties": dict(val.items())}
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


def run_traversal_query(cypher: str, params: dict | None = None, timeout: float = GRAPH_STAGE_TIMEOUT) -> tuple[list[dict], str | None]:
    """
    Executes a Cypher query against Neo4j with transaction timeout.
    Returns (rows, error_or_None).
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    rows: list[dict] = []
    error: str | None = None
    t0 = time.perf_counter()
    try:
        with driver.session() as session:
            # Enforce timeout via transaction configuration
            result = session.run(cypher, **(params or {}), timeout=timeout)
            for record in result:
                rows.append({k: _serialize(record[k]) for k in record.keys()})
    except Exception as exc:
        error = f"Neo4j query error: {exc}"
        print(f"[graph_traversal] {error}")
    finally:
        driver.close()
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    print(f"[graph_traversal] Query executed in {elapsed_ms}ms ({len(rows)} raw rows)")
    return rows, error


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def edge_to_sentence(edge: dict) -> str:
    """Formats an edge dict in stored direction."""
    src = edge.get("source_name") or "?"
    rel = edge.get("relationship") or "?"
    tgt = edge.get("target_name") or "?"
    return f"{src} -[{rel}]-> {tgt}"


def _cap_hub_fanout(edges: list[dict], question: str) -> list[dict]:
    """
    Hub protection: if any node has more than GRAPH_MAX_FANOUT edges connected to it,
    rank those incident edges by similarity to the question and retain the top GRAPH_MAX_FANOUT.
    """
    if not edges or GRAPH_MAX_FANOUT <= 0:
        return edges

    # Count incident edges per node (by node ID or name)
    node_edges: dict[str, list[dict]] = {}
    for e in edges:
        src = e.get("source_id") or e.get("source_name")
        tgt = e.get("target_id") or e.get("target_name")
        if src:
            node_edges.setdefault(src, []).append(e)
        if tgt and tgt != src:
            node_edges.setdefault(tgt, []).append(e)

    # Find hubs exceeding GRAPH_MAX_FANOUT
    hubs = {node: edg_list for node, edg_list in node_edges.items() if len(edg_list) > GRAPH_MAX_FANOUT}
    if not hubs:
        return edges

    print(f"[graph_traversal] Hub protection triggered for {len(hubs)} node(s) exceeding max fanout {GRAPH_MAX_FANOUT}.")

    # Pre-embed question for ranking if needed
    q_vec = None
    try:
        if question:
            q_vec = embed(question)
    except Exception as exc:
        print(f"[graph_traversal] Hub ranking embed error: {exc}")

    kept_rids: set[str] = set()
    for e in edges:
        kept_rids.add(e.get("rid", ""))

    for node, incident in hubs.items():
        if len(incident) <= GRAPH_MAX_FANOUT:
            continue
        print(f"[graph_traversal] Node '{node}' has {len(incident)} edges > fanout cap {GRAPH_MAX_FANOUT}. Truncating...")
        if q_vec:
            try:
                scored = []
                for inc in incident:
                    s_vec = embed(edge_to_sentence(inc))
                    sim = _cosine(q_vec, s_vec)
                    scored.append((sim, inc))
                scored.sort(key=lambda x: x[0], reverse=True)
                top_incident_rids = {item[1].get("rid", "") for item in scored[:GRAPH_MAX_FANOUT]}
                for inc in incident:
                    rid = inc.get("rid", "")
                    if rid not in top_incident_rids and rid in kept_rids:
                        kept_rids.remove(rid)
            except Exception as exc:
                print(f"[graph_traversal] Hub truncation score error: {exc}; slicing directly.")
                dropped_rids = {item.get("rid", "") for item in incident[GRAPH_MAX_FANOUT:]}
                kept_rids -= dropped_rids
        else:
            dropped_rids = {item.get("rid", "") for item in incident[GRAPH_MAX_FANOUT:]}
            kept_rids -= dropped_rids

    filtered_edges = [e for e in edges if e.get("rid", "") in kept_rids]
    print(f"[graph_traversal] Edges after hub fanout capping: {len(edges)} -> {len(filtered_edges)}")
    return filtered_edges


def find_paths_between_anchors(anchor_ids: list[str], max_path_len: int = GRAPH_MAX_PATH_LEN) -> tuple[list[dict], list[dict], str | None]:
    """
    Finds connecting paths between pairs of anchors using allShortestPaths up to max_path_len.
    Returns (distinct_edges, raw_paths, error_or_None).
    """
    if len(anchor_ids) < 2:
        return [], [], None

    cypher = f"""
    MATCH (a), (b)
    WHERE elementId(a) = $id1 AND elementId(b) = $id2 AND elementId(a) < elementId(b)
    MATCH p = allShortestPaths((a)-[*..{max_path_len}]-(b))
    RETURN
        [n IN nodes(p) | {{id: elementId(n), name: n.name, labels: labels(n)}}] AS path_nodes,
        [r IN relationships(p) | {{
            rid: elementId(r),
            source_name: startNode(r).name,
            source_labels: labels(startNode(r)),
            relationship: type(r),
            target_name: endNode(r).name,
            target_labels: labels(endNode(r)),
            source_id: elementId(startNode(r)),
            target_id: elementId(endNode(r))
        }}] AS path_edges
    LIMIT $max_rows
    """
    all_edges_dict: dict[str, dict] = {}
    raw_paths: list[dict] = []
    error: str | None = None

    for i in range(len(anchor_ids)):
        for j in range(i + 1, len(anchor_ids)):
            id1, id2 = anchor_ids[i], anchor_ids[j]
            if id1 > id2:
                id1, id2 = id2, id1
            params = {"id1": id1, "id2": id2, "max_rows": GRAPH_MAX_ROWS}
            rows, err = run_traversal_query(cypher, params)
            if err:
                error = err
            for row in rows:
                raw_paths.append(row)
                for edge in row.get("path_edges", []):
                    rid = edge.get("rid")
                    if rid and rid not in all_edges_dict:
                        all_edges_dict[rid] = edge

    return list(all_edges_dict.values()), raw_paths, error


def expand_anchors(anchor_ids: list[str], depth: int = GRAPH_MAX_DEPTH, question: str = "") -> tuple[list[dict], str | None]:
    """
    Performs variable-length expansion (1-hop or 2-hop) from anchor nodes.
    Deduplicates edges by elementId(r) in stored direction.
    Applies hub protection.
    Returns (edges, error_or_None).
    """
    if not anchor_ids:
        return [], None

    if depth <= 1:
        cypher = """
        MATCH (a)-[r]-(b)
        WHERE elementId(a) IN $anchor_ids
        RETURN DISTINCT
            elementId(r)             AS rid,
            startNode(r).name         AS source_name,
            labels(startNode(r))      AS source_labels,
            type(r)                   AS relationship,
            endNode(r).name           AS target_name,
            labels(endNode(r))        AS target_labels,
            elementId(startNode(r))   AS source_id,
            elementId(endNode(r))     AS target_id
        LIMIT $max_rows
        """
    else:
        cypher = """
        MATCH p = (a)-[*1..2]-(b)
        WHERE elementId(a) IN $anchor_ids
        UNWIND relationships(p) AS r
        RETURN DISTINCT
            elementId(r)             AS rid,
            startNode(r).name         AS source_name,
            labels(startNode(r))      AS source_labels,
            type(r)                   AS relationship,
            endNode(r).name           AS target_name,
            labels(endNode(r))        AS target_labels,
            elementId(startNode(r))   AS source_id,
            elementId(endNode(r))     AS target_id
        LIMIT $max_rows
        """

    rows, error = run_traversal_query(cypher, {"anchor_ids": anchor_ids, "max_rows": GRAPH_MAX_ROWS})
    if error:
        return [], error

    seen_rids: set[str] = set()
    deduped_edges: list[dict] = []
    for r in rows:
        rid = r.get("rid")
        if rid and rid not in seen_rids:
            seen_rids.add(rid)
            deduped_edges.append(r)

    capped_edges = _cap_hub_fanout(deduped_edges, question)
    return capped_edges, None


def rank_and_build_paths(edges: list[dict], question: str) -> tuple[list[dict], list[dict]]:
    """
    Ranks extracted edges by cosine similarity of their sentence representation to the question.
    Filters by GRAPH_REL_CUTOFF (relative to top score).
    Builds graph_results and graph_path.
    Returns (graph_results, graph_path).
    """
    if not edges:
        return [], []

    sentences = [edge_to_sentence(e) for e in edges]
    try:
        q_vec = embed(question)
        sent_vecs = [embed(s) for s in sentences]
        scores = [_cosine(q_vec, v) for v in sent_vecs]
    except Exception as exc:
        print(f"[graph_traversal] Fact ranking embedding error: {exc}")
        scores = [0.0] * len(edges)

    paired = sorted(zip(scores, edges, sentences), key=lambda x: x[0], reverse=True)
    top_score = paired[0][0] if paired else 0.0
    cutoff = top_score * GRAPH_REL_CUTOFF if top_score > 0 else 0.0

    graph_results = []
    graph_path = []

    for score, edge, sentence in paired:
        if score < cutoff and len(graph_results) > 0:
            continue
        edge_copy = dict(edge)
        edge_copy["_sentence"] = sentence
        edge_copy["_rank_score"] = round(score, 4)
        graph_results.append(edge_copy)

        graph_path.append({
            "source_name": edge.get("source_name"),
            "source_labels": edge.get("source_labels", []),
            "relationship": edge.get("relationship"),
            "target_name": edge.get("target_name"),
            "target_labels": edge.get("target_labels", []),
            "rank_score": round(score, 4),
        })

        if len(graph_results) >= GRAPH_TOP_K:
            break

    print(f"[graph_traversal] Fact ranking: {len(edges)} edges -> {len(graph_results)} kept (top={top_score:.4f}, cutoff={cutoff:.4f})")
    return graph_results, graph_path
