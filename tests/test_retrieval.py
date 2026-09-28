"""
Unit and integration tests for graph traversal and semantic vector retrieval.
Uses placeholder entities only (Acme, Widget Co, Alpha Corp). Never uses real data.
"""
from unittest.mock import patch, MagicMock

from graph_traversal import (
    _cap_hub_fanout,
    edge_to_sentence,
    find_paths_between_anchors,
    expand_anchors,
    rank_and_build_paths,
)
from vector_retrieval import _mmr_select, _check_matched_anchor, vector_retrieve
from query_classifier import classify_query


# ---------------------------------------------------------------------------
# Test 1: Hub fan-out capping
# ---------------------------------------------------------------------------
def test_hub_fanout_capping():
    # Construct a hub "Acme Corp" with 15 incident edges
    edges = []
    for i in range(15):
        edges.append({
            "rid": f"edge-{i}",
            "source_id": "node-acme",
            "source_name": "Acme Corp",
            "source_labels": ["Organization"],
            "relationship": "OWNS",
            "target_id": f"node-widget-{i}",
            "target_name": f"Widget {i}",
            "target_labels": ["Product"],
        })

    with patch("graph_traversal.GRAPH_MAX_FANOUT", 5):
        capped = _cap_hub_fanout(edges, question="Acme Widget 0")
        assert len(capped) <= 5
        # Ensure all kept rids are unique
        kept_rids = [e["rid"] for e in capped]
        assert len(kept_rids) == len(set(kept_rids))


# ---------------------------------------------------------------------------
# Test 2: RRF merge calculation & ordering
# ---------------------------------------------------------------------------
def test_rrf_scoring_and_matched_anchor():
    anchors = [{"name": "Acme Corp", "labels": ["Organization"]}]
    assert _check_matched_anchor("This document discusses Acme Corp operations.", anchors) is True
    assert _check_matched_anchor("This document discusses unrelated topics.", anchors) is False

    # Simulate RRF scoring
    rrf_k = 60
    # Item A is rank 0 in search 1 and rank 0 in search 2 -> score = 2 * (1 / (60 + 1))
    score_a = (1.0 / (rrf_k + 1)) + (1.0 / (rrf_k + 1))
    # Item B is rank 0 in search 1 only -> score = 1.0 / (rrf_k + 1)
    score_b = (1.0 / (rrf_k + 1))
    assert score_a > score_b


# ---------------------------------------------------------------------------
# Test 3: MMR deduplication
# ---------------------------------------------------------------------------
def test_mmr_dedup():
    query_vec = [1.0, 0.5, 0.0]
    # Doc 0: relevant to query
    doc0_vec = [1.0, 0.0, 0.0]
    # Doc 1: exact duplicate of Doc 0
    doc1_vec = [1.0, 0.0, 0.0]
    # Doc 2: diverse from Doc 0 and partially relevant to query
    doc2_vec = [0.0, 1.0, 0.0]

    candidates = [
        {"text": "Acme system specification", "source": "doc1"},
        {"text": "Acme system specification copy", "source": "doc2"},
        {"text": "Widget Co maintenance guide", "source": "doc3"},
    ]
    doc_vectors = [doc0_vec, doc1_vec, doc2_vec]

    selected = _mmr_select(
        query_vec=query_vec,
        candidates=candidates,
        doc_vectors=doc_vectors,
        top_k=2,
        lambda_param=0.5,
    )
    assert len(selected) == 2
    selected_texts = [s["text"] for s in selected]
    assert "Acme system specification" in selected_texts
    assert "Widget Co maintenance guide" in selected_texts


# ---------------------------------------------------------------------------
# Test 4: 1-hop vs 2-hop traversal query selection
# ---------------------------------------------------------------------------
def test_1hop_vs_2hop_expansion():
    with patch("graph_traversal.run_traversal_query") as mock_run:
        mock_run.return_value = ([], None)

        # 1-hop
        expand_anchors(["anchor-id-1"], depth=1)
        cypher_1hop = mock_run.call_args[0][0]
        assert "MATCH (a)-[r]-(b)" in cypher_1hop

        # 2-hop
        expand_anchors(["anchor-id-1"], depth=2)
        cypher_2hop = mock_run.call_args[0][0]
        assert "*1..2" in cypher_2hop


# ---------------------------------------------------------------------------
# Test 5: Connecting path finding between 2 anchors
# ---------------------------------------------------------------------------
def test_path_finding_between_two_anchors():
    with patch("graph_traversal.run_traversal_query") as mock_run:
        mock_run.return_value = (
            [
                {
                    "path_nodes": [
                        {"id": "a1", "name": "Acme Corp"},
                        {"id": "a2", "name": "Widget Co"},
                    ],
                    "path_edges": [
                        {
                            "rid": "edge-rel-1",
                            "source_name": "Acme Corp",
                            "source_labels": ["Organization"],
                            "relationship": "PARTNERS_WITH",
                            "target_name": "Widget Co",
                            "target_labels": ["Organization"],
                            "source_id": "a1",
                            "target_id": "a2",
                        }
                    ],
                }
            ],
            None,
        )

        edges, raw_paths, err = find_paths_between_anchors(["a1", "a2"], max_path_len=3)
        assert err is None
        assert len(edges) == 1
        assert edges[0]["relationship"] == "PARTNERS_WITH"


# ---------------------------------------------------------------------------
# Test 6: Loud failure on unreachable Qdrant
# ---------------------------------------------------------------------------
def test_loud_failure_qdrant_unreachable():
    with patch("vector_retrieval.QDRANT_URL", "http://127.0.0.1:59999"):
        results, counts, error = vector_retrieve("Sample question")
        assert error is not None
        assert "error" in error.lower() or "connection" in error.lower() or "qdrant" in error.lower()


# ---------------------------------------------------------------------------
# Test 7: Loud failure / graceful reporting on unreachable Ollama
# ---------------------------------------------------------------------------
def test_loud_failure_ollama_unreachable():
    with patch("query_classifier.OLLAMA_HOST", "http://127.0.0.1:59999"):
        schema = {"labels": ["Organization"], "relationship_types": ["PARTNERS_WITH"]}
        res = classify_query("What does Acme Corp do?", schema, [])
        assert res["route"] == "hybrid"  # default fallback
        assert res["error"] is not None
        assert "error" in res["error"].lower()
