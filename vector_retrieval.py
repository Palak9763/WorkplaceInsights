"""
Semantic vector retrieval module implementing:
1. Dual search: (a) plain top-k; (b) anchor-filtered search.
2. Reciprocal Rank Fusion (RRF) merge.
3. Relative score cutoff (VECTOR_REL_CUTOFF).
4. Maximal Marginal Relevance (MMR) deduplication (VECTOR_MMR_LAMBDA).
5. Optional Cross-Encoder reranking (RERANKER_MODEL).
6. Matched anchor boolean tagging on each chunk.
7. Stage hit-counts tracking for diagnostics.
"""
import math
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

from config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    VECTOR_TOP_K, RRF_K, VECTOR_REL_CUTOFF, VECTOR_MMR_LAMBDA,
    RERANKER_MODEL, RERANK_TOP_K
)
from embed import embed


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _check_matched_anchor(text: str, anchors: list[dict]) -> bool:
    if not text or not anchors:
        return False
    text_lower = text.lower()
    for a in anchors:
        name = a.get("name")
        if name and name.lower() in text_lower:
            return True
    return False


def _mmr_select(
    query_vec: list[float],
    candidates: list[dict],
    doc_vectors: list[list[float]],
    top_k: int = VECTOR_TOP_K,
    lambda_param: float = VECTOR_MMR_LAMBDA,
) -> list[dict]:
    """
    Selects documents using Maximal Marginal Relevance.
    MMR score = lambda * sim(q, d) - (1 - lambda) * max_{s in Selected} sim(d, s)
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    selected_indices: list[int] = []
    unselected_indices = list(range(len(candidates)))

    # First document: highest similarity to query
    q_sims = [_cosine(query_vec, v) for v in doc_vectors]
    best_first = max(range(len(candidates)), key=lambda i: q_sims[i])
    selected_indices.append(best_first)
    unselected_indices.remove(best_first)

    while len(selected_indices) < top_k and unselected_indices:
        best_score = -float("inf")
        best_idx = None

        for idx in unselected_indices:
            sim_to_query = q_sims[idx]
            max_sim_to_selected = max(_cosine(doc_vectors[idx], doc_vectors[s]) for s in selected_indices)
            mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected_indices]


def vector_retrieve(
    question: str, anchors: list[dict] | None = None
) -> tuple[list[dict], dict, str | None]:
    """
    Executes semantic vector retrieval with RRF fusion, relative cutoff, MMR,
    and optional cross-encoder reranking.

    Returns: (results: list[dict], vector_counts: dict, error_or_None: str | None)
    """
    anchors = anchors or []
    counts = {
        "plain_hits": 0,
        "filtered_hits": 0,
        "rrf_merged": 0,
        "cutoff_kept": 0,
        "mmr_kept": 0,
        "final_count": 0,
    }
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    error: str | None = None

    try:
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            error = f"Qdrant collection '{QDRANT_COLLECTION}' not found."
            print(f"[vector_retrieval] ERROR: {error}")
            return [], counts, error

        query_vector = embed(question)
        print(f"[vector_retrieval] Question embedded, vector len={len(query_vector)}")

        # --- Search 1: Plain semantic search ---
        plain_hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=VECTOR_TOP_K * 2,
        )
        counts["plain_hits"] = len(plain_hits)
        print(f"[vector_retrieval] Plain search returned {len(plain_hits)} hits")

        # --- Search 2: Anchor-filtered search (if anchors exist) ---
        filtered_hits = []
        if anchors:
            for a in anchors:
                a_name = a.get("name")
                if not a_name:
                    continue
                try:
                    f_hits = client.search(
                        collection_name=QDRANT_COLLECTION,
                        query_vector=query_vector,
                        query_filter=Filter(
                            must=[FieldCondition(key="text", match=MatchText(text=a_name))]
                        ),
                        limit=VECTOR_TOP_K * 2,
                    )
                    filtered_hits.extend(f_hits)
                except Exception as f_exc:
                    print(f"[vector_retrieval] Anchor filter search notice for '{a_name}': {f_exc}")
        counts["filtered_hits"] = len(filtered_hits)
        print(f"[vector_retrieval] Anchor-filtered search returned {len(filtered_hits)} hits")

        # --- Reciprocal Rank Fusion (RRF) ---
        # Track items by text/id
        items_map: dict[str, dict] = {}
        rrf_scores: dict[str, float] = {}

        def add_ranked_list(hits_list):
            seen_in_list = set()
            rank = 0
            for hit in hits_list:
                payload = dict(hit.payload) if hit.payload else {}
                text_key = payload.get("text") or str(hit.id)
                if text_key in seen_in_list:
                    continue
                seen_in_list.add(text_key)
                if text_key not in items_map:
                    items_map[text_key] = {
                        "text": payload.get("text", ""),
                        "source": payload.get("source", ""),
                        "date": payload.get("date", ""),
                        "score": hit.score,
                        "matched_anchor": _check_matched_anchor(payload.get("text", ""), anchors),
                    }
                rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + (1.0 / (RRF_K + rank + 1))
                rank += 1

        add_ranked_list(plain_hits)
        if filtered_hits:
            add_ranked_list(filtered_hits)

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        merged_items = []
        for k in sorted_keys:
            item = items_map[k]
            item["rrf_score"] = round(rrf_scores[k], 5)
            merged_items.append(item)
        counts["rrf_merged"] = len(merged_items)
        print(f"[vector_retrieval] RRF merged {len(merged_items)} distinct chunks")

        if not merged_items:
            return [], counts, None

        # --- Relative Cutoff ---
        top_rrf = merged_items[0]["rrf_score"]
        cutoff_threshold = top_rrf * VECTOR_REL_CUTOFF
        cutoff_items = [item for item in merged_items if item["rrf_score"] >= cutoff_threshold]
        counts["cutoff_kept"] = len(cutoff_items)
        print(f"[vector_retrieval] Relative cutoff kept {len(cutoff_items)}/{len(merged_items)} chunks (cutoff={cutoff_threshold:.5f})")

        # --- MMR Deduplication ---
        doc_vectors = []
        for item in cutoff_items:
            doc_vectors.append(embed(item.get("text", "")))

        mmr_items = _mmr_select(
            query_vec=query_vector,
            candidates=cutoff_items,
            doc_vectors=doc_vectors,
            top_k=VECTOR_TOP_K,
            lambda_param=VECTOR_MMR_LAMBDA,
        )
        counts["mmr_kept"] = len(mmr_items)
        print(f"[vector_retrieval] MMR kept {len(mmr_items)} chunks")

        # --- Optional Cross-Encoder Reranker ---
        final_results = mmr_items
        if RERANKER_MODEL.strip():
            try:
                print(f"[vector_retrieval] Running cross-encoder reranker ({RERANKER_MODEL})...")
                # If sentence_transformers is available, use CrossEncoder; else fallback gracefully with log
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder(RERANKER_MODEL)
                pairs = [[question, item.get("text", "")] for item in mmr_items]
                cross_scores = reranker.predict(pairs)
                for item, s in zip(mmr_items, cross_scores):
                    item["rerank_score"] = float(s)
                final_results = sorted(mmr_items, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:RERANK_TOP_K]
                print(f"[vector_retrieval] Reranker kept top {len(final_results)} chunks")
            except Exception as r_exc:
                print(f"[vector_retrieval] Cross-encoder reranker notice/error: {r_exc}")

        counts["final_count"] = len(final_results)
        return final_results, counts, None

    except Exception as exc:
        error = f"Vector retrieval error: {exc}"
        print(f"[vector_retrieval] ERROR: {error}")
        return [], counts, error
