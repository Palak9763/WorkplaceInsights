"""
Classifies a question as 'graph', 'vector', or 'hybrid', and detects
whether the query requires multihop traversal (depth 2) or aggregation/counting.

Prompt contains the live schema summary and linked anchor nodes so the
classification is data-agnostic. No hardcoded entity names, label names,
relationship names, keyword lists, or regex routing anywhere.
"""
import json
import re
import ollama
from config import OLLAMA_HOST, EXTRACTION_MODEL, ANCHOR_FORCES_HYBRID


def _build_classify_prompt(schema: dict, anchors: list[dict]) -> str:
    labels = schema.get("labels", [])
    rel_types = schema.get("relationship_types", [])
    anchor_lines = (
        "\n".join(f"  - {a['name']} {a['labels']}" for a in anchors)
        if anchors else "  (none)"
    )
    return f"""You are a query router for a GraphRAG system.
Schema info:
  Node labels (entity types): {labels or '(none yet)'}
  Relationship types: {rel_types or '(none yet)'}

Linked entities found in the question:
{anchor_lines}

ROUTING RULES:
- route:
    * "graph": Needs only structured relationships / paths between entities in the graph.
    * "vector": Needs only unstructured document text / policy / prose.
    * "hybrid": Needs BOTH graph relationships AND unstructured document context.
- multihop: true if answering requires traversing 2 or more hops (e.g. shared connections, dependencies of dependencies, indirect connections); false otherwise.
- aggregation: true if the query asks for counts, totals, or aggregations (e.g. "how many", "count of", "total"); false otherwise.

Respond ONLY with a JSON object in this exact format:
{{"route": "graph"|"vector"|"hybrid", "multihop": true|false, "aggregation": true|false}}"""


def classify_query(question: str, schema: dict, anchors: list[dict]) -> dict:
    """
    Returns dict:
      {
        "route": "graph" | "vector" | "hybrid",
        "multihop": bool,
        "aggregation": bool,
        "error": str | None
      }
    """
    print("==================================================")
    print("QUERY CLASSIFICATION")
    print("==================================================")
    print(f"Question: {question}")
    print(f"Anchor count: {len(anchors)}")

    system_prompt = _build_classify_prompt(schema, anchors)
    client = ollama.Client(host=OLLAMA_HOST)
    route = "hybrid"
    multihop = False
    aggregation = False
    error: str | None = None

    try:
        response = client.chat(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this question:\n{question}"},
            ],
            options={"temperature": 0.0},
        )
        content = response["message"]["content"].strip()
        # Parse JSON if possible
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                r = str(data.get("route", "")).strip().lower()
                if r in ("graph", "vector", "hybrid"):
                    route = r
                multihop = bool(data.get("multihop", False))
                aggregation = bool(data.get("aggregation", False))
            except Exception as j_err:
                error = f"JSON parse error in classification: {j_err} (raw: {content[:100]})"
        else:
            # Fallback word matching
            first_word = content.split()[0].strip("`'\".,!?:;*").lower() if content.split() else ""
            if first_word in ("graph", "vector", "hybrid"):
                route = first_word
            else:
                route = "hybrid"
                error = f"Unexpected classification response: {content[:100]}; defaulted to hybrid"
    except Exception as exc:
        error = f"LLM error during classification: {exc}"
        print(f"[classify] ERROR: {exc}")
        route = "hybrid"

    # Anchor signal override if ANCHOR_FORCES_HYBRID is set
    if anchors and ANCHOR_FORCES_HYBRID and route == "vector":
        print(f"[classify] Upgrading 'vector' -> 'hybrid' because {len(anchors)} anchor(s) found.")
        route = "hybrid"

    result = {
        "route": route,
        "multihop": multihop,
        "aggregation": aggregation,
        "error": error,
    }

    print(f"Classification: route={route}, multihop={multihop}, aggregation={aggregation}" +
          (f" [WARNING: {error}]" if error else ""))
    print()
    return result

