"""
Generates a single read-only Cypher query via the LLM.

Key design principles (all data-agnostic):
- The full live schema (labels, rel types, property keys, sample names) is
  injected into every prompt.
- Linked anchor nodes with their exact names are included so the model uses
  real names instead of guessing.
- Structural example patterns are built programmatically from the live schema
  using placeholder values — no real entity names appear in the prompt.
- GRAPH_MAX_ROWS from config is used for LIMIT; never hardcoded.
"""
import ollama
from groq import Groq
from config import OLLAMA_HOST, EXTRACTION_MODEL, GRAPH_MAX_ROWS, REASONING_PROVIDER, GROQ_API_KEY, GROQ_MODEL

_SYSTEM_INSTRUCTIONS = """\
You are an expert Neo4j Cypher query generator.
Convert the user question into exactly ONE valid, read-only Cypher query.

=== HARD RULES ===
1. Use ONLY the labels and relationship types listed in LIVE SCHEMA.
2. Use ONLY properties listed for each label in LIVE SCHEMA. Never invent properties.
3. Relationships carry NO properties in this graph. Never reference r.<anything>.
4. Never use inline property maps on nodes: (n:Label {{prop: 'val'}}) is FORBIDDEN.
   Instead: MATCH (n:Label) WHERE toLower(n.name) CONTAINS toLower('<term>')
5. The CONTAINS term must be the SHORTEST single token that identifies the entity.
   Use the exact anchor names provided in ANCHORS when available.
6. For "who"/"which entity" questions: use an undirected, untyped relationship pattern.
   RETURN subject.name AS name, labels(subject) AS labels,
          type(r) AS relationship, target.name AS target
7. Allowed clauses: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
   FORBIDDEN: CREATE, DELETE, DETACH, SET, MERGE, REMOVE.
8. Always end with LIMIT {max_rows}.
9. Output ONLY the Cypher — no prose, no markdown fences.
"""

_PATTERN_TEMPLATES = [
    # neighborhood of a named anchor
    "# Pattern: neighborhood of a specific entity\n"
    "MATCH (anchor)-[r]-(neighbor)\n"
    "WHERE toLower(anchor.name) CONTAINS toLower('<anchor_name>')\n"
    "RETURN anchor.name AS name, labels(anchor) AS labels,\n"
    "       type(r) AS relationship, neighbor.name AS target\n"
    "LIMIT {max_rows}",

    # directional outgoing from anchor
    "# Pattern: outgoing relationships from a specific entity\n"
    "MATCH (anchor)-[r]->(target)\n"
    "WHERE toLower(anchor.name) CONTAINS toLower('<anchor_name>')\n"
    "RETURN anchor.name AS name, labels(anchor) AS labels,\n"
    "       type(r) AS relationship, target.name AS target\n"
    "LIMIT {max_rows}",

    # entities connected via a specific rel type
    "# Pattern: all pairs connected by a relationship type\n"
    "MATCH (a)-[r:{example_rel}]->(b)\n"
    "RETURN a.name AS name, labels(a) AS labels,\n"
    "       type(r) AS relationship, b.name AS target\n"
    "LIMIT {max_rows}",
]


def _build_prompt(schema: dict, anchors: list[dict], question: str, error_context: str) -> tuple[str, str]:
    labels = schema.get("labels", [])
    rel_types = schema.get("relationship_types", [])
    label_props = schema.get("label_properties", {})
    label_samples = schema.get("label_samples", {})

    # --- Schema block ---
    schema_lines = ["=== LIVE SCHEMA ==="]
    schema_lines.append(f"Node labels: {labels or '(none)'}")
    schema_lines.append(f"Relationship types: {rel_types or '(none)'}")
    schema_lines.append("")
    for lbl in labels:
        props = label_props.get(lbl, ["name", "name_key", "location", "last_seen_source"])
        samples = label_samples.get(lbl, [])
        schema_lines.append(f"  :{lbl}  props={props}  sample_names={samples}")
    schema_block = "\n".join(schema_lines)

    # --- Anchors block ---
    if anchors:
        anchor_lines = "\n".join(
            f"  element_id={a.get('element_id') or 'n/a'}  name={a['name']!r}  labels={a['labels']}"
            for a in anchors
        )
        anchors_block = f"=== ANCHORS (use these exact names in WHERE clauses) ===\n{anchor_lines}"
    else:
        anchors_block = "=== ANCHORS ===\n  (none — use CONTAINS on likely terms from the question)"

    # --- Structural example patterns (no real entity names) ---
    example_rel = rel_types[0] if rel_types else "REL_TYPE"
    patterns = "\n\n".join(
        t.format(max_rows=GRAPH_MAX_ROWS, example_rel=example_rel)
        for t in _PATTERN_TEMPLATES
    )
    examples_block = f"=== EXAMPLE PATTERNS (templates only — replace placeholders) ===\n{patterns}"

    system_prompt = _SYSTEM_INSTRUCTIONS.format(max_rows=GRAPH_MAX_ROWS)
    user_prompt = f"{schema_block}\n\n{anchors_block}\n\n{examples_block}\n\nQUESTION: {question}\n"
    if error_context:
        user_prompt += (
            f"\nPREVIOUS ATTEMPT INVALID: {error_context}\n"
            "Fix the query, applying ALL hard rules. Output ONLY Cypher."
        )
    else:
        user_prompt += "\nGenerate the Cypher query:"

    return system_prompt, user_prompt


def _clean(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def generate_cypher(
    question: str, schema: dict, anchors: list[dict], error_context: str = ""
) -> tuple[str, str | None]:
    """
    Returns (cypher_string, error_or_None).
    On LLM failure returns a safe fallback Cypher and the error string.
    """
    print("==================================================")
    print("CYPHER GENERATION")
    print("==================================================")
    print(f"Question: {question}")
    print(f"Anchors passed: {[a['name'] for a in anchors]}")

    system_prompt, user_prompt = _build_prompt(schema, anchors, question, error_context)
    error: str | None = None

    try:
        if REASONING_PROVIDER == "groq" and GROQ_API_KEY:
            # ── Groq path (fast) ─────────────────────────────────────────
            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
            )
            cypher = _clean(completion.choices[0].message.content)
        else:
            # ── Ollama path (fallback) ────────────────────────────────
            ollama_client = ollama.Client(host=OLLAMA_HOST)
            response = ollama_client.chat(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0.0},
            )
            cypher = _clean(response["message"]["content"])
    except Exception as exc:
        error = f"LLM Cypher generation failed: {exc}"
        print(f"[cypher_gen] ERROR: {exc}")
        cypher = f"MATCH (n) RETURN n.name AS name, labels(n) AS labels LIMIT {GRAPH_MAX_ROWS}"

    print(f"Generated Cypher:\n{cypher}")
    print()
    return cypher, error
