"""
Validates a generated Cypher query before execution.

Checks (all data-agnostic, driven by live schema):
  (a) Read-only: rejects CREATE, DELETE, DETACH, SET, MERGE, REMOVE
  (b) No inline property-map node matching: (n:Label {prop: 'val'})
  (c) Label check: every label in the query must exist in the schema
      (case-insensitive; corrects casing to canonical form)
  (d) Relationship type check: same as (c)
  (e) LIMIT: appended from config if missing

Returns (is_valid: bool, validated_query_or_error_message: str)
"""
import re
from config import GRAPH_MAX_ROWS

_FORBIDDEN = ["CREATE", "DELETE", "DETACH", "SET", "MERGE", "REMOVE"]


def _strip_strings(cypher: str) -> str:
    return re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", "", cypher)


def _extract_labels_and_rels(cypher: str) -> tuple[set[str], set[str]]:
    no_strings = _strip_strings(cypher)

    # Relationship types inside [...] blocks
    rel_types: set[str] = set()
    for block in re.findall(r"\[.*?\]", no_strings, re.DOTALL):
        block = re.sub(r"\{.*?\}", "", block)
        for m1, m2 in re.findall(r":`([^`]+)`|:([A-Za-z0-9_]+)", block):
            t = m1 or m2
            if t:
                rel_types.add(t)

    # Node labels: strip rel blocks and prop maps first
    no_rels = re.sub(r"\[.*?\]", "", no_strings)
    no_props = re.sub(r"\{.*?\}", "", no_rels)
    labels: set[str] = set()
    for m1, m2 in re.findall(r":`([^`]+)`|:([A-Za-z0-9_]+)", no_props):
        lbl = m1 or m2
        if lbl:
            labels.add(lbl)

    return labels, rel_types


def validate_cypher(cypher: str, schema: dict) -> tuple[bool, str]:
    print("==================================================")
    print("CYPHER VALIDATION")
    print("==================================================")
    print(f"Input:\n{cypher}\n")

    # (a) Forbidden mutations
    for kw in _FORBIDDEN:
        if re.search(r"\b" + kw + r"\b", cypher, re.IGNORECASE):
            err = f"Forbidden write keyword '{kw}' found in query."
            print(f"[INVALID] {err}\n")
            return False, err

    # (b) Inline name property map in node pattern (e.g. {name: '...'})
    if re.search(r"\(\s*\w*\s*(?::\s*[\w`]+)?\s*\{[^}]*\bname\s*:", cypher, re.IGNORECASE):
        err = (
            "Inline name property map in node pattern detected (e.g. {name: '...'}). "
            "Use WHERE toLower(n.name) CONTAINS toLower('<term>') or WHERE elementId(n) = $id instead."
        )
        print(f"[INVALID] {err}")
        print(f"Rejected Query:\n{cypher}\n")
        return False, err

    # Case-insensitive canonical lookups
    labels_lower: dict[str, str] = schema.get("labels_lower", {})
    rel_types_lower: dict[str, str] = schema.get("rel_types_lower", {})

    # Build canonical sets for fallback (when *_lower dicts missing)
    schema_labels_ci = labels_lower or {
        lbl.lower(): lbl
        for lbl in (schema.get("labels") or schema.get("entity_types") or [])
    }
    schema_rels_ci = rel_types_lower or {
        rt.lower(): rt
        for rt in (schema.get("relationship_types") or [])
    }

    extracted_labels, extracted_rels = _extract_labels_and_rels(cypher)

    # (c) Label check — case-insensitive
    for lbl in extracted_labels:
        if schema_labels_ci and lbl.lower() not in schema_labels_ci:
            err = (
                f"Label '{lbl}' not in graph schema. "
                f"Available: {sorted(schema_labels_ci.values())}"
            )
            print(f"[INVALID] {err}\n")
            return False, err

    # (d) Relationship type check — case-insensitive
    for rt in extracted_rels:
        if schema_rels_ci and rt.lower() not in schema_rels_ci:
            err = (
                f"Relationship type '{rt}' not in graph schema. "
                f"Available: {sorted(schema_rels_ci.values())}"
            )
            print(f"[INVALID] {err}\n")
            return False, err

    # (e) Ensure LIMIT
    final = cypher.strip()
    if not re.search(r"\bLIMIT\b", final, re.IGNORECASE):
        final = f"{final}\nLIMIT {GRAPH_MAX_ROWS}"

    print(f"[VALID] Cypher validation passed\nFinal:\n{final}\n")
    return True, final
