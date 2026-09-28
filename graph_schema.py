"""
Fetches and caches the live Neo4j schema:
  - Node labels
  - Relationship types
  - Per-label property keys (db.schema.nodeTypeProperties)
  - Per-relationship-type property keys (db.schema.relTypeProperties)
  - Up to N sample `name` values per label (N from config)

Cache TTL from config.SCHEMA_CACHE_TTL.
Case-insensitive label/rel-type sets exposed as schema["labels_lower"] and
schema["rel_types_lower"] for downstream case-insensitive comparisons.
"""
import time
from neo4j import GraphDatabase
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    SCHEMA_CACHE_TTL, SCHEMA_SAMPLE_NAMES_PER_LABEL,
)

_schema_cache: dict | None = None
_cache_timestamp: float = 0.0


def get_live_schema(force_refresh: bool = False) -> dict:
    """
    Returns:
    {
      "labels":              ["LabelA", ...],
      "labels_lower":        {"labela": "LabelA", ...},   # canonical name lookup
      "entity_types":        <alias for "labels">,
      "relationship_types":  ["REL_A", ...],
      "rel_types_lower":     {"rel_a": "REL_A", ...},
      "label_properties":    {"LabelA": ["name", "prop1", ...], ...},
      "rel_properties":      {"REL_A": ["prop1", ...], ...},
      "label_samples":       {"LabelA": ["Foo", "Bar", ...], ...},
    }
    """
    global _schema_cache, _cache_timestamp
    now = time.time()
    if (
        not force_refresh
        and _schema_cache is not None
        and (now - _cache_timestamp < SCHEMA_CACHE_TTL)
    ):
        return _schema_cache

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    labels: list[str] = []
    rel_types: list[str] = []
    label_properties: dict[str, list[str]] = {}
    rel_properties: dict[str, list[str]] = {}
    label_samples: dict[str, list[str]] = {}

    try:
        with driver.session() as session:
            # ---- Labels ----
            lbl_result = session.run("CALL db.labels()")
            labels = [r[0] for r in lbl_result if r and r[0]]

            # ---- Relationship types ----
            rel_result = session.run("CALL db.relationshipTypes()")
            rel_types = [r[0] for r in rel_result if r and r[0]]

            # ---- Per-label property keys ----
            try:
                prop_result = session.run("CALL db.schema.nodeTypeProperties()")
                for record in prop_result:
                    lbl_list = record.get("nodeLabels") or record.get("nodeType") or []
                    prop_name = record.get("propertyName")
                    if isinstance(lbl_list, str):
                        lbl_list = [lbl_list]
                    for lbl in lbl_list:
                        if lbl and prop_name:
                            label_properties.setdefault(lbl, [])
                            if prop_name not in label_properties[lbl]:
                                label_properties[lbl].append(prop_name)
            except Exception as exc:
                print(f"[schema] db.schema.nodeTypeProperties() unavailable: {exc}")

            # ---- Per-relationship property keys ----
            try:
                rprop_result = session.run("CALL db.schema.relTypeProperties()")
                for record in rprop_result:
                    rt = record.get("relType") or ""
                    prop_name = record.get("propertyName")
                    # relType is often returned as ":`REL_NAME`"
                    rt = rt.strip().lstrip(":").strip("`")
                    if rt and prop_name:
                        rel_properties.setdefault(rt, [])
                        if prop_name not in rel_properties[rt]:
                            rel_properties[rt].append(prop_name)
            except Exception as exc:
                print(f"[schema] db.schema.relTypeProperties() unavailable: {exc}")

            # ---- Sample name values per label ----
            for lbl in labels:
                safe_lbl = lbl.replace("`", "")
                try:
                    sample_result = session.run(
                        f"MATCH (n:`{safe_lbl}`) WHERE n.name IS NOT NULL "
                        f"RETURN DISTINCT n.name AS name LIMIT {SCHEMA_SAMPLE_NAMES_PER_LABEL}"
                    )
                    label_samples[lbl] = [r["name"] for r in sample_result if r["name"]]
                except Exception as exc:
                    print(f"[schema] Could not fetch samples for '{lbl}': {exc}")
                    label_samples[lbl] = []

    except Exception as exc:
        print(f"[schema] ERROR fetching live schema: {exc}")
        if _schema_cache is not None:
            print("[schema] Returning stale cache.")
            return _schema_cache
        labels, rel_types = [], []
    finally:
        driver.close()

    schema = {
        "labels": labels,
        "labels_lower": {lbl.lower(): lbl for lbl in labels},
        "entity_types": labels,
        "relationship_types": rel_types,
        "rel_types_lower": {rt.lower(): rt for rt in rel_types},
        "label_properties": label_properties,
        "rel_properties": rel_properties,
        "label_samples": label_samples,
    }
    _schema_cache = schema
    _cache_timestamp = now
    print(
        f"[schema] Loaded: {len(labels)} labels, {len(rel_types)} rel types "
        f"(TTL={SCHEMA_CACHE_TTL}s)"
    )
    return schema
