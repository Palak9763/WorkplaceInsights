"""
Writes an ExtractionResult into Neo4j.

Uses MERGE (not CREATE) so re-running extraction on overlapping text
doesn't create duplicate nodes. Names are normalized before matching so
"Auth Service" and "The Auth Service" are treated as the same entity
instead of creating duplicates.
"""
import re
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from schema import ExtractionResult


def normalize_name(name: str) -> str:
    """
    Produces a consistent key for matching, without changing the
    display name. Strips leading "the"/"a"/"an", trims whitespace,
    collapses multiple spaces, and lowercases for comparison.
    """
    cleaned = name.strip()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def write_to_graph(result: ExtractionResult, source: str = ""):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # 1. Write entities - MERGE on the normalized key, but keep the
        #    original nicely-formatted name as the display name
        for entity in result.entities:
            key = normalize_name(entity.name)
            session.run(
                f"""
                MERGE (n:{entity.type} {{name_key: $key}})
                ON CREATE SET n.name = $name
                SET n.location = $location, n.last_seen_source = $source
                """,
                key=key,
                name=entity.name,
                location=entity.location,
                source=source,
            )

        # 2. Write relationships - match on the same normalized key
        for rel in result.relationships:
            from_key = normalize_name(rel.from_)
            to_key = normalize_name(rel.to)

            check = session.run(
                """
                MATCH (a {name_key: $from_key})
                MATCH (b {name_key: $to_key})
                RETURN a.name AS a_name, b.name AS b_name
                """,
                from_key=from_key,
                to_key=to_key,
            ).single()

            if check is None:
                print(f"  WARNING: could not match nodes for relationship "
                      f"'{rel.from_}' -{rel.type}-> '{rel.to}' - skipped. "
                      f"(from_key='{from_key}', to_key='{to_key}')")
                continue

            session.run(
                f"""
                MATCH (a {{name_key: $from_key}})
                MATCH (b {{name_key: $to_key}})
                MERGE (a)-[:{rel.type}]->(b)
                """,
                from_key=from_key,
                to_key=to_key,
            )

    driver.close()
    print(f"  Wrote {len(result.entities)} entities + {len(result.relationships)} relationships to Neo4j.")