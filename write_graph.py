"""
Writes an ExtractionResult into Neo4j.

Uses MERGE (not CREATE) so re-running extraction on overlapping text
doesn't create duplicate nodes - this is a basic form of the entity
resolution we discussed (exact-name matching; fuzzy matching for name
variants like "J. Sharma" vs "Priya Sharma" would be a later upgrade).
"""
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from schema import ExtractionResult


def write_to_graph(result: ExtractionResult, source: str = ""):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # 1. Write entities (MERGE avoids duplicates on re-run)
        for entity in result.entities:
            session.run(
                f"""
                MERGE (n:{entity.type} {{name: $name}})
                SET n.location = $location, n.last_seen_source = $source
                """,
                name=entity.name,
                location=entity.location,
                source=source,
            )

        # 2. Write relationships
        for rel in result.relationships:
            # Neo4j relationship types can't be parameterized directly - but since
            # rel.type is constrained by our Pydantic Literal schema, this is safe
            # (not raw user input - it's already been validated against a fixed set).
            session.run(
                f"""
                MATCH (a {{name: $from_name}})
                MATCH (b {{name: $to_name}})
                MERGE (a)-[:{rel.type}]->(b)
                """,
                from_name=rel.from_,
                to_name=rel.to,
            )

    driver.close()
    print(f"  Wrote {len(result.entities)} entities + {len(result.relationships)} relationships to Neo4j.")
