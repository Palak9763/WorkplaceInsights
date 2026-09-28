"""
Generates a Cypher query for a natural-language question, using the LIVE
schema (labels + relationship types actually in the graph right now).

KEY FIX: never generates exact {name: '...'} matches, since entity names
can vary slightly between how the question phrases them and how they're
actually stored (e.g. "Atlas" vs "Project Atlas"). Always uses a
case-insensitive CONTAINS match on the most distinctive keyword instead.
"""
import ollama
from config import OLLAMA_HOST, EXTRACTION_MODEL

SYSTEM_PROMPT = """You write Cypher queries for a Neo4j graph database, given
a natural-language question and the ACTUAL labels/relationship types that
currently exist in the graph.

CRITICAL RULE - NAME MATCHING:
NEVER use exact property matching like {name: 'Atlas'} or WHERE n.name = 'Atlas'.
Entity names in the graph may be longer or slightly different than how the
question phrases them (e.g. the question says "Atlas" but the actual node
is named "Project Atlas"). ALWAYS use case-insensitive partial matching instead:

  WHERE toLower(node.name) CONTAINS toLower('keyword')

Pick the single most distinctive keyword from the name mentioned in the
question - not the whole phrase, not generic words. For "Project Atlas",
use the keyword "atlas". For "Auth Service", use the keyword "auth".

Other rules:
- Only use labels and relationship types from the "Available schema" list
  below - never invent one that isn't listed.
- Only generate read-only queries: MATCH, WHERE, RETURN, ORDER BY, LIMIT.
  Never CREATE, DELETE, MERGE, SET, REMOVE, DETACH.
- Always include a LIMIT clause (25 if the question doesn't imply a number).
- Return ONLY the raw Cypher query, nothing else - no explanation, no
  markdown code fences, no preamble.

Example:
Question: "Who works on Project Atlas?"
Available schema - labels: [Person, Project], relationship types: [WORKS_ON]
Output: MATCH (p:Person)-[:WORKS_ON]->(pr:Project) WHERE toLower(pr.name) CONTAINS toLower('atlas') RETURN p.name LIMIT 25
"""


def generate_cypher(question: str, schema: dict, error_context: str = "") -> str:
    """
    schema is expected as {"labels": [...], "relationship_types": [...]}
    from graph_schema.get_live_schema().
    """
    client = ollama.Client(host=OLLAMA_HOST)

    user_prompt = (
        f"Question: \"{question}\"\n\n"
        f"Available schema - labels: {schema['labels']}, "
        f"relationship types: {schema['relationship_types']}"
    )

    if error_context:
        user_prompt += (
            f"\n\nYour previous query was invalid: {error_context}\n"
            f"Try again, following all the rules above, especially the "
            f"CONTAINS-based name matching rule."
        )

    response = client.chat(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )

    cypher = response["message"]["content"].strip()

    # Strip markdown fences if the model added them despite instructions
    if cypher.startswith("```"):
        cypher = cypher.split("```")[1]
        if cypher.lower().startswith("cypher"):
            cypher = cypher[6:]
    cypher = cypher.strip().rstrip(";")

    print(f"  Generated Cypher: {cypher}")
    return cypher