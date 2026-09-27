import ollama
from config import OLLAMA_HOST, EXTRACTION_MODEL

def generate_cypher(question: str, schema: dict) -> str:
    labels = ", ".join(schema.get("labels", []))
    rel_types = ", ".join(schema.get("relationship_types", []))
    
    prompt = f"""
You are a Cypher expert for Neo4j. Generate a Cypher query to answer the user's question based ONLY on the provided schema.

Schema Constraints:
- Available Node Labels: {labels}
- Available Relationship Types: {rel_types}
- You must use EXACTLY these labels and relationship types. Do not hallucinate any others.
- The graph nodes usually have a 'name' or 'name_key' property.
- Only return the raw Cypher query string. No explanations, no markdown blocks.

Question: {question}
Cypher Query:
"""
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        response = client.generate(model=EXTRACTION_MODEL, prompt=prompt)
        cypher = response.get("response", "").strip()
        
        # Strip markdown if model included it
        if cypher.startswith("```cypher"):
            cypher = cypher[len("```cypher"):].strip()
        if cypher.startswith("```"):
            cypher = cypher[len("```"):].strip()
        if cypher.endswith("```"):
            cypher = cypher[:-3].strip()
            
        return cypher
    except Exception as e:
        print(f"Cypher generation error: {e}")
        return ""
