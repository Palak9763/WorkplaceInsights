"""
Calls qwen2.5:3b-instruct via Ollama to extract entities + relationships
from a chunk of text. Retries on: malformed JSON, wrong schema types,
OR a relationship referencing an entity that wasn't independently listed.
"""
import json
import ollama

from config import OLLAMA_HOST, EXTRACTION_MODEL
from schema import validate_extraction, ExtractionResult, _normalize

MAX_RETRIES = 3
SYSTEM_PROMPT = """You are an enterprise knowledge graph extraction system.

Extract ONLY entities and relationships that are explicitly supported by the provided text.

STRICT RULES:

1. Do NOT invent relationships.
2. Do NOT infer relationships from general knowledge.
3. Every entity must come from the text.
4. Every relationship must be directly supported by a sentence or statement in the text.
5. Every entity appearing in a relationship MUST also appear in the entities list.
6. Do NOT create a relationship just because two entities appear in the same sentence.
7. Preserve the exact entity names from the text.
8. Choose the relationship type based strictly on the wording of the text.
9. If an entity has no clearly supported relationship, do NOT include that entity.
10. Before returning the answer, verify that every relationship endpoint exists in the entities list.
11. You can dynamically use any entity type or relationship type as long as it fits the text.

Return ONLY valid JSON in exactly this format:

{
  "entities": [
    {
      "type": "ENTITY_TYPE",
      "name": "exact entity name"
    }
  ],
  "relationships": [
    {
      "from": "exact entity name",
      "type": "RELATIONSHIP_TYPE",
      "to": "exact entity name"
    }
  ]
}"""




def _call_model(text: str, error_context: str = "") -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    user_prompt = f"TEXT:\n{text}"
    if error_context:
        user_prompt += f"\n\nYour previous response was invalid: {error_context}\nTry again, returning ONLY valid JSON."

    response = client.chat(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


def ground_extraction(result: ExtractionResult):
    seen_entities = {}
    grounded_entities = []
    rejected_entities = []
    
    for e in result.entities:
        norm_name = _normalize(e.name)
        if norm_name not in seen_entities:
            seen_entities[norm_name] = e
            grounded_entities.append(e)
        else:
            rejected_entities.append(e)
            
    seen_relationships = set()
    grounded_relationships = []
    rejected_relationships = []
    
    for r in result.relationships:
        norm_from = _normalize(r.from_)
        norm_to = _normalize(r.to)
        
        if norm_from not in seen_entities or norm_to not in seen_entities:
            rejected_relationships.append(r)
            continue
            
        r.from_ = seen_entities[norm_from].name
        r.to = seen_entities[norm_to].name
        
        rel_key = (r.from_, r.type, r.to)
        if rel_key not in seen_relationships:
            seen_relationships.add(rel_key)
            grounded_relationships.append(r)
        else:
            rejected_relationships.append(r)
            
    return ExtractionResult(entities=grounded_entities, relationships=grounded_relationships), rejected_entities, rejected_relationships


def extract(text: str, chunk_id: str = "") -> ExtractionResult | None:
    error_context = ""

    for attempt in range(1, MAX_RETRIES + 1):
        raw_output = _call_model(text, error_context)

        print("==================================================")
        print("RAW QWEN EXTRACTION")
        print("==================================================")
        print(raw_output)
        print()

        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            print("==================================================")
            print("PARSED EXTRACTION")
            print("==================================================")
            print("Entities:")
            for e in parsed.get("entities", []):
                print(f"- {e}")
            print("\nRelationships:")
            for r in parsed.get("relationships", []):
                print(f"- {r}")
            print()
        except json.JSONDecodeError as e:
            print("==================================================")
            print("VALIDATION")
            print("==================================================")
            print(f"✗ Validation failed:\nJSON parse error: {e}")
            print()
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Invalid JSON: {e}")
            error_context = f"JSON parse error: {e}. Raw output was: {raw_output[:200]}"
            continue

        result, error = validate_extraction(parsed)
        
        print("==================================================")
        print("VALIDATION")
        print("==================================================")
        if result is not None:
            print("✓ Extraction valid")
            print()
            
            grounded_result, rejected_entities, rejected_relationships = ground_extraction(result)
            
            print("==================================================")
            print("DYNAMIC GROUNDING")
            print("==================================================")
            print()
            print("Accepted entities:")
            for e in grounded_result.entities:
                print(f"- {e}")
            print("\nRejected entities:")
            for e in rejected_entities:
                print(f"- {e}")
            print("\nAccepted relationships:")
            for r in grounded_result.relationships:
                print(f"- {r}")
            print("\nRejected relationships:")
            for r in rejected_relationships:
                print(f"- {r}")
            print()
            
            print("==================================================")
            print("FINAL GRAPH DATA")
            print("==================================================")
            print()
            print("Entities:")
            for e in grounded_result.entities:
                print(f"- {e}")
            print("\nRelationships:")
            for r in grounded_result.relationships:
                print(f"- {r}")
            print()
            
            print("==================================================")
            print("NEO4J INSERTION")
            print("==================================================")
            print("Writing validated entities and relationships...")
            print()

            print(f"  [attempt {attempt}/{MAX_RETRIES}] Extracted {len(grounded_result.entities)} entities, "
                  f"{len(grounded_result.relationships)} relationships.")
            return grounded_result
        else:
            print(f"✗ Validation failed:\n{error}")
            print()

        print(f"  [attempt {attempt}/{MAX_RETRIES}] Validation failed: {error}")
        error_context = error

    print(f"  FAILED after {MAX_RETRIES} attempts on chunk '{chunk_id}'. Flagging for manual review.")
    return None