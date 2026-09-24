"""
Calls qwen2.5:3b-instruct via Ollama to extract entities + relationships
from a chunk of text. Retries on: malformed JSON, wrong schema types,
OR a relationship referencing an entity that wasn't independently listed.
"""
import json
import ollama

from config import OLLAMA_HOST, EXTRACTION_MODEL
from schema import validate_extraction, ExtractionResult

MAX_RETRIES = 3
SYSTEM_PROMPT = """You extract entities and relationships from text into JSON.

Allowed entity types: Person, Project, Technology, Service, Incident
Allowed relationship types: WORKS_ON, USES, CAUSED_DELAY_TO, INVOLVED

Return ONLY valid JSON matching this exact structure, nothing else -
no preamble, no explanation, no markdown code fences:

{"entities": [{"type": "Person", "name": "...", "location": "..."}],
 "relationships": [{"from": "...", "type": "WORKS_ON", "to": "..."}]}

CRITICAL RULES:
1. Every name in "relationships" (from/to) MUST also appear as its own
   entry in "entities". Never reference a name without listing it.
2. Every entity in "entities" MUST be connected by at least one
   relationship. Never list an entity that isn't actually related to
   anything else in the text - if it has no relationship, leave it out
   entirely instead.
3. Never invent placeholder names like "Unknown", "N/A", or generic
   labels. Only extract names that are explicitly written in the text.
4. Only extract what is explicitly stated - do not infer relationships
   that aren't directly supported by the text.
5. "location" is optional - omit it for non-Person entities.

Example:
Text: "Project Atlas uses PostgreSQL as its database."
Output: {"entities": [{"type": "Project", "name": "Project Atlas"}, {"type": "Technology", "name": "PostgreSQL"}], "relationships": [{"from": "Project Atlas", "type": "USES", "to": "PostgreSQL"}]}
Both entities are connected by the USES relationship - nothing is orphaned, nothing is a placeholder.
"""




def _call_model(text: str, error_context: str = "") -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    user_prompt = f"Text: \"{text}\""
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


def extract(text: str, chunk_id: str = "") -> ExtractionResult | None:
    error_context = ""

    for attempt in range(1, MAX_RETRIES + 1):
        raw_output = _call_model(text, error_context)

        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Invalid JSON: {e}")
            error_context = f"JSON parse error: {e}. Raw output was: {raw_output[:200]}"
            continue

        result, error = validate_extraction(parsed)
        if result is not None:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Extracted {len(result.entities)} entities, "
                  f"{len(result.relationships)} relationships.")
            return result

        print(f"  [attempt {attempt}/{MAX_RETRIES}] Validation failed: {error}")
        error_context = error

    print(f"  FAILED after {MAX_RETRIES} attempts on chunk '{chunk_id}'. Flagging for manual review.")
    return None