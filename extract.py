"""
Calls qwen2.5:3b-instruct via Ollama to extract entities + relationships
from a chunk of text. Includes the retry loop we discussed - if the
model returns malformed JSON, we re-prompt with the error, up to
MAX_RETRIES times, before giving up on that chunk.
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

Rules:
- Only extract what is explicitly stated. Do not infer relationships
  that aren't directly supported by the text.
- "location" is optional - omit it entirely for non-Person entities,
  or if a Person's location isn't mentioned.
- Use the exact entity names as they appear in the source text so they
  can be matched consistently across documents.

Example:
Text: "Priya Sharma (London) led the backend migration for Project Atlas."
Output: {"entities": [{"type": "Person", "name": "Priya Sharma", "location": "London"}, {"type": "Project", "name": "Project Atlas"}], "relationships": [{"from": "Priya Sharma", "type": "WORKS_ON", "to": "Project Atlas"}]}
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
        options={"temperature": 0.1},  # low temperature - we want consistent, literal extraction, not creativity
    )
    return response["message"]["content"]


def extract(text: str, chunk_id: str = "") -> ExtractionResult | None:
    """
    Extract entities/relationships from a chunk of text.
    Returns None if extraction fails after MAX_RETRIES attempts.
    """
    error_context = ""

    for attempt in range(1, MAX_RETRIES + 1):
        raw_output = _call_model(text, error_context)

        # Models sometimes wrap JSON in markdown fences despite instructions - strip if present
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

        result = validate_extraction(parsed)
        if result is not None:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Extracted {len(result.entities)} entities, "
                  f"{len(result.relationships)} relationships.")
            return result

        error_context = "JSON parsed but didn't match the required schema (check entity/relationship types)."

    print(f"  FAILED after {MAX_RETRIES} attempts on chunk '{chunk_id}'. Flagging for manual review.")
    return None
