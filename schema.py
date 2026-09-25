"""
The extraction schema, with full-graph validation - not just checking
individual fields, but checking the SHAPE of what gets extracted:
no orphaned entities, no placeholder/hallucinated names, no self-loops.
"""
import re
from pydantic import BaseModel, ValidationError
from typing import List, Literal

EntityType = Literal["Person", "Project", "Technology", "Service", "Incident"]

# Names that are almost always hallucinations/placeholders, not real entities
BLACKLISTED_NAMES = {
    "unknown", "n/a", "na", "none", "null", "unspecified", "undefined",
    "entity", "person", "project", "service", "technology", "incident",
    "", "the", "it", "this", "that",
}


class Entity(BaseModel):
    type: EntityType
    name: str
    location: str | None = None


class Relationship(BaseModel):
    from_: str
    type: str
    to: str

    class Config:
        populate_by_name = True
        fields = {"from_": "from"}


class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


def _normalize(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def validate_extraction(raw_json: dict) -> tuple[ExtractionResult | None, str | None]:
    """
    Returns (result, error_message). result is None if any check fails -
    error_message explains exactly what's wrong, fed back into the retry prompt.

    Checks, in order:
    1. Valid JSON matching the basic schema (types, structure)
    2. No blacklisted/placeholder entity names
    3. Every relationship's from/to must exist in entities (no phantom references)
    4. No self-loops (an entity relating to itself)
    5. If there's more than 1 entity, every entity must appear in at
       least one relationship (no orphans added just to dodge check #3)
    """
    try:
        for rel in raw_json.get("relationships", []):
            if "from" in rel:
                rel["from_"] = rel.pop("from")
        result = ExtractionResult(**raw_json)
    except ValidationError as e:
        return None, f"JSON parsed but didn't match the schema: {e}"
        
    for rel in result.relationships:
        if not rel.type.strip():
            return None, "Relationship 'type' cannot be empty or just whitespace."

    # Check 2: blacklisted/placeholder names
    bad_names = [
        e.name for e in result.entities
        if _normalize(e.name) in BLACKLISTED_NAMES or len(e.name.strip()) < 2
    ]
    if bad_names:
        bad_list = ", ".join(f"'{n}'" for n in bad_names)
        return None, (
            f"These are not real entity names, they look like placeholders "
            f"or hallucinations: {bad_list}. Remove them - only extract "
            f"entities that are explicitly named in the text."
        )

    entity_keys = {_normalize(e.name) for e in result.entities}

    # Check 3: every relationship name must exist in entities
    missing = set()
    for rel in result.relationships:
        if _normalize(rel.from_) not in entity_keys:
            missing.add(rel.from_)
        if _normalize(rel.to) not in entity_keys:
            missing.add(rel.to)
    if missing:
        missing_list = ", ".join(f"'{m}'" for m in missing)
        return None, (
            f"These names were used in relationships but not listed in "
            f"'entities': {missing_list}. Every name mentioned in a "
            f"relationship must also appear as its own entity."
        )

    # Check 4: no self-loops
    self_loops = [
        rel for rel in result.relationships
        if _normalize(rel.from_) == _normalize(rel.to)
    ]
    if self_loops:
        return None, (
            f"Found a relationship where 'from' and 'to' are the same "
            f"entity ('{self_loops[0].from_}'). An entity cannot relate to itself."
        )

    # Check 5: no orphaned entities (added but never actually connected)
    if len(result.entities) > 1:
        used_keys = set()
        for rel in result.relationships:
            used_keys.add(_normalize(rel.from_))
            used_keys.add(_normalize(rel.to))

        orphans = [e.name for e in result.entities if _normalize(e.name) not in used_keys]
        if orphans:
            orphan_list = ", ".join(f"'{o}'" for o in orphans)
            return None, (
                f"These entities were listed but have NO relationship "
                f"connecting them to anything: {orphan_list}. Either add "
                f"the relationship connecting them to another entity "
                f"(check the text again), or remove them if they're not "
                f"actually mentioned."
            )

    return result, None