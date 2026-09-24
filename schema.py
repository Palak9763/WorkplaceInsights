"""
The extraction schema. This is what we force the LLM's JSON output to
match. Keep this list of types TIGHT - fewer allowed types means fewer
ways for a 3B model to go wrong (we discussed this in the optimization
conversation - schema strictness matters more than model size here).
"""
from pydantic import BaseModel, ValidationError
from typing import List, Literal

EntityType = Literal["Person", "Project", "Technology", "Service", "Incident"]
RelationshipType = Literal["WORKS_ON", "USES", "CAUSED_DELAY_TO", "INVOLVED"]


class Entity(BaseModel):
    type: EntityType
    name: str
    location: str | None = None  # optional, only Person entities tend to have this


class Relationship(BaseModel):
    from_: str
    type: RelationshipType
    to: str

    class Config:
        # allows us to use "from" as the JSON key even though it's a Python keyword
        populate_by_name = True
        fields = {"from_": "from"}


class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


def validate_extraction(raw_json: dict) -> ExtractionResult | None:
    """Returns a validated ExtractionResult, or None if it doesn't match schema."""
    try:
        # normalize "from" -> "from_" for Python
        for rel in raw_json.get("relationships", []):
            if "from" in rel:
                rel["from_"] = rel.pop("from")
        return ExtractionResult(**raw_json)
    except ValidationError as e:
        print(f"Schema validation failed: {e}")
        return None
