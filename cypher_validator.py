from msilib import schema
import re

def validate_cypher(cypher: str, schema: dict) -> tuple[bool, str]:
    if not cypher:
        return False, "Empty Cypher query"
        
    cypher_upper = cypher.upper()
    
    # a) Reject mutating clauses
    forbidden = ["CREATE", "DELETE", "DETACH", "SET", "MERGE", "REMOVE"]
    for word in forbidden:
        if re.search(rf"\b{word}\b", cypher_upper):
            return False, f"Forbidden clause '{word}' found in query."
            
    # b) Validate labels and relationship types
    tokens_after_colon = set(re.findall(r":`([^`]+)`|:([A-Za-z0-9_]+)", cypher))
    tokens_after_colon = {a or b for a, b in tokens_after_colon}
   
    
    valid_labels = set(schema.get("labels", []))
    valid_rels = set(schema.get("relationship_types", []))
    valid_all = valid_labels.union(valid_rels)
    
    for token in tokens_after_colon:
        if token not in valid_all:
            return False, f"Unknown label or relationship type: {token}"
            
    # c) Add LIMIT 25 if missing
    if "LIMIT " not in cypher_upper:
        cypher = cypher.strip() + " LIMIT 25"
        
    return True, cypher
