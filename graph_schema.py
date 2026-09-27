import time
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

_schema_cache = None
_cache_timestamp = 0
CACHE_TTL = 30

def get_live_schema() -> dict:
    global _schema_cache, _cache_timestamp
    
    if _schema_cache is not None and time.time() - _cache_timestamp < CACHE_TTL:
        return _schema_cache
        
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    labels = []
    rel_types = []
    
    with driver.session() as session:
        labels_result = session.run("CALL db.labels()")
        labels = [record[0] for record in labels_result]
        
        rels_result = session.run("CALL db.relationshipTypes()")
        rel_types = [record[0] for record in rels_result]
        
    driver.close()
    
    _schema_cache = {
        "labels": labels,
        "relationship_types": rel_types
    }
    _cache_timestamp = time.time()
    
    return _schema_cache
