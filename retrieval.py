from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from embed import embed

def graph_retrieve(cypher: str) -> list[dict]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []
    try:
        with driver.session() as session:
            record_iter = session.run(cypher)
            for record in record_iter:
                results.append(record.data())
    except Exception as e:
        print(f"Graph retrieval error: {e}")
    finally:
        driver.close()
    return results

def vector_retrieve(question: str, top_k: int = 5) -> list[dict]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        query_vector = embed(question)
        search_result = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k
        )
        
        results = []
        for hit in search_result:
            results.append(hit.payload)
        return results
    except Exception as e:
        print(f"Vector retrieval error: {e}")
        return []
