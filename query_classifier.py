import ollama
from config import OLLAMA_HOST, EXTRACTION_MODEL

def classify_query(question: str) -> str:
    prompt = f"""
You are an intelligent query router. Classify the user's question into one of three categories:
1. "graph": The question asks about relationships, connections, paths, or structured entities (e.g., "Who does John work with?", "How is Project X connected to Team Y?").
2. "vector": The question asks for semantic concepts, descriptions, or general knowledge found in text (e.g., "What is the summary of the report?", "Explain the main idea of document Z.").
3. "hybrid": The question requires both structured relationships and semantic text retrieval (e.g., "Find all employees who worked on Project X and summarize their performance reviews.").

Examples:
- "Who is the manager of the IT department?" -> graph
- "What does the employee handbook say about PTO?" -> vector
- "Give me a summary of the projects John Smith is leading." -> hybrid

Respond with exactly one word: "graph", "vector", or "hybrid". If ambiguous, default to "hybrid".

Question: {question}
Classification:
"""
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        response = client.generate(model=EXTRACTION_MODEL, prompt=prompt)
        answer = response.get("response", "").strip().lower()
        if "graph" in answer: return "graph"
        if "vector" in answer: return "vector"
        if "hybrid" in answer: return "hybrid"
    except Exception as e:
        print(f"Classification error: {e}")
    
    return "hybrid"
