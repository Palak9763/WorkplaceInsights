import json
import ollama
from config import OLLAMA_HOST, EXTRACTION_MODEL

def generate_answer(question: str, graph_results: list, vector_results: list) -> str:
    if not graph_results and not vector_results:
        return "I don't have enough information to answer that."
        
    context_parts = []
    
    if graph_results:
        context_parts.append("Graph Context:")
        context_parts.append(json.dumps(graph_results, indent=2))
        
    if vector_results:
        context_parts.append("Text Context:")
        for doc in vector_results:
            source = doc.get("source", "Unknown")
            text = doc.get("text", "")
            context_parts.append(f"Source [{source}]: {text}")
            
    context_str = "\n\n".join(context_parts)
    
    prompt = f"""
You are an AI assistant answering a user's question based on the provided context.
The context may contain structured graph data and/or unstructured text from documents.

Instructions:
1. Answer the question in plain English using ONLY the provided context.
2. Cite the sources that support your claims. For text, cite the document name (Source [name]). For graph facts, mention the relationship path or entities involved.
3. If the context does not contain enough information to answer the question fully, say so.

Context:
{context_str}

Question:
{question}

Answer:
"""
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        response = client.generate(model=EXTRACTION_MODEL, prompt=prompt)
        return response.get("response", "").strip()
    except Exception as e:
        print(f"Reasoning error: {e}")
        return "Sorry, there was an error generating the answer."
