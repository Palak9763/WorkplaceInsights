"""
FastAPI ingestion endpoint. Accepts a file upload (PDF, DOCX, image,
Excel, or CSV), routes it through the right parser, then runs the
existing extract -> Neo4j / Qdrant pipeline unchanged.

Run:  uvicorn ingest_api:app --reload
Then upload via http://localhost:8000/docs (FastAPI's built-in test UI)
"""
from datetime import date
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from graph_schema import get_live_schema
from query_classifier import classify_query
from cypher_generator import generate_cypher
from cypher_validator import validate_cypher
from retrieval import graph_retrieve, vector_retrieve
from reasoning import generate_answer

class QueryRequest(BaseModel):
    question: str

from parsers import parse_file
from chunker import chunk_text
from extract import extract
from write_graph import write_to_graph
from write_vector import write_to_vector_store

app = FastAPI(title="GraphRAG Ingestion API")


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename
    file_bytes = await file.read()
    source = f"upload/{filename}"
    today = str(date.today())

    try:
        parsed = parse_file(filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build the list of text pieces to run through the pipeline,
    # depending on which path the file took
    if parsed["mode"] == "prose":
        pieces = chunk_text(parsed["text"])
    else:  # "rows" - Excel/CSV, each row is already its own small piece
        pieces = parsed["rows"]

    if not pieces:
        raise HTTPException(status_code=400, detail="No extractable text found in file.")

    results = {"filename": filename, "chunks_processed": 0, "chunks_failed": 0,
               "entities_extracted": 0, "relationships_extracted": 0}

    for i, piece in enumerate(pieces, start=1):
        chunk_id = f"{filename}-chunk-{i}"
        print(f"Processing {chunk_id}...")

        extraction = extract(piece, chunk_id=chunk_id)

        if extraction is not None:
            write_to_graph(extraction, source=source)
            results["entities_extracted"] += len(extraction.entities)
            results["relationships_extracted"] += len(extraction.relationships)
        else:
            results["chunks_failed"] += 1

        # Vector store gets the raw text regardless of extraction success
        write_to_vector_store(piece, source=source, date=today)
        results["chunks_processed"] += 1

    return results


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    question = req.question
    print(f"--- Processing Query: '{question}' ---")
    
    # 1. Classify
    classification = classify_query(question)
    print(f"Classification: {classification}")
    
    cypher_used = None
    graph_results = []
    vector_results = []
    
    # 2. Graph Retrieval
    if classification in ["graph", "hybrid"]:
        schema = get_live_schema()
        print("Fetched live graph schema.")
        
        cypher = generate_cypher(question, schema)
        print(f"Generated Cypher:\n{cypher}")
        
        is_valid, cypher_or_err = validate_cypher(cypher, schema)
        
        if not is_valid:
            print(f"Validation failed: {cypher_or_err}. Retrying once...")
            # Retry generation with error message fed back
            error_msg = cypher_or_err
            retry_question = f"{question} (Note: Your previous Cypher failed validation with error: {error_msg}. Please fix it.)"
            cypher = generate_cypher(retry_question, schema)
            print(f"Retry Generated Cypher:\n{cypher}")
            is_valid, cypher_or_err = validate_cypher(cypher, schema)
            
        if is_valid:
            cypher_used = cypher_or_err
            print("Validation passed. Running graph retrieval...")
            graph_results = graph_retrieve(cypher_used)
            print(f"Graph retrieval returned {len(graph_results)} results.")
        else:
            print(f"Validation failed again on retry: {cypher_or_err}. Skipping graph retrieval.")
            
    # 3. Vector Retrieval
    if classification in ["vector", "hybrid"]:
        print("Running vector retrieval...")
        vector_results = vector_retrieve(question)
        print(f"Vector retrieval returned {len(vector_results)} chunks.")
        
    # 4. Reasoning
    print("Generating answer...")
    answer = generate_answer(question, graph_results, vector_results)
    print("Answer generated.")
    
    return {
        "answer": answer,
        "classification": classification,
        "cypher_used": cypher_used,
        "graph_results": graph_results,
        "vector_results": vector_results
    }