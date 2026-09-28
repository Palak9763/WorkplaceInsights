"""
FastAPI ingestion endpoint. Accepts a file upload (PDF, DOCX, image,
Excel, or CSV), routes it through the right parser, then runs the
existing extract -> Neo4j / Qdrant pipeline unchanged.

Run:  uvicorn ingest_api:app --reload
Then upload via http://localhost:8000/docs (FastAPI's built-in test UI)
"""
from datetime import date
from fastapi import FastAPI, UploadFile, File, HTTPException

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