"""
FastAPI ingestion + query endpoint.

Run:  uvicorn ingest_api:app --reload
Docs: http://localhost:8000/docs
"""
import time
import asyncio
from datetime import date
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from db_mongo import (
    register_user,
    authenticate_user,
    handle_oauth_user,
    decode_access_token,
    get_user_by_id,
    create_conversation,
    list_conversations,
    save_message,
    get_conversation_messages,
    delete_conversation,
    log_query_analytics,
    get_analytics_summary,
    check_mongo_connection,
)

from parsers import parse_file
from chunker import chunk_text
from extract import extract
from write_graph import write_to_graph
from write_vector import write_to_vector_store

from graph_schema import get_live_schema
from entity_linker import link_entities
from query_classifier import classify_query
from cypher_generator import generate_cypher
from cypher_validator import validate_cypher
from graph_traversal import (
    expand_anchors,
    find_paths_between_anchors,
    rank_and_build_paths,
    run_traversal_query,
)
from vector_retrieval import vector_retrieve
from reasoning import generate_answer
from config import (
    CYPHER_RETRIES,
    ANCHOR_FORCES_HYBRID,
    GRAPH_MAX_DEPTH,
    GRAPH_MAX_PATH_LEN,
    GRAPH_MAX_ROWS,
)

app = FastAPI(title="GraphRAG Ingestion & Query API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class UserRegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class UserLoginRequest(BaseModel):
    email_or_username: str
    password: str


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = "New Conversation"


class OAuthLoginRequest(BaseModel):
    provider: str  # "google" | "github"
    email: Optional[str] = ""
    name: Optional[str] = ""
    provider_id: str
    avatar_url: Optional[str] = ""
    token_or_code: Optional[str] = ""


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split("Bearer ")[1].strip()
    return decode_access_token(token)


def get_current_user_required(authorization: Optional[str] = Header(None)) -> dict:
    user = get_current_user_optional(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required or token expired.")
    return user


# ---------------------------------------------------------------------------
# /ingest  (unchanged logic)
# ---------------------------------------------------------------------------
@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename
    file_bytes = await file.read()
    source = f"upload/{filename}"
    today = str(date.today())

    try:
        parsed = parse_file(filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    pieces = chunk_text(parsed["text"]) if parsed["mode"] == "prose" else parsed["rows"]
    if not pieces:
        raise HTTPException(status_code=400, detail="No extractable text found in file.")

    results = {
        "filename": filename,
        "chunks_processed": 0,
        "chunks_failed": 0,
        "entities_extracted": 0,
        "relationships_extracted": 0,
    }
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
        write_to_vector_store(piece, source=source, date=today)
        results["chunks_processed"] += 1
    return results


# ---------------------------------------------------------------------------
# /query  — fully data-agnostic pipeline with graph traversal & semantic retrieval
# ---------------------------------------------------------------------------
@app.post("/query")
async def query_endpoint(request: QueryRequest, authorization: Optional[str] = Header(None)):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Identify user (if authenticated)
    user = get_current_user_optional(authorization)
    user_id = user["sub"] if user else "anonymous"
    conv_id = request.conversation_id

    # If conversation_id is provided or created, save user message
    if conv_id:
        try:
            save_message(conv_id, user_id, "user", question)
        except Exception as msg_err:
            print(f"[mongo] Notice saving user message: {msg_err}")

    diagnostics: dict = {}  # per-stage errors + timings

    # -----------------------------------------------------------------------
    # Stage 0: Live schema
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    schema = get_live_schema()
    diagnostics["schema_ms"] = round((time.perf_counter() - t0) * 1000)

    # -----------------------------------------------------------------------
    # Stage 1+2: Entity linking & Classification — run CONCURRENTLY
    # (linking hits Neo4j; classification hits Groq — fully independent)
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()

    anchors: list[dict] = []
    cls_dict: dict = {}

    try:
        link_future = loop.run_in_executor(None, link_entities, question, schema)
        classify_future = loop.run_in_executor(None, classify_query, question, schema, [])
        anchors_result, cls_dict_prelim = await asyncio.gather(link_future, classify_future)
        anchors = anchors_result
    except Exception as exc:
        diagnostics["linker_error"] = str(exc)
        print(f"[query] Entity linker/classify parallel error: {exc}")
        cls_dict_prelim = {"route": "hybrid", "multihop": False, "aggregation": False, "error": str(exc)}

    diagnostics["linker_ms"] = round((time.perf_counter() - t0) * 1000)
    diagnostics["anchor_count"] = len(anchors)

    # If anchors were found, re-classify with anchors context (fast, uses cached Groq)
    # Only re-classify if anchors changed the picture meaningfully
    if anchors:
        try:
            cls_dict = classify_query(question, schema, anchors)
        except Exception as exc:
            cls_dict = cls_dict_prelim
            diagnostics["classify_error"] = str(exc)
    else:
        cls_dict = cls_dict_prelim

    route = cls_dict["route"]
    multihop = cls_dict["multihop"]
    aggregation = cls_dict["aggregation"]
    if cls_dict.get("error"):
        diagnostics["classify_error"] = cls_dict["error"]
    diagnostics["classify_ms"] = round((time.perf_counter() - t0) * 1000)

    # Routing rules:
    # If anchors found and ANCHOR_FORCES_HYBRID -> run both graph & vector
    # Else follow classifier route
    run_graph = (len(anchors) > 0 and ANCHOR_FORCES_HYBRID) or route in ("graph", "hybrid")
    run_vector = (len(anchors) > 0 and ANCHOR_FORCES_HYBRID) or route in ("vector", "hybrid")

    # -----------------------------------------------------------------------
    # Stage 3: Graph retrieval (Traversal first, LLM Cypher fallback/aggregation)
    # -----------------------------------------------------------------------
    cypher_used = ""
    graph_raw_count = 0
    graph_result_count = 0
    graph_results: list[dict] = []
    graph_path: list[dict] = []
    graph_path_source = "none"
    graph_fallback_used = False

    if run_graph:
        t0 = time.perf_counter()
        anchor_ids = [a["element_id"] for a in anchors if a.get("element_id")]
        traversal_edges: list[dict] = []

        # 1. Traversal path (if not aggregation only)
        if not aggregation:
            # (a) If 2+ anchors, find connecting paths
            if len(anchor_ids) >= 2:
                p_edges, _, p_err = find_paths_between_anchors(anchor_ids, max_path_len=GRAPH_MAX_PATH_LEN)
                if p_err:
                    diagnostics["path_finding_error"] = p_err
                traversal_edges.extend(p_edges)

            # (b) Expand anchor neighborhoods
            depth = 2 if (multihop or GRAPH_MAX_DEPTH >= 2) else 1
            e_edges, e_err = expand_anchors(anchor_ids, depth=depth, question=question)
            if e_err:
                diagnostics["anchor_expansion_error"] = e_err
            traversal_edges.extend(e_edges)

            # Deduplicate traversal edges by rid
            seen_rids: set[str] = set()
            dedup_edges = []
            for edg in traversal_edges:
                rid = edg.get("rid")
                if rid and rid not in seen_rids:
                    seen_rids.add(rid)
                    dedup_edges.append(edg)

            if dedup_edges:
                graph_raw_count = len(dedup_edges)
                graph_results, graph_path = rank_and_build_paths(dedup_edges, question)
                graph_result_count = len(graph_results)
                if graph_results:
                    graph_path_source = "traversal"
                    print(f"[query] Graph traversal produced {graph_result_count} result(s).")

        # 2. LLM Cypher fallback (if traversal empty or query is aggregation)
        if not graph_results or aggregation:
            reason = "aggregation required" if aggregation else "traversal yielded 0 results"
            print(f"[query] Invoking LLM Cypher generator ({reason})...")

            cypher, gen_error = generate_cypher(question, schema, anchors)
            if gen_error:
                diagnostics["cypher_gen_error"] = gen_error

            is_valid, val_result = validate_cypher(cypher, schema)
            if not is_valid:
                diagnostics["cypher_val_attempt1"] = val_result
                print(f"[query] Validation failed (attempt 1): {val_result}")
                for attempt in range(2, CYPHER_RETRIES + 2):
                    cypher, gen_error = generate_cypher(question, schema, anchors, error_context=val_result)
                    if gen_error:
                        diagnostics[f"cypher_gen_error_a{attempt}"] = gen_error
                    is_valid, val_result = validate_cypher(cypher, schema)
                    if is_valid:
                        break
                    diagnostics[f"cypher_val_attempt{attempt}"] = val_result
                    print(f"[query] Validation failed (attempt {attempt}): {val_result}")

            if is_valid:
                cypher_used = val_result
                cypher_rows, retrieve_error = run_traversal_query(cypher_used)
                if retrieve_error:
                    diagnostics["cypher_retrieve_error"] = retrieve_error
                if cypher_rows:
                    graph_fallback_used = True
                    graph_raw_count = len(cypher_rows)
                    # Convert cypher rows to graph_results format
                    formatted_edges = []
                    for row in cypher_rows:
                        formatted_edges.append({
                            "rid": row.get("rid", "cypher-row"),
                            "source_name": row.get("name") or row.get("source_name") or str(row),
                            "source_labels": row.get("labels", []),
                            "relationship": row.get("relationship", "REL"),
                            "target_name": row.get("target") or row.get("target_name") or "",
                            "target_labels": row.get("target_labels", []),
                        })
                    graph_results, graph_path = rank_and_build_paths(formatted_edges, question)
                    graph_result_count = len(graph_results)
                    graph_path_source = "llm_cypher"
                    print(f"[query] LLM Cypher produced {graph_result_count} result(s).")
            else:
                diagnostics["cypher_final_error"] = val_result

        diagnostics["graph_ms"] = round((time.perf_counter() - t0) * 1000)

    # -----------------------------------------------------------------------
    # Stage 4: Semantic Vector retrieval
    # -----------------------------------------------------------------------
    vector_results: list[dict] = []
    vector_counts: dict = {}
    if run_vector:
        t0 = time.perf_counter()
        vector_results, vector_counts, vec_error = vector_retrieve(question, anchors)
        if vec_error:
            diagnostics["vector_error"] = vec_error
        diagnostics["vector_ms"] = round((time.perf_counter() - t0) * 1000)

    # -----------------------------------------------------------------------
    # Stage 5: Answer generation & Reasoning
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    answer, grounding_warning, reasoning_provider_used = generate_answer(
        question, graph_results, vector_results
    )
    diagnostics["reasoning_ms"] = round((time.perf_counter() - t0) * 1000)
    diagnostics["reasoning_provider_used"] = reasoning_provider_used

    # Effective classification reflects actual retrieval performed
    if graph_results and vector_results:
        effective_classification = "hybrid"
    elif graph_results and not vector_results:
        effective_classification = "graph"
    elif vector_results and not graph_results:
        effective_classification = "vector"
    else:
        effective_classification = route

    # -----------------------------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------------------------
    print("==================================================")
    print("QUERY COMPLETE")
    print("==================================================")
    print(f"  classification:     {effective_classification} (route={route}, multihop={multihop}, agg={aggregation})")
    print(f"  anchors:            {[a['name'] for a in anchors]}")
    print(f"  graph_path_source:  {graph_path_source}")
    print(f"  graph_raw_count:    {graph_raw_count}")
    print(f"  graph_result_count: {graph_result_count}")
    print(f"  vector chunks:      {len(vector_results)}")
    print(f"  reasoning_provider: {reasoning_provider_used}")
    print(f"  grounding_warning:  {grounding_warning}")
    print(f"  diagnostics:        {diagnostics}")
    print()

    response_payload = {
        "answer": answer,
        "classification": effective_classification,
        "anchors": anchors,
        "cypher_used": cypher_used,
        "graph_fallback_used": graph_fallback_used,
        "graph_raw_count": graph_raw_count,
        "graph_result_count": graph_result_count,
        "graph_path_source": graph_path_source,
        "graph_path": graph_path,
        "graph_results": graph_results,
        "vector_results": vector_results,
        "vector_counts": vector_counts,
        "grounding_warning": grounding_warning,
        "reasoning_provider_used": reasoning_provider_used,
        "diagnostics": diagnostics,
        "conversation_id": conv_id,
    }

    # Persist assistant response to MongoDB conversation
    if conv_id:
        try:
            save_message(
                conversation_id=conv_id,
                user_id=user_id,
                role="assistant",
                content=answer,
                metadata={
                    "classification": effective_classification,
                    "grounding_warning": grounding_warning,
                    "reasoning_provider_used": reasoning_provider_used,
                    "anchor_count": len(anchors),
                }
            )
        except Exception as save_err:
            print(f"[mongo] Notice saving assistant message: {save_err}")

    # Log query analytics & performance metrics
    try:
        log_query_analytics(
            question=question,
            answer=answer,
            route=effective_classification,
            multihop=multihop,
            anchors=anchors,
            graph_facts_count=graph_result_count,
            vector_chunks_count=len(vector_results),
            provider=reasoning_provider_used,
            diagnostics=diagnostics,
            grounding_warnings=grounding_warning,
            user_id=user_id,
            conversation_id=conv_id,
            status="success",
        )
    except Exception as log_err:
        print(f"[mongo] Notice logging query analytics: {log_err}")

    return response_payload


# ---------------------------------------------------------------------------
# Authentication Endpoints (JWT + MongoDB)
# ---------------------------------------------------------------------------
@app.post("/api/auth/register")
async def register(req: UserRegisterRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    try:
        return register_user(req.email, req.username, req.password)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/auth/login")
async def login(req: UserLoginRequest):
    try:
        return authenticate_user(req.email_or_username, req.password)
    except ValueError as val_err:
        raise HTTPException(status_code=401, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user_required)):
    user_data = get_user_by_id(user["sub"])
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found.")
    return user_data


@app.post("/api/auth/oauth")
async def oauth_login(req: OAuthLoginRequest):
    if req.provider not in ("google", "github"):
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider.")
    if not req.provider_id:
        raise HTTPException(status_code=400, detail="Missing provider ID.")

    try:
        return handle_oauth_user(
            provider=req.provider,
            email=req.email or "",
            name=req.name or "",
            provider_id=req.provider_id,
            avatar_url=req.avatar_url or "",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Persistent Chat History Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/conversations")
async def get_conversations(user: dict = Depends(get_current_user_required)):
    return list_conversations(user_id=user["sub"])


@app.post("/api/conversations")
async def create_new_conversation(
    req: ConversationCreateRequest,
    user: dict = Depends(get_current_user_required)
):
    return create_conversation(user_id=user["sub"], title=req.title or "New Conversation")


@app.get("/api/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    user: dict = Depends(get_current_user_required)
):
    return get_conversation_messages(conversation_id)


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: str,
    user: dict = Depends(get_current_user_required)
):
    success = delete_conversation(conversation_id, user["sub"])
    return {"success": success}


# ---------------------------------------------------------------------------
# Query Analytics & Monitoring
# ---------------------------------------------------------------------------
@app.get("/api/analytics")
async def get_analytics():
    return get_analytics_summary()


@app.get("/api/health/mongo")
async def mongo_health():
    is_ok, msg = check_mongo_connection()
    return {"connected": is_ok, "message": msg}


@app.get("/health")
async def health():
    return {"status": "ok"}