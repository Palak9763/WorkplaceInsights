"""
Quick diagnostic — tests Groq, Google AI Studio, Qdrant, and Neo4j connectivity.
Run: python scripts/check_keys.py
"""
import sys
import json
import urllib.request
import urllib.error

# Force UTF-8 output on Windows terminals
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from config import (
    REASONING_PROVIDER,
    GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
    GOOGLE_API_KEY, GOOGLE_MODEL, GOOGLE_BASE_URL,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)

SEP = "=" * 55
print(SEP)
print(f"API KEY DIAGNOSTICS (Active Provider: {REASONING_PROVIDER})")
print(SEP)

# ── 1. Groq Cloud ───────────────────────────────────────────
print(f"\n[1] Groq API")
print(f"    Model : {GROQ_MODEL}")
if GROQ_API_KEY:
    print(f"    Key   : {GROQ_API_KEY[:10]}...{GROQ_API_KEY[-4:]}")
    try:
        url = f"{GROQ_BASE_URL}/chat/completions"
        payload = json.dumps({
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": "Reply with just: OK"}],
            "max_tokens": 10,
        }).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "User-Agent": "WorkplaceInsights/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        reply = data["choices"][0]["message"]["content"].strip()
        print(f"    STATUS: OK  (Groq replied: \"{reply}\")")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"    STATUS: FAIL  HTTP {e.code}")
        try:
            err = json.loads(body)
            print(f"    Detail: {err.get('error', {}).get('message', body[:200])}")
        except Exception:
            print(f"    Detail: {body[:200]}")
    except Exception as e:
        print(f"    STATUS: FAIL  {e}")
else:
    print("    STATUS: NOT CONFIGURED (GROQ_API_KEY is empty in .env)")

# ── 2. Google AI Studio ─────────────────────────────────────
print(f"\n[2] Google AI Studio")
print(f"    Model : {GOOGLE_MODEL}")
if GOOGLE_API_KEY:
    print(f"    Key   : {GOOGLE_API_KEY[:12]}...{GOOGLE_API_KEY[-4:]}")
    try:
        url = (
            f"{GOOGLE_BASE_URL}/models/{GOOGLE_MODEL}"
            f":generateContent?key={GOOGLE_API_KEY}"
        )
        payload = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": "Reply with just: OK"}]}],
            "generationConfig": {"maxOutputTokens": 200},
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
            reply = candidates[0]["content"]["parts"][0].get("text", "").strip()
            print(f"    STATUS: OK  (Google replied: \"{reply}\")")
        else:
            print(f"    STATUS: OK  (connected, finishReason: {candidates[0].get('finishReason') if candidates else 'none'})")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"    STATUS: FAIL  HTTP {e.code}")
        try:
            err = json.loads(body)
            print(f"    Detail: {err.get('error', {}).get('message', body[:200])}")
        except Exception:
            print(f"    Detail: {body[:200]}")
    except Exception as e:
        print(f"    STATUS: FAIL  {e}")
else:
    print("    STATUS: NOT CONFIGURED (GOOGLE_API_KEY is empty)")

# ── 3. Qdrant ────────────────────────────────────────────────
print(f"\n[3] Qdrant Cloud")
print(f"    URL        : {QDRANT_URL}")
print(f"    Collection : {QDRANT_COLLECTION}")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    info = client.get_collection(QDRANT_COLLECTION)
    count = info.points_count
    print(f"    STATUS: OK  ({count} points in collection)")
    payload_schema = info.payload_schema or {}
    text_idx = payload_schema.get("text")
    if text_idx:
        print(f"    text index : PRESENT  (type={text_idx.data_type})")
    else:
        print(f"    text index : MISSING")
except Exception as e:
    print(f"    STATUS: FAIL  {e}")

# ── 4. Neo4j ─────────────────────────────────────────────────
print(f"\n[4] Neo4j")
print(f"    URI  : {NEO4J_URI}")
print(f"    User : {NEO4J_USER}")
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN 1 AS ok")
        val = result.single()["ok"]
    driver.close()
    print(f"    STATUS: OK  (ping returned {val})")
except Exception as e:
    print(f"    STATUS: FAIL  {e}")

# ── 5. MongoDB ───────────────────────────────────────────────
print(f"\n[5] MongoDB (Auth, Chat History & Analytics)")
from config import MONGO_URI, MONGO_DB_NAME
print(f"    URI  : {MONGO_URI}")
print(f"    DB   : {MONGO_DB_NAME}")
try:
    from db_mongo import check_mongo_connection
    is_ok, msg = check_mongo_connection()
    if is_ok:
        print(f"    STATUS: OK  ({msg})")
    else:
        print(f"    STATUS: NOTICE ({msg})")
except Exception as e:
    print(f"    STATUS: FAIL  {e}")

print(f"\n{SEP}")
