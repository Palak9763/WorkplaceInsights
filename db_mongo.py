"""
MongoDB integration module for:
1. User Accounts & JWT Authentication (bcrypt + PyJWT)
2. Persistent Chat History (Conversations & Message threads)
3. Query Analytics & Latency Logging
"""
import uuid
import datetime
from typing import Optional, List, Dict, Any
import bcrypt
import jwt
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, ConnectionFailure

from config import (
    MONGO_URI,
    MONGO_DB_NAME,
    JWT_SECRET,
    JWT_ALGORITHM,
    JWT_EXPIRES_MINUTES,
)

# Global client cache
_client: Optional[MongoClient] = None
_db = None


def get_mongo_db():
    """Returns the MongoDB database instance, initializing connection if needed."""
    global _client, _db
    if _db is not None:
        return _db

    if not MONGO_URI:
        return None

    try:
        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )
        # Ping database
        _client.admin.command("ping")
        _db = _client[MONGO_DB_NAME]

        # Ensure indexes (idempotent)
        _ensure_indexes(_db)
        print(f"[mongo] Connected to MongoDB database: '{MONGO_DB_NAME}'")
        return _db
    except Exception as exc:
        print(f"[mongo] NOTICE: MongoDB not connected ({exc}). Chat history / auth will run in memory / fallback mode.")
        _db = None
        return None


def _ensure_indexes(db):
    """Creates indexes for users, conversations, messages, and analytics logs."""
    try:
        # Users: unique email & username
        db.users.create_index([("email", ASCENDING)], unique=True)
        db.users.create_index([("username", ASCENDING)], unique=True)

        # Conversations: user_id + updated_at
        db.conversations.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])

        # Messages: conversation_id + timestamp
        db.messages.create_index([("conversation_id", ASCENDING), ("timestamp", ASCENDING)])

        # Analytics logs: timestamp, user_id, route
        db.query_logs.create_index([("timestamp", DESCENDING)])
        db.query_logs.create_index([("user_id", ASCENDING)])
        db.query_logs.create_index([("route", ASCENDING)])
    except Exception as exc:
        print(f"[mongo] Index creation notice: {exc}")


def check_mongo_connection() -> tuple[bool, str]:
    """Tests MongoDB connection and returns (is_ok, message)."""
    try:
        db = get_mongo_db()
        if db is None:
            return False, "MongoDB connection failed or MONGO_URI is not set."
        db.command("ping")
        return True, "MongoDB connected successfully."
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# 1. USER ACCOUNTS & JWT AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(user_id: str, email: str, username: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXPIRES_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "username": username,
        "exp": expire,
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def register_user(email: str, username: str, password: str) -> dict:
    """Registers a new user in MongoDB."""
    db = get_mongo_db()
    if db is None:
        raise RuntimeError("MongoDB is not available. Please start MongoDB or check MONGO_URI in .env")

    # Check if exists
    if db.users.find_one({"email": email.lower()}):
        raise ValueError("A user with this email already exists.")
    if db.users.find_one({"username": username}):
        raise ValueError("A user with this username already exists.")

    user_id = str(uuid.uuid4())
    doc = {
        "_id": user_id,
        "email": email.lower().strip(),
        "username": username.strip(),
        "password_hash": hash_password(password),
        "created_at": datetime.datetime.utcnow(),
    }
    db.users.insert_one(doc)
    token = create_access_token(user_id, doc["email"], doc["username"])
    return {
        "user": {
            "id": user_id,
            "email": doc["email"],
            "username": doc["username"],
        },
        "token": token,
    }


def authenticate_user(email_or_username: str, password: str) -> dict:
    """Authenticates a user and returns their user profile + JWT token."""
    db = get_mongo_db()
    if db is None:
        raise RuntimeError("MongoDB is not available. Please start MongoDB or check MONGO_URI in .env")

    identifier = email_or_username.strip()
    user = db.users.find_one({
        "$or": [
            {"email": identifier.lower()},
            {"username": identifier}
        ]
    })

    if not user or not verify_password(password, user.get("password_hash", "")):
        raise ValueError("Invalid email/username or password.")

    token = create_access_token(str(user["_id"]), user["email"], user["username"])
    return {
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "username": user["username"],
        },
        "token": token,
    }


def get_user_by_id(user_id: str) -> Optional[dict]:
    db = get_mongo_db()
    if db is None:
        return None
    user = db.users.find_one({"_id": user_id})
    if user:
        return {
            "id": str(user["_id"]),
            "email": user["email"],
            "username": user["username"],
            "avatar_url": user.get("avatar_url", ""),
            "auth_provider": user.get("auth_provider", "local"),
            "created_at": user.get("created_at"),
        }
    return None


def handle_oauth_user(
    provider: str,
    email: str,
    name: str,
    provider_id: str,
    avatar_url: str = ""
) -> dict:
    """Handles Google & GitHub OAuth login/registration in MongoDB."""
    db = get_mongo_db()
    if db is None:
        raise RuntimeError("MongoDB is not available. Check MONGO_URI in .env")

    email_clean = email.lower().strip() if email else f"{provider}_{provider_id}@oauth.local"
    username = (name or email_clean.split("@")[0] or f"{provider}_user").strip()

    # Look for existing user by email or provider_id
    user = db.users.find_one({
        "$or": [
            {"email": email_clean},
            {"auth_provider": provider, "provider_id": provider_id}
        ]
    })

    now = datetime.datetime.utcnow()
    if user:
        user_id = str(user["_id"])
        # Update user profile
        db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "auth_provider": provider,
                    "provider_id": provider_id,
                    "avatar_url": avatar_url or user.get("avatar_url", ""),
                    "last_login": now,
                }
            }
        )
        final_username = user.get("username", username)
    else:
        # Create new OAuth user
        user_id = str(uuid.uuid4())
        # Ensure unique username
        existing_uname = db.users.find_one({"username": username})
        if existing_uname:
            username = f"{username}_{str(uuid.uuid4())[:4]}"

        doc = {
            "_id": user_id,
            "email": email_clean,
            "username": username,
            "auth_provider": provider,
            "provider_id": provider_id,
            "avatar_url": avatar_url,
            "created_at": now,
            "last_login": now,
        }
        db.users.insert_one(doc)
        final_username = username

    token = create_access_token(user_id, email_clean, final_username)
    return {
        "user": {
            "id": user_id,
            "email": email_clean,
            "username": final_username,
            "avatar_url": avatar_url,
            "auth_provider": provider,
        },
        "token": token,
    }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. PERSISTENT CHAT HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_conversation(user_id: str, title: str = "New Conversation") -> dict:
    """Creates a new conversation thread for the user."""
    db = get_mongo_db()
    conv_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()
    doc = {
        "_id": conv_id,
        "user_id": user_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }
    if db is not None:
        db.conversations.insert_one(doc)
    return {
        "id": conv_id,
        "user_id": user_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """Lists conversations for a given user, ordered by most recently updated."""
    db = get_mongo_db()
    if db is None:
        return []
    cursor = db.conversations.find({"user_id": user_id}).sort("updated_at", DESCENDING).limit(limit)
    res = []
    for c in cursor:
        res.append({
            "id": str(c["_id"]),
            "user_id": c.get("user_id"),
            "title": c.get("title", "Untitled Conversation"),
            "created_at": c.get("created_at", datetime.datetime.utcnow()).isoformat() if isinstance(c.get("created_at"), datetime.datetime) else str(c.get("created_at")),
            "updated_at": c.get("updated_at", datetime.datetime.utcnow()).isoformat() if isinstance(c.get("updated_at"), datetime.datetime) else str(c.get("updated_at")),
        })
    return res


def save_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None
) -> dict:
    """Saves a message (user or assistant) in a conversation."""
    db = get_mongo_db()
    msg_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()
    doc = {
        "_id": msg_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": role,  # 'user' | 'assistant'
        "content": content,
        "metadata": metadata or {},
        "timestamp": now,
    }

    if db is not None:
        db.messages.insert_one(doc)
        # Update conversation updated_at & title if first user message
        db.conversations.update_one(
            {"_id": conversation_id},
            {"$set": {"updated_at": now}}
        )
        if role == "user":
            # Auto-title from the first question if title is "New Conversation"
            conv = db.conversations.find_one({"_id": conversation_id})
            if conv and conv.get("title") in ("New Conversation", "Untitled Conversation", ""):
                short_title = content[:40] + ("..." if len(content) > 40 else "")
                db.conversations.update_one(
                    {"_id": conversation_id},
                    {"$set": {"title": short_title}}
                )

    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "timestamp": now.isoformat(),
    }


def get_conversation_messages(conversation_id: str, limit: int = 100) -> list[dict]:
    """Retrieves all messages for a given conversation thread."""
    db = get_mongo_db()
    if db is None:
        return []
    cursor = db.messages.find({"conversation_id": conversation_id}).sort("timestamp", ASCENDING).limit(limit)
    messages = []
    for m in cursor:
        messages.append({
            "id": str(m["_id"]),
            "conversation_id": m.get("conversation_id"),
            "role": m.get("role"),
            "content": m.get("content"),
            "metadata": m.get("metadata", {}),
            "timestamp": m.get("timestamp").isoformat() if isinstance(m.get("timestamp"), datetime.datetime) else str(m.get("timestamp")),
        })
    return messages


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Deletes a conversation and all its messages."""
    db = get_mongo_db()
    if db is None:
        return False
    db.conversations.delete_one({"_id": conversation_id, "user_id": user_id})
    db.messages.delete_many({"conversation_id": conversation_id})
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. QUERY ANALYTICS LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_query_analytics(
    question: str,
    answer: str,
    route: str,
    multihop: bool,
    anchors: list,
    graph_facts_count: int,
    vector_chunks_count: int,
    provider: str,
    diagnostics: dict,
    grounding_warnings: list,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    status: str = "success",
    error_message: Optional[str] = None,
):
    """Logs detailed telemetry and analytics for each query to MongoDB."""
    db = get_mongo_db()
    if db is None:
        return

    now = datetime.datetime.utcnow()
    total_latency_ms = sum(
        v for k, v in diagnostics.items()
        if isinstance(v, (int, float)) and k.endswith("_ms")
    ) if diagnostics else 0

    doc = {
        "_id": str(uuid.uuid4()),
        "user_id": user_id or "anonymous",
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer,
        "route": route,
        "multihop": multihop,
        "anchors": anchors,
        "anchor_count": len(anchors),
        "graph_facts_count": graph_facts_count,
        "vector_chunks_count": vector_chunks_count,
        "provider_used": provider,
        "diagnostics": diagnostics,
        "total_latency_ms": total_latency_ms,
        "grounding_warnings": grounding_warnings,
        "has_hallucination_warning": bool(grounding_warnings),
        "status": status,
        "error_message": error_message,
        "timestamp": now,
    }

    try:
        db.query_logs.insert_one(doc)
    except Exception as exc:
        print(f"[mongo] Analytics logging error: {exc}")


def get_analytics_summary(days: int = 7) -> dict:
    """Returns aggregated query analytics for dashboards."""
    db = get_mongo_db()
    if db is None:
        return {
            "total_queries": 0,
            "routes": {},
            "providers": {},
            "avg_latency_ms": 0,
            "recent_queries": [],
        }

    since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    filter_q = {"timestamp": {"$gte": since}}

    total_queries = db.query_logs.count_documents(filter_q)

    # Route distribution aggregation
    route_agg = db.query_logs.aggregate([
        {"$match": filter_q},
        {"$group": {"_id": "$route", "count": {"$sum": 1}}},
    ])
    routes = {r["_id"]: r["count"] for r in route_agg if r["_id"]}

    # Provider distribution aggregation
    provider_agg = db.query_logs.aggregate([
        {"$match": filter_q},
        {"$group": {"_id": "$provider_used", "count": {"$sum": 1}}},
    ])
    providers = {p["_id"]: p["count"] for p in provider_agg if p["_id"]}

    # Average latency
    latency_agg = db.query_logs.aggregate([
        {"$match": filter_q},
        {"$group": {"_id": None, "avg_latency": {"$avg": "$total_latency_ms"}}},
    ])
    avg_latency = 0
    for l in latency_agg:
        avg_latency = round(l.get("avg_latency", 0), 2)

    # Recent queries
    recent_cursor = db.query_logs.find(filter_q).sort("timestamp", DESCENDING).limit(20)
    recent = []
    for r in recent_cursor:
        recent.append({
            "id": str(r["_id"]),
            "question": r.get("question"),
            "route": r.get("route"),
            "provider_used": r.get("provider_used"),
            "total_latency_ms": r.get("total_latency_ms", 0),
            "status": r.get("status", "success"),
            "timestamp": r.get("timestamp").isoformat() if isinstance(r.get("timestamp"), datetime.datetime) else str(r.get("timestamp")),
        })

    return {
        "total_queries": total_queries,
        "routes": routes,
        "providers": providers,
        "avg_latency_ms": avg_latency,
        "recent_queries": recent,
    }
