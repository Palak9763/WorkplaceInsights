"""
Answer generation with grounding check.

Steps:
1. Render graph rows as directed sentences.
2. If both graph and vector results are empty, return the "not enough info" string
   without calling the LLM.
3. Call the LLM once with graph sentences + vector chunks.
4. Grounding check: extract capitalized tokens and numbers from the answer;
   verify each appears in the provided context. If any doesn't, regenerate once
   with the suspicious tokens listed as forbidden. If still failing, return the
   answer with a grounding_warning list.

Returns (answer: str, grounding_warning: list[str])
"""
import re
import json
import ollama
from config import OLLAMA_HOST, REASONING_MODEL

_NOT_ENOUGH = "I don't have enough information to answer that."

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.

RULES (follow strictly):
1. Answer using ONLY the Graph Facts and Document Chunks provided below.
2. Attribute each claim only to the specific subject, date, and document where it appears.
3. Inline-cite every claim using [Source: <source_name>] or [Graph: <sentence>].
4. COMPREHENSIVENESS / MULTI-EVENT SYNTHESIS: If the context describes multiple distinct incidents, outages, or delays across different dates/times (for example, an August delay AND a September delay), your answer MUST report BOTH separate events with their specific details (dates, durations, causes, and impacts). Never omit or drop one event in favor of another.
5. Keep the answer concise, factual, and complete. Never invent details not in the context.
"""


def _rows_to_sentences(rows: list[dict]) -> list[str]:
    """Convert graph result rows to directed sentence strings."""
    sentences = []
    for row in rows:
        if "_sentence" in row:
            sentences.append(row["_sentence"])
        elif "a_is_source" in row:
            a = row.get("a_name", "?")
            rel = row.get("rel", "?")
            b = row.get("b_name", "?")
            sent = f"{a} -[{rel}]-> {b}" if row["a_is_source"] else f"{b} -[{rel}]-> {a}"
            sentences.append(sent)
        elif "relationship" in row:
            subj = row.get("name", "?")
            rel = row.get("relationship", "?")
            tgt = row.get("target", "?")
            sentences.append(f"{subj} -[{rel}]-> {tgt}")
        else:
            sentences.append(json.dumps(row, default=str))
    return sentences


_IGNORE_META_TOKENS = {
    "source", "graph", "according", "based", "the", "in", "this", "there",
    "yes", "no", "none", "confluence", "slack", "upload", "doc", "document"
}


def _extract_suspicious_tokens(answer: str) -> list[str]:
    """
    Extracts capitalized words/phrases and numbers from the answer prose
    (excluding bracketed inline citations) that could be invented hallucinations.
    """
    # Strip bracketed citations first, e.g. [Source: ...] or [Graph: ...]
    clean_prose = re.sub(r"\[(?:Source|Graph):[^\]]*\]", "", answer)

    tokens: list[str] = []
    # Multi-word capitalized names
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9]*)(?:\s+[A-Z][a-zA-Z0-9]*)*\b", clean_prose):
        tok = m.group(0).strip()
        if tok.lower() not in _IGNORE_META_TOKENS and len(tok) > 1:
            tokens.append(tok)
    # Standalone numbers
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", clean_prose):
        tokens.append(m.group(0).strip())
    # Deduplicate
    seen: set[str] = set()
    result = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _token_in_context(token: str, context: str) -> bool:
    return token.lower() in context.lower()


def _call_llm(question: str, graph_sentences: list[str], vector_results: list[dict],
               forbidden: list[str] | None = None) -> str:
    graph_block = (
        "\n".join(f"  {i+1}. {s}" for i, s in enumerate(graph_sentences))
        if graph_sentences else "  (none)"
    )

    vector_lines = []
    if vector_results:
        for i, c in enumerate(vector_results, start=1):
            src = c.get("source", "unknown")
            dt = c.get("date", "unknown")
            txt = c.get("text", "")
            vector_lines.append(f"[Document {i}] (Source: {src}, Date: {dt}):\n{txt}")
        vector_block = "\n\n".join(vector_lines)
    else:
        vector_block = "  (none)"

    forbidden_note = ""
    if forbidden:
        forbidden_note = (
            f"\n\nDo NOT use or mention these tokens — they are NOT in the context: "
            f"{forbidden}"
        )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"GRAPH FACTS:\n{graph_block}\n\n"
        f"DOCUMENT CHUNKS:\n{vector_block}"
        f"{forbidden_note}\n\n"
        "Write a concise answer with inline citations, making sure to report all distinct incidents/events found:"
    )

    print(f"\n[reasoning] === SYSTEM PROMPT ===\n{_SYSTEM_PROMPT}")
    print(f"\n[reasoning] === USER PROMPT ===\n{user_prompt}\n")

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=REASONING_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"].strip()



def generate_answer(
    question: str, graph_results: list[dict], vector_results: list[dict]
) -> tuple[str, list[str]]:
    """
    Returns (answer: str, grounding_warning: list[str]).
    grounding_warning is [] when no hallucination was detected.
    """
    print("==================================================")
    print("REASONING & ANSWER GENERATION")
    print("==================================================")
    print(f"Question: {question}")
    print(f"Graph rows: {len(graph_results)}  |  Vector chunks: {len(vector_results)}")

    if not graph_results and not vector_results:
        print(f"[answer] No context -> returning 'not enough info'")
        return _NOT_ENOUGH, []

    graph_sentences = _rows_to_sentences(graph_results)
    # Build a single context string for grounding checks
    context = (
        "\n".join(graph_sentences)
        + "\n"
        + "\n".join(
            f"{c.get('text', '')} {c.get('source', '')}"
            for c in vector_results
        )
    )

    # --- First LLM call ---
    error: str | None = None
    try:
        answer = _call_llm(question, graph_sentences, vector_results)
    except Exception as exc:
        error = f"LLM error: {exc}"
        print(f"[answer] ERROR: {exc}")
        return f"I encountered an error generating the final answer. ({exc})", []

    # --- Grounding check ---
    suspicious = _extract_suspicious_tokens(answer)
    ungrounded = [t for t in suspicious if not _token_in_context(t, context)]

    grounding_warning: list[str] = []
    if ungrounded:
        print(f"[grounding] Suspicious tokens not in context: {ungrounded}")
        # One regeneration attempt with forbidden list
        try:
            answer2 = _call_llm(question, graph_sentences, vector_results, forbidden=ungrounded)
            suspicious2 = _extract_suspicious_tokens(answer2)
            still_ungrounded = [t for t in suspicious2 if not _token_in_context(t, context)]
            if still_ungrounded:
                grounding_warning = still_ungrounded
                answer = answer2  # use improved attempt even if imperfect
                print(f"[grounding] Still ungrounded after retry: {still_ungrounded}")
            else:
                answer = answer2
                print("[grounding] Regeneration resolved all grounding issues.")
        except Exception as exc2:
            print(f"[grounding] Regeneration LLM error: {exc2}")
            grounding_warning = ungrounded
    else:
        print("[grounding] All tokens grounded in context.")

    print(f"[ANSWER]\n{answer}\n")
    if grounding_warning:
        print(f"[grounding_warning] {grounding_warning}")
    return answer, grounding_warning
