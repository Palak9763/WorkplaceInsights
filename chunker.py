"""
Splits parsed text into chunks for extraction.

KEY FIX: instead of blindly sliding a 400-word window across the ENTIRE
document (which merged all 8 employees in the NovaTech PDF into one
giant blob), this now respects natural document structure first:

1. Text is split on blank-line boundaries - since parsers already
   separate distinct pages/records this way (each employee = one PDF
   page = one block).
2. A block that looks like a standalone structured record (has 2+
   "Key: value" style lines, like an employee profile) is kept as its
   OWN chunk, even if small - one person per chunk, not eight.
3. Plain prose blocks (no structure, just regular paragraphs) get
   merged together up to the word cap, so short paragraphs don't
   become wastefully tiny chunks.
4. Any single block that's still too large gets the old sliding-window
   split applied, just to that block.
"""
import re

CHUNK_SIZE_WORDS = 400
OVERLAP_WORDS = 50


def _looks_like_structured_record(block: str) -> bool:
    """
    Heuristic: does this block look like a standalone record (e.g. one
    person's profile) rather than a fragment of continuous prose?
    True if it has 2+ lines matching "Key: value" style.
    """
    key_value_lines = re.findall(r'^[A-Za-z][A-Za-z\s]{1,25}:\s', block, flags=re.MULTILINE)
    return len(key_value_lines) >= 2


def _split_long_block(block: str) -> list[str]:
    """Sliding-window word split - used only when a single block is
    itself too large to send as one chunk."""
    words = block.split()
    if len(words) <= CHUNK_SIZE_WORDS:
        return [block] if block.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE_WORDS - OVERLAP_WORDS
    return chunks


def chunk_text(text: str) -> list[str]:
    raw_blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if not raw_blocks:
        return []

    chunks = []
    buffer = ""

    for block in raw_blocks:
        if _looks_like_structured_record(block):
            # Flush any accumulated plain-prose buffer first
            if buffer.strip():
                chunks.extend(_split_long_block(buffer.strip()))
                buffer = ""
            # This record stands alone - keep it as its own chunk
            chunks.extend(_split_long_block(block))
        else:
            # Plain prose - accumulate into buffer up to the word cap
            candidate = (buffer + "\n\n" + block).strip() if buffer else block
            if len(candidate.split()) > CHUNK_SIZE_WORDS:
                if buffer.strip():
                    chunks.extend(_split_long_block(buffer.strip()))
                buffer = block
            else:
                buffer = candidate

    if buffer.strip():
        chunks.extend(_split_long_block(buffer.strip()))

    return chunks