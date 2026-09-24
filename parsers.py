"""
Extracts plain text from different file formats.
PDF/DOCX/images -> prose text (Path A, goes through LLM extraction).
Excel/CSV -> structured rows (Path B, converted to sentences directly).
"""
import fitz  # PyMuPDF
import docx
import pandas as pd
from io import BytesIO

from vision_ocr import read_image_text


def parse_pdf(file_bytes: bytes) -> str:
    """
    Extracts text per page. If a page has no text layer (scanned image),
    falls back to qwen2.5vl to read it as an image.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if text:
            full_text.append(text)
        else:
            # No text layer - this page is likely a scan. Render it as an
            # image and send to the vision model instead.
            print(f"  Page {page_num} has no text layer - using vision model...")
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            ocr_text = read_image_text(img_bytes)
            full_text.append(ocr_text)

    doc.close()
    return "\n\n".join(full_text)


def parse_docx(file_bytes: bytes) -> str:
    doc = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def parse_image(file_bytes: bytes) -> str:
    return read_image_text(file_bytes)


def parse_excel_or_csv(file_bytes: bytes, filename: str) -> list[str]:
    """
    Returns a list of sentence-ified rows (Path B - structured data).
    Each row becomes one plain-English sentence, e.g.:
    "Row: Name=Priya Sharma, Role=Engineer, Project=Atlas"
    This still gets extracted by the same LLM pipeline, but each row
    is kept as its own small chunk rather than free-flowing prose.
    """
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))

    sentences = []
    for _, row in df.iterrows():
        parts = [f"{col}={row[col]}" for col in df.columns if pd.notna(row[col])]
        sentences.append("Row: " + ", ".join(parts))

    return sentences


def parse_file(filename: str, file_bytes: bytes) -> dict:
    """
    Main entry point. Detects type from extension, returns:
    {"mode": "prose", "text": "..."}  for PDF/DOCX/images (Path A)
    {"mode": "rows", "rows": [...]}   for Excel/CSV (Path B)
    """
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        return {"mode": "prose", "text": parse_pdf(file_bytes)}
    elif ext == "docx":
        return {"mode": "prose", "text": parse_docx(file_bytes)}
    elif ext in ("png", "jpg", "jpeg", "webp"):
        return {"mode": "prose", "text": parse_image(file_bytes)}
    elif ext in ("xlsx", "xls", "csv"):
        return {"mode": "rows", "rows": parse_excel_or_csv(file_bytes, filename)}
    else:
        raise ValueError(f"Unsupported file type: .{ext}")