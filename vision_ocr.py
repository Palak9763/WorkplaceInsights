"""
Reads text out of an image using qwen2.5vl:3b (vision-capable model).
Used for images and scanned PDF pages - anything without a clean text layer.
"""
import base64
import ollama
from config import OLLAMA_HOST

VISION_MODEL = "qwen2.5vl:3b"


def read_image_text(image_bytes: bytes) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Transcribe ALL visible text in this image exactly as written. "
                    "Return ONLY the transcribed text, no commentary, no description "
                    "of the image itself - just the text content."
                ),
                "images": [b64_image],
            }
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"].strip()