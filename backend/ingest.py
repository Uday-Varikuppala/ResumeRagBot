import os
import json
import re
from typing import List

import numpy as np
import faiss
from pypdf import PdfReader
from fastembed import TextEmbedding


PDF_DIR = "pdfs"
OUT_INDEX = "faiss.index"
OUT_CHUNKS = "chunks.json"

# Chunking tuned for 1-page resume (fast + accurate)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Lightweight embedding model (fastembed downloads it on first run)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)
    return "\n".join(pages)


def clean_text(text: str) -> str:
    # Keep it readable; reduce weird spacing common in PDFs
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def main():
    if not os.path.isdir(PDF_DIR):
        raise RuntimeError(f"Missing folder: {PDF_DIR}. Create it and put Resume.pdf inside.")

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise RuntimeError(f"No PDFs found in {PDF_DIR}. Add Resume.pdf (or any .pdf).")

    full_texts = []
    for pdf in pdf_files:
        path = os.path.join(PDF_DIR, pdf)
        print(f"Reading: {path}")
        raw = extract_text_from_pdf(path)
        cleaned = clean_text(raw)
        if cleaned:
            full_texts.append(cleaned)

    combined = "\n\n".join(full_texts)
    chunks = chunk_text(combined, CHUNK_SIZE, CHUNK_OVERLAP)

    if len(chunks) < 1:
        raise RuntimeError("No text chunks created. Your PDF may be image-only or empty.")

    print(f"Chunks: {len(chunks)}")

    # Embed
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    vectors = np.array(list(embedder.embed(chunks)), dtype=np.float32)
    vectors = l2_normalize(vectors)

    # Build FAISS (cosine similarity via inner product on normalized vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save
    faiss.write_index(index, OUT_INDEX)
    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embed_model": EMBED_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunks": chunks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {OUT_INDEX}, {OUT_CHUNKS}")
    print("Done.")


if __name__ == "__main__":
    main()
