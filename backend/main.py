import os
import json
import time
from typing import List, Tuple

import numpy as np
import faiss
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding

# =====================================================
# CONFIG
# =====================================================
INDEX_PATH = "faiss.index"
CHUNKS_PATH = "chunks.json"

TOP_K = 5
GREETINGS = {"hi", "hello", "hlo", "hey", "hai"}

SYSTEM_RULES = (
    "You are a professional assistant answering questions about a person's resume. "
    "Use ONLY the resume context provided. "
    "Be concise, accurate, and recruiter-friendly. "
    "If the resume does not include the answer, reply exactly: "
    "\"I don’t see that information in the resume.\""
)

# Hugging Face models (free tier)
HF_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/flan-t5-large",
]

# Groq model (reliable fallback)
GROQ_MODEL = "llama-3.1-8b-instant"

# =====================================================
# DATA MODELS
# =====================================================
class ChatRequest(BaseModel):
    question: str

# =====================================================
# HELPERS
# =====================================================
def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def load_artifacts() -> Tuple[faiss.Index, List[str], str]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise RuntimeError("Missing FAISS index or chunks.json")

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return index, data["chunks"], data["embed_model"]


def build_context(chunks: List[str], indices: List[int]) -> str:
    return "\n\n---\n\n".join(
        chunks[i] for i in indices if 0 <= i < len(chunks)
    )

# =====================================================
# LLM CALLS
# =====================================================
def hf_generate(prompt: str) -> str:
    hf_token = os.getenv("HF_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 160,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }

    for model in HF_MODELS:
        url = f"https://api-inference.huggingface.co/models/{model}"

        for _ in range(2):  # retry per model
            r = requests.post(url, headers=headers, json=payload, timeout=60)

            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "").strip()
                if isinstance(data, dict):
                    return data.get("generated_text", "").strip()

            if r.status_code in (503, 504):
                time.sleep(2)
                continue

            break  # non-retryable error

    raise RuntimeError("Hugging Face unavailable")


def groq_generate(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"].strip()

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Hybrid Resume Chatbot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX, CHUNKS, EMBED_MODEL = load_artifacts()
EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)

# =====================================================
# ROUTES
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        return {"answer": "Please type a question."}

    if question.lower() in GREETINGS:
        return {
            "answer": "Hi 👋 Ask me anything about the resume — skills, experience, or projects."
        }

    # Embed + search
    q_vec = np.array(list(EMBEDDER.embed([question])), dtype=np.float32)
    q_vec = l2_normalize(q_vec)
    _, idx = INDEX.search(q_vec, TOP_K)
    context = build_context(CHUNKS, idx[0].tolist())

    prompt = f"""
You are answering questions about a resume.

INSTRUCTIONS:
- Write the answer in your own words.
- Do NOT copy sentences verbatim from the resume.
- Be professional, concise, and recruiter-friendly.
- Use complete sentences.
- Limit the answer to 3–5 sentences.
- Use only the information from the resume context.

RESUME CONTEXT:
{context}

QUESTION:
{question}

ANSWER (professional summary):
"""


    # 1️⃣ Try Hugging Face
    try:
        answer = hf_generate(prompt)
        if answer:
            return {"answer": answer}
    except Exception:
        pass

    # 2️⃣ Fallback to Groq
    try:
        answer = groq_generate(prompt)
        return {"answer": answer}
    except Exception:
        return {
            "answer": "The chatbot is temporarily unavailable. Please try again later."
        }
