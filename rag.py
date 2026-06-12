import json
import re
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===== LOAD DATA =====
with open("data/admissions.txt", encoding="utf-8") as f:
    RAW_TEXT = f.read()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_by_section_headers(text):
    sections = re.split(r'\n=+\n', text)
    return [s.strip() for s in sections if len(s.strip()) > 100]

CHUNKS = chunk_by_section_headers(RAW_TEXT)
CHUNK_EMBEDS = embedder.encode(CHUNKS)

def normalize_question(q):
    return q.lower().replace("vlits","").replace("vignan","").strip()

def generate(question):

    q_emb = embedder.encode([normalize_question(question)])
    scores = cosine_similarity(q_emb, CHUNK_EMBEDS)[0]

    if max(scores) < 0.35:
        return {
            "response": "The requested information is not available.",
            "context": None
        }

    top_indices = sorted(range(len(CHUNKS)),
                        key=lambda i: scores[i],
                        reverse=True)[:3]

    context = "\n\n".join([CHUNKS[i] for i in top_indices])

    prompt = f"""
Answer ONLY using this context.
If not present say:
"The requested information is not available."

Context:
{context}

Question:
{question}

Answer:
"""

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 80
            }        
        }
    )
    
    answer = resp.json().get("response", "").strip()

    return {
        "response": answer,
        "context": context
    }