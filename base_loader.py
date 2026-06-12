import json
import re
from sentence_transformers import SentenceTransformer

# ===== LOAD FILES =====
with open("data/admissions.txt", encoding="utf-8") as f:
    RAW_TEXT = f.read()

with open("data/fees.json") as f:
    FEES = json.load(f)

with open("data/cutoffs.json") as f:
    CUTOFFS = json.load(f)

with open("data/seats.json") as f:
    SEATS = json.load(f)

with open("data/hostel.json") as f:
    HOSTEL = json.load(f)

with open("data/programs.json") as f:
    PROGRAMS = json.load(f)

# ===== EMBEDDINGS =====
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_by_section_headers(text):
    sections = re.split(r'\n=+\n', text)
    return [s.strip() for s in sections if len(s.strip()) > 100]

CHUNKS = chunk_by_section_headers(RAW_TEXT)
CHUNK_EMBEDS = embedder.encode(CHUNKS)

def normalize_question(q):
    return q.lower().replace("vlits", "").replace("vignan", "").strip()

def detect_branch(text):
    t = text.lower()
    words = re.findall(r'\b\w+\b', t)

    if any(w in words for w in ["cse","computer"]): return "CSE"
    if any(w in words for w in ["ai","data","ds"]): return "AI & DS"
    if any(w in words for w in ["ece","electronics"]): return "ECE"
    if any(w in words for w in ["eee","electrical"]): return "EEE"
    if any(w in words for w in ["mech","mechanical","me"]): return "ME"
    if any(w in words for w in ["civil","ce"]): return "CE"
    if "it" in words and "bit" not in words: return "IT"
    return None