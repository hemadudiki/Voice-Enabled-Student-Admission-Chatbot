import json
import re
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================== LOAD DATA ==================

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

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================== CHUNKING ==================

def chunk_by_section_headers(text):
    sections = re.split(r'\n=+\n', text)
    return [s.strip() for s in sections if len(s.strip()) > 100]

CHUNKS = chunk_by_section_headers(RAW_TEXT)
CHUNK_EMBEDS = embedder.encode(CHUNKS)

# ================== HELPERS ==================

def normalize_question(q):
    q = q.lower()
    q = q.replace("vlits", "")
    q = q.replace("vignan", "")
    return q.strip()

def detect_branch(text):
    t = text.lower()
    words = re.findall(r'\b\w+\b', t)

    if any(word in words for word in ["cse", "computer"]): return "CSE"
    if any(word in words for word in ["ai", "data", "ds"]): return "AI & DS"
    if any(word in words for word in ["ece", "electronics"]): return "ECE"
    if any(word in words for word in ["eee", "electrical"]): return "EEE"
    if any(word in words for word in ["mech", "mechanical", "me"]): return "ME"
    if any(word in words for word in ["civil", "ce"]): return "CE"
    if "it" in words and "bit" not in words: return "IT"
    return None

def extract_rank(text):
    t = text.lower().replace(",", "")
    m = re.search(r"(\d+)\s*k", t)
    if m: return int(m.group(1)) * 1000
    m = re.search(r"\b(\d{3,6})\b", t)
    return int(m.group(1)) if m else None

def calculate_tuition_fee(branch, category):
    return FEES[category][branch]

def calculate_total_fee(branch, category, hostel=None):
    tuition_fee = FEES[category][branch]
    hostel_fee = 0
    laundry_fee = 0

    if hostel:
        hostel_fee = HOSTEL[hostel]["hostel_fee"]
        laundry_fee = HOSTEL[hostel]["laundry_fee"]

    total = tuition_fee + hostel_fee + laundry_fee
    return (
        f"Total annual fee for {branch} ({category}) is ₹{total:,}\n"
        f"- Tuition Fee : ₹{tuition_fee:,}\n"
        f"- Hostel Fee  : ₹{hostel_fee:,}\n"
        f"- Laundry Fee : ₹{laundry_fee:,}"
    )

# ================== HYBRID GENERATOR ==================
def generate(question):

    q = question.lower()

    # ---- SEATS ----
    if "seats" in q or "seat" in q or "intake" in q:
        branch = detect_branch(q)
        seats = SEATS.get(branch)
        if seats:
            return {
                "response": f"{seats} seats are currently available in {branch}.",
                "context": None
            }

    # ---- CUTOFF ----
    if "cutoff" in q and "rank" in q:
        branch = detect_branch(q)
        cutoff = CUTOFFS.get(branch)
        if cutoff:
            return {
                "response": f"Cutoff Rank for {branch} is {cutoff}.",
                "context": None
            }

    # ---- ELIGIBILITY ----
    if "eligible" in q or ("rank" in q and "placements" not in q and "eamcet" in q):
        branch = detect_branch(q)
        rank = extract_rank(q)

        if not branch or not rank:
            return {
                "response": "Please mention both your rank and branch.",
                "context": None
            }

        if "category b" in q:
            return {
                "response": f"Yes. You are eligible for {branch} under Category-B (subject to seats).",
                "context": None
            }

        cutoff = CUTOFFS.get(branch)
        if cutoff is None:
            return {
                "response": "Cutoff information is not available for the selected branch.",
                "context": None
            }

        if rank <= cutoff:
            return {
                "response": f"Yes. Your rank {rank} is eligible for {branch} under Category-A. Cutoff Rank is {cutoff}.",
                "context": None
            }
        else:
            return {
                "response": f"No. Your rank {rank} is not eligible for {branch} under Category-A. Cutoff Rank is {cutoff}. You are eligible under Category B.",
                "context": None
            }

    # ---- HOSTEL COST ----
    if "hostel fee" in q or ("cost" in q and "hostel" in q):
        ac_total = HOSTEL["AC"]["hostel_fee"] + HOSTEL["AC"]["laundry_fee"]
        non_ac_total = HOSTEL["Non-AC"]["hostel_fee"] + HOSTEL["Non-AC"]["laundry_fee"]

        return {
            "response": f"AC Hostel: ₹{ac_total}\nNon-AC Hostel: ₹{non_ac_total}\nLaundry Fees Included.",
            "context": None
        }

    # ---- TUITION ----
    if ("tuition" in q or ("fee" in q and not any(x in q for x in ["total","hostel","pay","transport","bus","travel"]))):
        branch = detect_branch(q)
        if not branch:
            return {
                "response": "Please specify the branch.",
                "context": None
            }

        category = "Category-B" if any(x in q for x in ["category b","category-b","management"]) else "Category-A"
        tuition = calculate_tuition_fee(branch, category)

        return {
            "response": f"Tuition fee for {branch} under {category} is ₹{tuition:,} per year.",
            "context": None
        }

    # ---- TOTAL FEE ----
    fee_keywords = ["total fee","total annual fee","how much","pay","cost"]
    if any(kw in q for kw in fee_keywords) and "fee" in q:
        branch = detect_branch(q)
        if not branch:
            return {
                "response": "Please specify the branch.",
                "context": None
            }

        category = "Category-B" if any(x in q for x in ["category b","category-b","management"]) else "Category-A"

        hostel = None
        if "non" in q or "non-ac" in q:
            hostel = "Non-AC"
        elif "ac" in q:
            hostel = "AC"

        return {
            "response": calculate_total_fee(branch, category, hostel),
            "context": None
        }

    # ---- PROGRAMS ----
    if "undergraduate" in q and ("branches" in q or "courses" in q):
        return {
            "response": "B.Tech programs: " + ", ".join(PROGRAMS["BTech"]),
            "context": None
        }

    if "postgraduate" in q or "pg" in q:
        return {
            "response": "PG programs: " + ", ".join(PROGRAMS["PG"]),
            "context": None
        }

    # ---- LOCATION ----
    if "location" in q or "address" in q:
        return {
            "response": "Vadlamudi Village, Chebrolu Mandal, Guntur District, Andhra Pradesh.",
            "context": None
        }

    if "documents" in q:
        return {
            "response": "Required documents:\n- 10th class marks memo\n- 12th marks memo\n- Transfer Certificate\n- Aadhaar card\n- Entrance rank card",
            "context": None
        }

    # ================== RAG FALLBACK ==================

    q_emb = embedder.encode([normalize_question(question)])
    scores = cosine_similarity(q_emb, CHUNK_EMBEDS)[0]

    if max(scores) < 0.35:
        return {
            "response": "The requested information is not available.",
            "context": None
        }

    top_indices = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)[:3]
    context = "\n\n".join([CHUNKS[i] for i in top_indices])

    prompt = f"""
You are an admission information assistant.

STRICT RULES:
- Answer ONLY using the information explicitly stated in the context.
- Do NOT infer, assume, or generalize.
- Do NOT use external knowledge.
- Do NOT mention the context, documents, or sources.
- Do NOT explain where the information comes from.
- If the question asks about a specific branch, answer only for that branch.
- If the question asks for a list of branches or programs, list all branches explicitly mentioned in the context.
- If the answer is not explicitly present, respond EXACTLY with:
  "The requested information is not available."
- For eligibility or requirement questions, respond with "Yes" or "No" first, followed by one factual sentence.
- If the question is a "Yes" or "No" question and the context implies availability or absence,
answer with "Yes" or "No" followed by one supporting phrase from the context.

Stop immediately after the answer.


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