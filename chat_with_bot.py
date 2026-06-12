from voice_input import get_voice_query
import requests
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================== STARTUP ==================
print("\n🎓 Welcome to the University Admission Chatbot!")
print("Type your questions below.")
print("Type 'exit' to end the chat.\n")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

def normalize_question(q):
    q = q.lower()
    q = q.replace("vlits", "")
    q = q.replace("vignan", "")
    return q.strip()


# ================== CHUNKING ==================
def chunk_by_section_headers(text):
    # Split on lines that contain only equal signs (section separators)
    sections = re.split(r'\n=+\n', text)
    cleaned = [s.strip() for s in sections if len(s.strip()) > 100]
    return cleaned



CHUNKS = chunk_by_section_headers(RAW_TEXT)
CHUNK_EMBEDS = embedder.encode(CHUNKS)

# print("\n================= CHUNK STRUCTURE =================\n")

# for i in range(len(CHUNKS)):
#     print(f"\n--- Chunk {i} ---")
#     print(f"Length: {len(CHUNKS[i])} characters")
#     print(CHUNKS[i][:1000])
#     print("\n--------------------------------------------------")


# print(f"\nTotal Chunks Created: {len(CHUNKS)}")
# print("\n===================================================\n")


# ================== HELPERS ==================
def detect_branch(text):
    t = text.lower()
    words = re.findall(r'\b\w+\b', t)  # split into words
    
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
    tuition_fee = FEES[category][branch]
    return tuition_fee

def calculate_total_fee(branch, category, hostel=None):
    tuition_fee = FEES[category][branch]
    hostel_fee = 0
    laundry_fee = 0
    
    if hostel:
        hostel_fee = HOSTEL[hostel]["hostel_fee"]
        laundry_fee = HOSTEL[hostel]["laundry_fee"]
    
    total = tuition_fee + hostel_fee + laundry_fee
    print(f"\nBot: Total annual fee for {branch} ({category}) is ₹{total:,}"
          f"\n- Tuition Fee : ₹{tuition_fee:,}\n"
          f"- Hostel Fee  : ₹{hostel_fee:,}\n"
          f"- Laundry Fee : ₹{laundry_fee:,}\n")
    return total

# ================== STATE ==================
CACHE = {}
PENDING = None   # stores pending intent
# ===== SELECT MODE ONCE =====
mode = input("\nChoose input mode (text / voice): ").strip().lower()

if mode not in ["text", "voice"]:
    print("Invalid mode selected. Defaulting to text mode.")
    mode = "text"

# ===== CHAT LOOP =====
while True:

  # question = input("\nYou: ").strip()
    if mode == "exit":
        print("\nBot: Thank you for contacting VLITS. Have a great day! 👋")
        break
    if mode == "voice":
        
        consent = input("\nPress Enter to speak (or type 'text' to switch mode): ").strip().lower()
    
        if consent == "text":
            mode = "text"
            print("Switched to text mode.")
            continue
    
        question = get_voice_query()
        if question is None:
            print("Bot: I couldn’t understand that. Please try again.")
            continue
        if question in ["exit", "quit", "bye"]:
            print("\nBot: Thank you for contacting VLITS. Have a great day! 👋")
            break

    else:
        question = input("\nYou: ").strip()

    q = question.lower()


    # ---- EXIT / GREETING ----
    if q in ["exit", "quit", "bye"]:
        print("\nBot: Thank you for contacting VLITS. Have a great day! 👋")
        break
    if q== "switch to voice":
        mode = "voice"
        print("Switched to voice mode.")
        continue

    if q == "switch to text":
        mode = "text"
        print("Switched to text mode.")
        continue

    if q in ["hi", "hello", "hey"]:
        print("\n Bot: Hello! 👋 How can I help you with admissions, fees, hostel, or eligibility?")
        continue
    if "seats" in q or "seat" in q or "intake" in q:
        branch = detect_branch(q)
        seats = SEATS.get(branch)
        print(f"Bot: {seats} seats are currently available in {branch}.\n")
        continue

    if "cutoff" in q  and "rank" in q  :
        branch = detect_branch(q)
        cutoff = CUTOFFS.get(branch)
        print(f"Bot: Cutoff Rank for {branch} is {cutoff} .")
        continue

    # ---- ELIGIBILITY ----
    if "eligible" in q  or "rank" in q and "placements" not in q and "EAMCET" in q:
        branch = detect_branch(q)
        rank = extract_rank(q)

        if not branch or not rank:
            print("\nBot: Please mention both your rank and branch.")
            continue

        if "category b" in q:
            print(f"\nBot: Yes. You are eligible for {branch} under Category-B (subject to seats).")
            continue

        cutoff = CUTOFFS.get(branch)
        if cutoff is None: 
            print("\nBot: Cutoff information is not available for the selected branch.") 
            continue
        if rank <= cutoff:
            print(f"\nBot: Yes. Your rank {rank} is eligible for {branch} under Category-A.\n     Cutoff Rank for {branch} is {cutoff}.")
        else:
            print(f"\nBot: No. Your rank {rank} is not eligible for {branch} under Category-A.\n     Cutoff Rank for {branch} is {cutoff}. \n     You are eligible under Category B which doesn't require EAPCET rank eligiblity . ")
        continue
    
    # ---------- HOSTEL COST PRIORITY ----------
    if "hostel fee " in q or ("cost" in q and "hostel" in q ):
        ac_total = HOSTEL["AC"]["hostel_fee"] + HOSTEL["AC"]["laundry_fee"]
        non_ac_total = HOSTEL["Non-AC"]["hostel_fee"] + HOSTEL["Non-AC"]["laundry_fee"]

        print(
            "\nBot: The total hostel cost per year is:\n"
            f"- AC Hostel: ₹{ac_total}\n"
            f"- Non-AC Hostel: ₹{non_ac_total}"
            "\n Laundry Fees Included.\n"
        )
        continue

# ---------- TUITION FEE HANDLER ----------
    if ("tuition" in q or ("fee" in q and not any(x in q for x in ["total", "hostel", "pay","transport","bus","travel"]))) :
        branch = detect_branch(q)
        if not branch:
            print("\nBot: Please specify the branch (CSE, ECE, EEE, AI & DS, IT, ME, CE).")
            continue

        category = "Category-B" if any(x in q.lower() for x in ["category b", "category-b", "management"]) else "Category-A"
        tuition = calculate_tuition_fee(branch, category)

        print(f"\nBot: Tuition fee for {branch} under {category} is ₹{tuition:,} per year.")
        continue

    # ---- TOTAL FEE ----
    fee_keywords = ["total fee", "total annual fee", "how much", "pay", "cost"]
    if any(kw in q.lower() for kw in fee_keywords) and "fee" in q:
        branch = detect_branch(q)
        if not branch:
            PENDING = "total_fee"
            print("\nBot: Please specify the branch.")
            continue
        
        category = "Category-B" if any(x in q.lower() for x in ["category b", "category-b", "management"]) else "Category-A"
        
        # Hostel detection
        hostel = None
        q_lower = q.lower()
        if any(x in q_lower for x in ["non", "non-ac"]):
            hostel = "Non-AC"
        elif "ac" in q_lower:
            hostel = "AC"
        elif "without" in q_lower and "hostel" in q_lower:
            hostel = None
        elif "hostel" in q_lower and "without" not in q_lower:
            hostel = "Non-AC"  # Default
        
        calculate_total_fee(branch, category, hostel)
        continue
    # ---- PROGRAM LIST ----
    if  " undergraduate" in q and( "branches" in q or "courses" in q ):
        print("\nBot: B.Tech programs offered and accredited:")
        for p in PROGRAMS["BTech"]:
            print("-", p)
        continue
    if  " postgraduate" in q or "pg" in q  :
        print("\nBot: PG programs offered and accredited:")
        for p in PROGRAMS["PG"]:
            print("-", p)
        continue

    # ---- LOCATION ----
    if "full address" in q or "located" in q or "location" in q:
        print("\nBot: VLITS is located at Vadlamudi Village, Chebrolu Mandal, Guntur District, Andhra Pradesh.")
        continue
    if "documents" in q :
        print("The following documents are required during the admission process at VLITS:"
                "\n- 10th class marks memo\n"
                "- 12th / Intermediate marks memo\n"
                "- Transfer Certificate (TC)\n"
                "- Aadhaar card\n"
                "- Entrance examination rank card (if applicable)\n")
        continue

    if ("bus" in q or "transport" in q) and "fees" in q :
        print("Annual transport fee varies by route. Fees are informed during admission counseling.")
    

    # ---- CACHE ----
    if question in CACHE:
        print("\nBot:", CACHE[question])
        continue

    # ================== RAG ==================
    q_emb = embedder.encode([normalize_question(question)])
    scores = cosine_similarity(q_emb, CHUNK_EMBEDS)[0]
    
    # print("\nSimilarity Scores:")
    # for i, score in enumerate(scores):
    #     print(f"Chunk {i}: {score:.4f}")

    indices = list(range(len(CHUNKS)))
    # if not any(k in q for k in ["where", "location", "district", "address", "college", "Vijayawada", "private", "affiliated", "affiliation"]):
    #     indices.remove(0)

   # Get top 3 most relevant chunks
    top_k = 3
    top_indices = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_k]

    context = "\n\n".join([CHUNKS[i] for i in top_indices])
    # print(f"Total Chunks Created: {len(CHUNKS)}")
    # print(f"\n[Retrieved Chunk Indices: {top_indices}]")
    # print(context)
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
        json={"model": "phi3", "prompt": prompt, "stream": False, "options": {"num_predict": 80}}
    )
    
    answer = resp.json().get("response", "").strip()
    CACHE[question] = answer
    print("\nBot:", answer)

   