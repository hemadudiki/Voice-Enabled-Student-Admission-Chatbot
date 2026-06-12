import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# Initialize models once
embedder = SentenceTransformer("all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def factual_score(predicted, ground_truth):

    p = predicted.lower().strip().replace(".", "")
    g = ground_truth.lower().strip().replace(".", "")

    if g in p:
        return 1

    # numeric match
    import re
    nums_p = re.findall(r'\d+\.?\d*', p)
    nums_g = re.findall(r'\d+\.?\d*', g)

    if nums_g and all(n in nums_p for n in nums_g):
        return 1

    # keyword overlap
    if any(word in p for word in g.split()):
        return 0.5

    return 0
import re

def detect_hallucination(predicted, context):

    if context is None:
        return 0

    if predicted.strip() == "The requested information is not available.":
        return 0

    p = predicted.lower()
    c = context.lower()

    # numeric grounding
    nums_p = re.findall(r'\d+\.?\d*', p)
    nums_c = re.findall(r'\d+\.?\d*', c)

    if any(n not in nums_c for n in nums_p):
        return 1

    # keyword containment ratio
    words = [w for w in p.split() if len(w) > 3]
    overlap = sum(1 for w in words if w in c)

    if len(words) == 0:
        return 0

    ratio = overlap / len(words)

    if ratio < 0.5:
        return 1

    return 0
def evaluate(system_func, dataset, system_name="System"):

    total_score = 0
    hallucinations = 0
    total_time = 0
    rouge_total = 0

    print(f"\n--- Evaluating {system_name} ---")

    for item in dataset:

        q = item["question"]
        gt = item["ground_truth"]

        start = time.time()
        output = system_func(q)
        end = time.time()

        total_time += (end - start)

        response = output["response"]
        context = output["context"]

        # Accuracy
        score = factual_score(response, gt)
        total_score += score

        # Hallucination
        hallucination=detect_hallucination(response, context)
        hallucinations += hallucination

        # ROUGE-L
        rouge_score_value = scorer.score(gt, response)['rougeL'].fmeasure
        rouge_total += rouge_score_value

        # if score == 0:
            
        #     print("\n❌ WRONG ANSWER (ACCURACY)")
        #     print("Question:", q)
        #     print("Expected:", gt)
        #     print("Response:", response)
        #     print("-" * 50)
        # if hallucination == 1:
        #     print("\n❌ WRONG ANSWER ( HALLUCINATION)")
        #     print("Question:", q)
        #     print("Expected:", gt)
        #     print("Response:", response)
        #     print("-" * 50)    

    return {
        "Accuracy (%)": (total_score / len(dataset)) * 100,
        "Hallucination (%)": (hallucinations / len(dataset)) * 100,
        "ROUGE-L": rouge_total / len(dataset),
        "Avg Latency (s)": total_time / len(dataset)
    }