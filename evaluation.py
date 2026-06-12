import json
import requests
from rouge_score import rouge_scorer

# Load dataset
with open("evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

total_r1 = total_r2 = total_rl = 0

for i, item in enumerate(dataset):
    question = item["question"]
    reference = item["reference_answer"]

    # Call your local LLM API (same as chatbot)
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": question,
            "stream": False
        }
    )

    generated = resp.json()["response"].strip()

    scores = scorer.score(reference, generated)

    total_r1 += scores["rouge1"].fmeasure
    total_r2 += scores["rouge2"].fmeasure
    total_rl += scores["rougeL"].fmeasure

    print(f"\nQuestion {i+1}")
    print("Generated:", generated)
    print("ROUGE-1:", scores["rouge1"].fmeasure)
    print("ROUGE-2:", scores["rouge2"].fmeasure)
    print("ROUGE-L:", scores["rougeL"].fmeasure)

n = len(dataset)

print("\n===== FINAL AVERAGE SCORES =====")
print("ROUGE-1:", total_r1 / n)
print("ROUGE-2:", total_r2 / n)
print("ROUGE-L:", total_rl / n)