import json
from evaluator import evaluate
from systems import vanilla, rag, hybrid

with open("data/evaluation_dataset.json") as f:
    dataset = json.load(f)

systems = {
    "Vanilla": vanilla.generate,
    "RAG": rag.generate,
    "Hybrid": hybrid.generate
}

for name, func in systems.items():
    print(f"\nEvaluating {name}...")
    results = evaluate(func, dataset)
    for k, v in results.items():
        print(f"{k}: {v:.2f}")