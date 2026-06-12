import requests

def generate(question):

    prompt = f"""
You are a university admission assistant.
Answer clearly and directly.

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
            "options": {"num_predict": 100}
        }
    )

    answer = resp.json().get("response", "").strip()

    return {
        "response": answer,
        "context": None
    }