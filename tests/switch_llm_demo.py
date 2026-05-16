"""
LLM Switch Demo — shows runtime provider switching via API.
Run: python tests/switch_llm_demo.py
"""
import requests, json

BASE = "http://localhost:8000/api/v1"
QUERY = "Who has the strongest Python and data engineering background?"

providers = [
    {"provider": "groq",       "model": "llama-3.1-8b-instant"},
    {"provider": "openrouter", "model": "google/gemma-7b-it"},
    {"provider": "ollama",     "model": "mistral"},
]

print("=" * 60)
print("  JobLens — Live LLM Switch Demo")
print("=" * 60)

for p in providers:
    # 1. Switch provider
    sw = requests.post(f"{BASE}/switch-llm", json=p).json()
    print(f"\n🔄 Switched → {sw['provider']}/{sw['model']}")

    # 2. Ask same question
    res = requests.post(f"{BASE}/query", json={"query": QUERY, "top_k": 3}).json()
    print(f"💬 [{sw['provider']}] {res['answer'][:300]}...")

print("\n✅ Demo complete.")
