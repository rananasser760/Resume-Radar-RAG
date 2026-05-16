"""
Phase 4 — Evaluation & Error Analysis
Run this script AFTER ingesting resumes to test edge cases.

Usage:
    python tests/evaluation.py

Requires API to be running on localhost:8000
"""
import json
import time
import requests

BASE = "http://localhost:8000/api/v1"

# ── Test Cases ────────────────────────────────────────
# Mix of English, Arabic, ambiguous, and adversarial queries

TEST_CASES = [
    # ── English — normal ──────────────────────────────
    {
        "id": "EN-01",
        "desc": "Standard skill match",
        "query": "Who has experience with Python and machine learning?",
        "expect": "Should retrieve ML/Python candidates",
        "lang": "english",
    },
    {
        "id": "EN-02",
        "desc": "Leadership / soft skill query",
        "query": "Find candidates who have led a team or managed a project",
        "expect": "Should retrieve managerial profiles",
        "lang": "english",
    },
    {
        "id": "EN-03",
        "desc": "Education filter",
        "query": "Which candidates have a Master's degree or PhD in Computer Science?",
        "expect": "Should retrieve postgraduate profiles only",
        "lang": "english",
    },
    # ── Arabic — Bonus Phase ───────────────────────────
    {
        "id": "AR-01",
        "desc": "Arabic basic query",
        "query": "من لديه خبرة في تطوير تطبيقات الويب؟",
        "expect": "Should retrieve web developer profiles (cross-lingual)",
        "lang": "arabic",
    },
    {
        "id": "AR-02",
        "desc": "Arabic — data science",
        "query": "ابحث عن المرشحين الذين يعرفون تعلم الآلة والذكاء الاصطناعي",
        "expect": "Should match AI/ML candidates from Arabic query",
        "lang": "arabic",
    },
    # ── Edge Cases / Expected Failures ────────────────
    {
        "id": "EDGE-01",
        "desc": "Domain mismatch — no relevant candidates",
        "query": "Who has experience in marine biology and coral reef research?",
        "expect": "LLM should say 'not found' — not hallucinate names",
        "lang": "english",
        "edge_case": True,
        "expected_failure": "Hallucination risk: LLM might invent names",
    },
    {
        "id": "EDGE-02",
        "desc": "Multi-page fragmentation — degree on page 2",
        "query": "List all candidates who graduated from Cairo University",
        "expect": "May miss candidates if degree chunk is fragmented from name chunk",
        "lang": "english",
        "edge_case": True,
        "expected_failure": "Chunking boundary splits name (p1) from university (p2)",
    },
    {
        "id": "EDGE-03",
        "desc": "Synonym / terminology gap",
        "query": "Find candidates with cloud infrastructure experience",
        "expect": "May miss AWS/GCP/Azure candidates due to embedding gap",
        "lang": "english",
        "edge_case": True,
        "expected_failure": "Embedding may not map 'cloud infrastructure' → 'AWS Lambda'",
    },
]


def run_query(query: str, top_k: int = 5) -> dict:
    r = requests.post(
        f"{BASE}/query",
        json={"query": query, "top_k": top_k},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def run_evaluation():
    print("=" * 60)
    print("  JobLens — Phase 4 Evaluation Suite")
    print("=" * 60)

    # Check API is up
    try:
        requests.get(f"{BASE.replace('/api/v1','')}/health", timeout=5).raise_for_status()
    except Exception:
        print("❌ API not reachable at localhost:8000. Start with docker-compose up.")
        return

    results = []
    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['desc']}")
        print(f"  Query    : {tc['query']}")
        print(f"  Expected : {tc['expect']}")
        if tc.get("edge_case"):
            print(f"  ⚠️  Edge  : {tc.get('expected_failure', '')}")

        t0 = time.time()
        try:
            res = run_query(tc["query"])
            elapsed = round(time.time() - t0, 2)

            top_sources = [c["source"] for c in res["retrieved_chunks"][:3]]
            top_scores = [c["score"] for c in res["retrieved_chunks"][:3]]

            print(f"  ✅ Answer  ({elapsed}s): {res['answer'][:200]}...")
            print(f"  Sources  : {top_sources}")
            print(f"  Scores   : {top_scores}")
            print(f"  LLM      : {res['llm_provider']}/{res['llm_model']}")

            results.append({
                **tc,
                "status": "ok",
                "elapsed_s": elapsed,
                "answer_preview": res["answer"][:300],
                "top_sources": top_sources,
                "top_scores": top_scores,
                "chunks_used": res["chunks_used"],
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({**tc, "status": "error", "error": str(e)})

    # Save report
    report_path = "tests/eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  Results: {ok}/{len(results)} passed")
    print(f"  Report saved → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
