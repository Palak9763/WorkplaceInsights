"""
Evaluation harness for the /query endpoint.

Usage:
  python eval.py [--questions eval_questions.json] [--url http://127.0.0.1:8000]

For each question, checks:
  - graph/hybrid/vector: each expected_fact appears (case-insensitive) in
    the answer OR in the graph_results sentences OR in vector_results text.
  - unanswerable: answer contains the "not enough information" phrase.

Prints a pass/fail table and a total score.
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error

NOT_ENOUGH_PHRASE = "not enough information"


def call_query(url: str, question: str) -> dict:
    req = urllib.request.Request(
        f"{url}/query",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def fact_in_response(fact: str, response: dict) -> bool:
    fact_lower = fact.lower()
    # Check answer text
    if fact_lower in response.get("answer", "").lower():
        return True
    # Check graph result sentences
    for row in response.get("graph_results", []):
        sentence = row.get("_sentence") or ""
        if fact_lower in sentence.lower():
            return True
        # Also check raw field values
        for v in row.values():
            if isinstance(v, str) and fact_lower in v.lower():
                return True
    # Check vector result text
    for chunk in response.get("vector_results", []):
        if fact_lower in chunk.get("text", "").lower():
            return True
    return False


def run_eval(questions_path: str, url: str) -> dict:
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    passed = 0
    total = len(questions)

    print(f"\nRunning evaluation: {total} questions against {url}\n")
    print(f"{'ID':<8} {'Cat':<12} {'Pass':<6} {'Details'}")
    print("-" * 80)

    for q in questions:
        qid = q.get("id", "?")
        category = q.get("category", "?")
        question = q["question"]
        expected_facts = q.get("expected_facts", [])
        unanswerable = q.get("unanswerable", False)

        try:
            t0 = time.perf_counter()
            response = call_query(url, question)
            elapsed = round((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            results.append({
                "id": qid, "category": category, "pass": False,
                "error": str(exc), "elapsed_ms": -1,
            })
            print(f"{qid:<8} {category:<12} {'FAIL':<6} ERROR: {exc}")
            continue

        answer = response.get("answer", "")

        if unanswerable:
            ok = NOT_ENOUGH_PHRASE in answer.lower()
            detail = "got 'not enough info'" if ok else f"unexpected answer: {answer[:60]}"
        elif expected_facts:
            missing = [f for f in expected_facts if not fact_in_response(f, response)]
            ok = len(missing) == 0
            detail = "all facts found" if ok else f"missing: {missing[:2]}"
        else:
            ok = True
            detail = "no expected facts (manual review needed)"

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "pass": ok,
            "detail": detail,
            "elapsed_ms": elapsed,
            "answer_snippet": answer[:100],
            "grounding_warning": response.get("grounding_warning", []),
            "diagnostics": response.get("diagnostics", {}),
        })
        print(f"{qid:<8} {category:<12} {status:<6} {detail}")

    score = round(passed / total * 100, 1) if total > 0 else 0
    print("-" * 80)
    print(f"\nScore: {passed}/{total}  ({score}%)\n")

    return {"score_pct": score, "passed": passed, "total": total, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    summary = run_eval(args.questions, args.url)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
