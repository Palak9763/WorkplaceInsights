"""
Test runner for tests/test_retrieval.py.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from tests.test_retrieval import (
    test_hub_fanout_capping,
    test_rrf_scoring_and_matched_anchor,
    test_mmr_dedup,
    test_1hop_vs_2hop_expansion,
    test_path_finding_between_two_anchors,
    test_loud_failure_qdrant_unreachable,
    test_loud_failure_ollama_unreachable,
)

if __name__ == "__main__":
    tests = [
        ("test_hub_fanout_capping", test_hub_fanout_capping),
        ("test_rrf_scoring_and_matched_anchor", test_rrf_scoring_and_matched_anchor),
        ("test_mmr_dedup", test_mmr_dedup),
        ("test_1hop_vs_2hop_expansion", test_1hop_vs_2hop_expansion),
        ("test_path_finding_between_two_anchors", test_path_finding_between_two_anchors),
        ("test_loud_failure_qdrant_unreachable", test_loud_failure_qdrant_unreachable),
        ("test_loud_failure_ollama_unreachable", test_loud_failure_ollama_unreachable),
    ]

    passed = 0
    failed = 0
    print("==================================================")
    print("RUNNING RETRIEVAL TESTS")
    print("==================================================")
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("==================================================")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("==================================================")
    if failed > 0:
        sys.exit(1)
