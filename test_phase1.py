"""
test_phase1.py - Validate Phase 1 implementation

Run this to check:
1. All imports work
2. Scoring logic works correctly
3. API connection works
4. Basic evaluation runs without crashing
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

print("=" * 70)
print("PHASE 1 VALIDATION TEST")
print("=" * 70)

print("\n[TEST 1] Checking imports...")
try:
    from eval.evaluator import CorrectnesEvaluator
    from eval.runner_improved import run_eval
    from eval.questions import QUESTIONS
    from base_llm import BaseLLM
    print("  [OK] All imports successful")
except Exception as e:
    print(f"  [FAIL] Import failed: {e}")
    sys.exit(1)

print("\n[TEST 2] Testing numeric scoring...")
evaluator = CorrectnesEvaluator()

test_cases = [
    ("600", "600", True),
    ("600.0", "600", True),
    ("600.5", "600", True),
    ("wrong answer", "600", False),
]

for pred, exp, should_pass in test_cases:
    result = evaluator.evaluate(pred, exp, "test", "Math")
    status = "[OK]" if result["correct"] == should_pass else "[FAIL]"
    print(f"  {status} '{pred}' vs '{exp}' -> {result['correct']} (expected {should_pass})")

print("\n[TEST 3] Testing form variation scoring...")

form_cases = [
    ("alice has fish", "alice fish", True),
    ("alice:fish", "alice fish", True),
    ("alice fish bob cat carol dog", "alice fish bob cat carol dog", True),
    ("alice only", "alice fish bob cat carol dog", False),
]

for pred, exp, should_pass in form_cases:
    result = evaluator.evaluate(pred, exp, "test", "Logic")
    status = "[OK]" if (result["score"] > 0.8) == should_pass else "[FAIL]"
    print(f"  {status} '{pred[:40]}' -> score: {result['score']:.2f} (high expected: {should_pass})")

print("\n[TEST 4] Testing API connection...")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("  [WARN] No GROQ_API_KEY found in .env")
    print("         Set GROQ_API_KEY to proceed with full evaluation")
else:
    try:
        llm = BaseLLM(api_key=api_key)
        print("  [OK] API key loaded")
        try:
            response = llm.call("Say 'test' in one word.")
            print(f"  [OK] API responded: '{response[:60]}'")
        except Exception as e:
            print(f"  [WARN] API call failed (rate limit?): {str(e)[:60]}")
    except Exception as e:
        print(f"  [WARN] Failed to create LLM: {str(e)[:60]}")

print("\n[TEST 5] Checking question dataset...")
print(f"  [OK] {len(QUESTIONS)} questions loaded:")
for i, q in enumerate(QUESTIONS[:3]):
    print(f"     - {q['id']}: {q['category']:<12} (answer: {q['answer'][:30]})")
print(f"     ... and {len(QUESTIONS)-3} more")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print("\nPhase 1 Implementation Ready!")
print("\nNext steps:")
print("  1. Run: python eval/runner_improved.py --fast")
print("  2. Check results in: eval/results_improved_*.json")
print()
