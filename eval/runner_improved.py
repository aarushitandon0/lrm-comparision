"""
eval/runner_improved.py - Full evaluation across all reasoning layers.
Supports checkpoint/resume when Groq rate limits hit.
"""

import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from base_llm import BaseLLM, EVAL_MODEL, SMART_MODEL, GroqQuotaExceeded
from eval.evaluator import CorrectnesEvaluator
from eval.questions import QUESTIONS, filter_questions
from reasoning.linear import LinearReasoning
from reasoning.self_consistent import SelfConsistentReasoning
from reasoning.tree import TreeReasoning
from reasoning.graph import GraphReasoning
from reasoning.mcts import MCTSReasoning
from reasoning.hybrid import HybridTreeConsistentReasoning

DELAY_BETWEEN_RUNS = 2


def build_layers(
    llm: BaseLLM,
    include_mcts: bool = True,
    include_hybrid: bool = True,
    mcts_iterations: int = 4,
) -> dict:
    mcts_llm_model = EVAL_MODEL  # cheap model for MCTS expand/rollout
    layers = {
        "Linear": LinearReasoning(llm),
        "Self-Consistent": SelfConsistentReasoning(llm),
        "Tree": TreeReasoning(llm),
        "Graph": GraphReasoning(llm),
    }
    if include_mcts:
        layers["MCTS"] = MCTSReasoning(
            llm,
            num_iterations=mcts_iterations,
            rollout_model=mcts_llm_model,
            expand_model=mcts_llm_model,
        )
    if include_hybrid:
        layers["Hybrid"] = HybridTreeConsistentReasoning(llm, n_chains=4)
    return layers


def _default_output(model: str) -> str:
    safe = model.replace(".", "_").replace("-", "_")
    return f"eval/results_improved_{safe}.json"


def _load_checkpoint(path: str, layer_names: list) -> dict:
    if not os.path.isfile(path):
        return {name: {} for name in layer_names}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for name in layer_names:
        data.setdefault(name, {})
    return data


def _save_checkpoint(path: str, results: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _is_done(results: dict, layer: str, q_id: str) -> bool:
    r = results.get(layer, {}).get(q_id)
    if not r:
        return False
    if r.get("answer") == "ERROR" and "rate limit" in r.get("reason", "").lower():
        return False
    if r.get("answer") == "ERROR" and "quota" in r.get("reason", "").lower():
        return False
    return True


def run_eval(
    model: str,
    use_judge: bool = False,
    questions=None,
    include_mcts: bool = True,
    include_hybrid: bool = True,
    mcts_iterations: int = 4,
    output_file: str | None = None,
    resume: bool = False,
):
    questions = questions or QUESTIONS
    output_file = output_file or _default_output(model)

    print(f"\n{'='*80}")
    print(f"MODEL: {model}")
    print(f"JUDGE: {'LLM judge' if use_judge else 'heuristics'}")
    print(f"QUESTIONS: {len(questions)}")
    print(f"OUTPUT: {output_file}")
    if resume:
        print("RESUME: skipping completed (layer, question) pairs")
    print(f"{'='*80}\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set in .env")
        return None

    llm = BaseLLM(api_key=api_key, model=model)
    evaluator = CorrectnesEvaluator(
        llm=llm if use_judge else None,
        api_key=api_key if use_judge else None,
    )
    layers = build_layers(
        llm,
        include_mcts=include_mcts,
        include_hybrid=include_hybrid,
        mcts_iterations=mcts_iterations,
    )
    results = _load_checkpoint(output_file, list(layers.keys())) if resume else {
        name: {} for name in layers
    }
    total_qs = len(questions)

    layer_list = ", ".join(layers.keys())
    print(f"LAYERS ({len(layers)}): {layer_list}")
    if include_mcts:
        print(f"MCTS: {mcts_iterations} iters, expand/rollout on {EVAL_MODEL} (saves TPD)")
    print(
        f"Starting: {total_qs} questions x {len(layers)} layers "
        f"= {total_qs * len(layers)} runs\n"
    )
    print(
        "TIP: Groq free tier is 100k tokens/day. If you hit the limit, wait ~10 min "
        "then: python eval/runner_improved.py --limit 10 --resume --no-mcts\n"
    )

    quota_hit = False

    for qi, q in enumerate(questions):
        if quota_hit:
            break

        print(f"\n{'-'*80}")
        print(
            f"[{qi+1:2d}/{total_qs}] {q['id']} "
            f"({q['category']}/{q['difficulty']}) {q['question'][:55]}..."
        )
        print(f"{'-'*80}")

        for name, layer in layers.items():
            if _is_done(results, name, q["id"]):
                r = results[name][q["id"]]
                mark = "OK" if r.get("correct") else "WRONG"
                print(
                    f"  {name:18s} [skip] {mark} (cached) "
                    f"{str(r.get('answer', ''))[:35]}"
                )
                continue

            print(f"  {name:18s}", end=" ", flush=True)
            t0 = time.time()
            try:
                result = layer.solve(q["question"])
                elapsed = round(time.time() - t0, 1)
                predicted = result["answer"]

                eval_result = evaluator.evaluate(
                    predicted=predicted,
                    expected=q["answer"],
                    question=q["question"],
                    category=q["category"],
                )

                results[name][q["id"]] = {
                    "answer": predicted[:200],
                    "correct": eval_result["correct"],
                    "score": round(eval_result["score"], 3),
                    "confidence": eval_result["confidence"],
                    "strategy": eval_result["strategy_used"],
                    "reason": eval_result["reason"],
                    "tokens": result["stats"]["total_tokens"],
                    "calls": result["stats"]["llm_calls"],
                    "time_s": elapsed,
                    "category": q["category"],
                    "difficulty": q["difficulty"],
                }

                mark = "OK" if eval_result["correct"] else "WRONG"
                print(
                    f"{mark} {eval_result['score']*100:.0f}% | "
                    f"{predicted[:35]:35s} | "
                    f"{result['stats']['total_tokens']:5d} tok | {elapsed:.1f}s"
                )
            except GroqQuotaExceeded as e:
                elapsed = round(time.time() - t0, 1)
                results[name][q["id"]] = {
                    "answer": "ERROR",
                    "correct": False,
                    "score": 0.0,
                    "confidence": "low",
                    "strategy": "quota",
                    "reason": str(e)[:200],
                    "tokens": 0,
                    "calls": 0,
                    "time_s": elapsed,
                    "category": q.get("category"),
                    "difficulty": q.get("difficulty"),
                }
                _save_checkpoint(output_file, results)
                print(f"QUOTA {str(e)[:50]}")
                print(f"\nStopped. Progress saved to {output_file}")
                print(
                    "When quota resets, re-run the SAME command with --resume "
                    "(add --no-mcts to save tokens)."
                )
                quota_hit = True
                break
            except Exception as e:
                elapsed = round(time.time() - t0, 1)
                results[name][q["id"]] = {
                    "answer": "ERROR",
                    "correct": False,
                    "score": 0.0,
                    "confidence": "low",
                    "strategy": "error",
                    "reason": str(e)[:120],
                    "tokens": 0,
                    "calls": 0,
                    "time_s": elapsed,
                    "category": q.get("category"),
                    "difficulty": q.get("difficulty"),
                }
                print(f"ERR {str(e)[:60]}")

            _save_checkpoint(output_file, results)
            time.sleep(DELAY_BETWEEN_RUNS)

    print(f"\n\n{'='*100}")
    print("SUMMARY (completed runs only)")
    print(f"{'='*100}\n")

    print(f"{'ID':<6} {'Cat':<12} ", end="")
    for name in layers:
        print(f"{name:<14}", end="")
    print()
    print("-" * 100)

    correct_by_layer = {name: 0 for name in layers}
    counted_by_layer = {name: 0 for name in layers}

    for q in questions:
        print(f"{q['id']:<6} {q['category']:<12} ", end="")
        for name in layers:
            r = results[name].get(q["id"])
            if not r:
                print(f"{'---':<14}", end="")
                continue
            counted_by_layer[name] += 1
            if r["correct"]:
                correct_by_layer[name] += 1
            cell = "Y" if r["correct"] else "N"
            print(f"{cell} {r.get('tokens', 0):4d}t     ", end="")
        print()

    print("\nACCURACY BY LAYER:")
    for name in layers:
        c = correct_by_layer[name]
        n = counted_by_layer[name]
        if n:
            print(f"  {name:<18} {c}/{n} ({100*c/n:.1f}%)")
        else:
            print(f"  {name:<18} no data")

    print(f"\nSaved: {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-layer evaluation")
    parser.add_argument("--model", default=SMART_MODEL)
    parser.add_argument("--fast", action="store_true", help="Use fast 8B for all layers")
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge scoring")
    parser.add_argument("--limit", type=int, default=None, help="Max questions")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--no-mcts", action="store_true", help="Skip MCTS (saves ~80%% TPD)")
    parser.add_argument("--no-hybrid", action="store_true", help="Skip Hybrid layer")
    parser.add_argument("--mcts-iters", type=int, default=4, help="MCTS iterations (default 4)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from saved JSON; skip finished pairs",
    )
    args = parser.parse_args()

    model = EVAL_MODEL if args.fast else args.model
    qs = filter_questions(
        category=args.category, difficulty=args.difficulty, limit=args.limit
    )

    run_eval(
        model=model,
        use_judge=args.judge,
        questions=qs,
        include_mcts=not args.no_mcts,
        include_hybrid=not args.no_hybrid,
        mcts_iterations=args.mcts_iters,
        output_file=args.output,
        resume=args.resume,
    )
