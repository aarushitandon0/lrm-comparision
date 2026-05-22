"""
Publication-style analysis: heatmaps, efficiency frontier, confidence calibration,
statistical significance, optional tree/graph path plots.
"""

import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.questions import QUESTIONS, get_metadata

OUTPUT_DIR = Path(__file__).parent / "figures"


def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def accuracy_matrix(results: dict) -> dict:
    layers = list(results.keys())
    categories = sorted({q["category"] for q in QUESTIONS})
    difficulties = sorted({q["difficulty"] for q in QUESTIONS})
    matrix = {}

    for layer in layers:
        matrix[layer] = {}
        for cat in categories:
            matrix[layer][cat] = {}
            for diff in difficulties:
                ids = [
                    q["id"] for q in QUESTIONS
                    if q["category"] == cat and q["difficulty"] == diff
                ]
                if not ids:
                    matrix[layer][cat][diff] = None
                    continue
                correct = sum(
                    1 for qid in ids
                    if results[layer].get(qid, {}).get("correct")
                )
                matrix[layer][cat][diff] = 100 * correct / len(ids)
    return matrix, categories, difficulties


def plot_heatmap(results: dict, out_path: Path):
    layers = sorted(results.keys())
    categories = sorted({q["category"] for q in QUESTIONS})
    data = np.zeros((len(layers), len(categories)))

    for i, layer in enumerate(layers):
        for j, cat in enumerate(categories):
            ids = [q["id"] for q in QUESTIONS if q["category"] == cat]
            if ids:
                correct = sum(
                    1 for qid in ids
                    if results[layer].get(qid, {}).get("correct")
                )
                data[i, j] = 100 * correct / len(ids)

    fig, ax = plt.subplots(figsize=(10, max(4, len(layers) * 0.8)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    for i in range(len(layers)):
        for j in range(len(categories)):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title("Accuracy Heatmap: Layer x Category (%)")
    plt.colorbar(im, ax=ax, label="Accuracy %")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_efficiency_frontier(results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer, per_q in results.items():
        tokens, correct, total, time_s = 0, 0, 0, 0
        for r in per_q.values():
            tokens += r.get("tokens", 0)
            time_s += r.get("time_s", 0)
            total += 1
            if r.get("correct"):
                correct += 1
        if total == 0:
            continue
        acc = 100 * correct / total
        tpp = tokens / total
        ax.scatter(tpp, acc, s=120, label=layer, alpha=0.85)
        ax.annotate(layer, (tpp, acc), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Average tokens per question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Token Efficiency Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_confidence_calibration(results: dict, out_path: Path):
    layers = list(results.keys())
    n = len(layers)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        confs, accs = [], []
        for qid, r in results[layer].items():
            score = r.get("score", 0.5)
            for lo, hi in bins:
                if lo <= score < hi:
                    confs.append((lo + hi) / 2 * 100)
                    accs.append(100 if r.get("correct") else 0)
                    break
        if confs:
            ax.scatter(confs, accs, alpha=0.5)
        ax.plot([0, 100], [0, 100], "--", color="gray", alpha=0.5)
        ax.set_title(layer)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Eval score bin (%)")
        ax.set_ylabel("Correct (%)")

    for j in range(len(layers), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Confidence Calibration (eval score vs correctness)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def statistical_report(results: dict, out_path: Path) -> str:
    lines = ["STATISTICAL SIGNIFICANCE REPORT", "=" * 50, ""]
    z = 1.96

    for layer, per_q in sorted(results.items()):
        total = len(per_q)
        correct = sum(1 for r in per_q.values() if r.get("correct"))
        p = correct / total if total else 0
        moe = z * (p * (1 - p) / total) ** 0.5 if total else 0
        lines.append(
            f"{layer:20} {p*100:.1f}%  [95% CI: {(p-moe)*100:.1f}% - {(p+moe)*100:.1f}%]  "
            f"({correct}/{total})"
        )

    lines.append("")
    lines.append("LAYER x DIFFICULTY ACCURACY")
    lines.append("-" * 40)
    matrix, cats, diffs = accuracy_matrix(results)
    for layer in matrix:
        lines.append(f"\n{layer}:")
        for cat in cats:
            row = []
            for d in diffs:
                v = matrix[layer][cat].get(d)
                row.append(f"{v:.0f}%" if v is not None else "N/A")
            lines.append(f"  {cat:12} easy/med/hard: {' / '.join(row)}")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


def run_all_analyses(results_path: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(results_path)

    paths = {
        "heatmap": plot_heatmap(results, OUTPUT_DIR / "heatmap_layer_category.png"),
        "efficiency": plot_efficiency_frontier(
            results, OUTPUT_DIR / "efficiency_frontier.png"
        ),
        "calibration": plot_confidence_calibration(
            results, OUTPUT_DIR / "confidence_calibration.png"
        ),
    }
    report = statistical_report(results, OUTPUT_DIR / "statistical_significance.txt")

    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    summary_lines = [
        "ANALYSIS OUTPUT SUMMARY",
        f"Results file: {results_path}",
        "",
        "Generated figures:",
        f"  heatmap_layer_category.png - Layer x category accuracy grid",
        f"  efficiency_frontier.png - Accuracy vs average tokens per question",
        f"  confidence_calibration.png - Eval score bins vs actual correctness",
        "",
        "Generated reports:",
        f"  statistical_significance.txt - 95% CIs and layer x difficulty breakdown",
        "",
        report[:2000],
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Figures saved to {OUTPUT_DIR}")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print(f"  report: {OUTPUT_DIR / 'statistical_significance.txt'}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="eval/results_improved.json",
        help="Path to results JSON",
    )
    args = parser.parse_args()
    base = Path(__file__).parent.parent
    path = base / args.results if not os.path.isabs(args.results) else args.results
    if not path.exists():
        print(f"Results not found: {path}")
        print("Run: python eval/runner_improved.py first")
    else:
        run_all_analyses(str(path))
