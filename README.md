# EDI2: Language Reasoning Model Research Framework

Complete technical reference for comparing six reasoning architectures on a structured question bank, with evaluation, analysis plots, and publication-oriented reporting.

---

## Table of Contents

1. [Project purpose](#1-project-purpose)
2. [Repository layout](#2-repository-layout)
3. [Setup and configuration](#3-setup-and-configuration)
4. [Reasoning layers (technical detail)](#4-reasoning-layers-technical-detail)
5. [Evaluation pipeline](#5-evaluation-pipeline)
6. [Dataset (55 questions)](#6-dataset-55-questions)
7. [Analysis and output figures](#7-analysis-and-output-figures)
8. [Stored results and how to read them](#8-stored-results-and-how-to-read-them)
9. [Running the system end to end](#9-running-the-system-end-to-end)
10. [Research notes and extensions](#10-research-notes-and-extensions)

---

## 1. Project purpose

This project implements multiple **reasoning layers** that wrap a Groq-hosted LLM. Each layer answers the same questions using a different search or decomposition strategy. An evaluation runner scores predictions against gold answers, saves JSON results, and an analysis module produces charts for research comparison.

Goals:

- Compare Linear, Self-Consistent, Tree, Graph, MCTS, and Hybrid approaches fairly (same model, same questions).
- Replace weak heuristics with **LLM-as-judge** scoring inside layers where it matters (Tree branches, Self-Consistent votes, Graph consistency).
- Support **self-correction** (Linear reflection and contradiction checks; Graph cycle breaking and consistency passes).
- Scale evaluation toward publication quality (55 questions, difficulty tiers, category breakdown, statistical reporting).

---

## 2. Repository layout

```
edi2/
  base_llm.py              # Groq client, retries, token stats, optional model override per call
  main.py                  # Demo: run all layers on one question
  requirements.txt
  reasoning/
    base.py                # Abstract ReasoningLayer interface
    llm_utils.py           # Score parsing, answer normalization for clustering
    linear.py              # Sequential chain + contradiction + reflection + confidence
    self_consistent.py     # N chains, quality-weighted clustered vote
    tree.py                # Branching search, LLM scorer, early exit
    graph.py               # Decompose, cycle-safe deps, consistency validation
    mcts.py                # UCB selection, expand, cheap rollout, backpropagate
    hybrid.py              # Tree prefix + Self-Consistent vote at leaf
  eval/
    questions.py           # 55-question bank with category and difficulty
    evaluator.py           # Numeric tolerance, keyword, form match, optional LLM judge
    runner_improved.py     # Full benchmark runner
    analysis.py            # Heatmap, efficiency frontier, calibration, CI report
    results_improved_*.json  # Benchmark output (gitignored)
    figures/               # Created after running analysis.py
  test_phase1.py           # Local validation without full API run
```

All documentation is consolidated in this README.

---

## 3. Setup and configuration

### 3.1 Dependencies

```bash
pip install -r requirements.txt
```

Packages: `groq`, `python-dotenv`, `matplotlib`, `numpy`.

### 3.2 API key

Create `.env` in the project root:

```
GROQ_API_KEY=your_key_here
```

### 3.3 Models (`base_llm.py`)

| Constant | Model | Role |
|----------|-------|------|
| `SMART_MODEL` | `llama-3.3-70b-versatile` | Default reasoning |
| `EVAL_MODEL` | `llama-3.1-8b-instant` | Fast eval / MCTS rollouts |
| `JUDGE_MODEL` | `llama-3.1-8b-instant` | Answer correctness judge |

`BaseLLM.call(prompt, model=None)` uses `model` when set (MCTS simulations use `EVAL_MODEL`).

Retry policy: exponential backoff, parses Groq "try again in Xs" messages, up to 6 attempts.

---

## 4. Reasoning layers (technical detail)

All layers implement `solve(question) -> dict` with at least `answer`, `steps`, `stats` (`llm_calls`, `total_tokens`, `model`).

### 4.1 Linear (`reasoning/linear.py`)

**Algorithm:** Up to 8 sequential steps. Each step prompt includes prior steps. Stops on `Final Answer:` or max steps.

**Phase 2 additions:**

| Feature | Mechanism | Extra LLM calls per step |
|---------|-----------|--------------------------|
| Contradiction detection | After each new step, ask YES/NO if it contradicts prior steps; on YES, pop last step and regenerate with consistency guidance | +1 |
| Step confidence | Rate step 1-10 for logical validity | +1 |
| Reflection | Final pass: "Review each step... produce corrected final answer" | +1 at end |

**Returns:** `step_scores`, `avg_confidence`, `backtrack_count`.

**Config:** `enable_contradiction_check`, `enable_reflection` (default True).

### 4.2 Self-Consistent (`reasoning/self_consistent.py`)

**Algorithm:** `N_CHAINS=8` independent full-chain generations. Extract short answer per chain. Score coherence 1-10. Drop chains with quality below 3.0 (outliers). Cluster answers via normalized forms (`600`, `600.0`, `six hundred` map to `num:600.0`). Pick cluster with highest **weighted score** (sum of quality scores).

**Voting fix:** Poor chains no longer count equally; formatting variants no longer split votes.

**Returns:** `chains`, `clusters`, `outliers_dropped`.

### 4.3 Tree (`reasoning/tree.py`)

**Algorithm:** Depth-limited beam search. At each depth, expand `BRANCH_COUNT=3` options, score each with LLM 1-10 ("logical validity and progress toward answer"), keep top `TOP_K=2`. Final answers allowed from `MIN_DEPTH=3`.

**Phase 2 additions:**

- **LLM scorer** replaces length/keyword heuristics (`_score_step`).
- **Early exit:** If any candidate scores >= `8.5` and contains `Final Answer:`, return immediately.
- **UCB constant** `1.41` documented for future branch selection (branches track `visits`).

**Returns:** `early_exit`, `exit_score` when triggered, `tree_structure` placeholder for visualization.

### 4.4 Graph (`reasoning/graph.py`)

**Algorithm:**

1. Decompose into 3-4 sub-questions (nodes).
2. LLM infers `depends_on` edges between nodes.
3. **Cycle detection** (DFS); break cycles by removing deps from first node in cycle.
4. **Topological order** with high fan-in nodes prioritized.
5. Resolve each node with context from dependencies.
6. Try global conclude; if insufficient, spawn one new node (max `MAX_NODES=8`).
7. **Post-resolution consistency:** LLM asked if conclusion follows; on INCONSISTENT, re-resolve flagged node IDs.

**Returns:** `graph` (resolved nodes), `is_consistent`.

### 4.5 MCTS (`reasoning/mcts.py`) NEW

Monte Carlo Tree Search over reasoning steps (AlphaGo-style loop):

| Phase | Action |
|-------|--------|
| Select | Walk from root choosing child with highest UCB until unvisited or leaf |
| Expand | Generate K=2 child steps from LLM |
| Simulate | Roll out with **cheap model** (`EVAL_MODEL`) to answer or max depth |
| Score rollout | LLM rates validity 1-10; reward = score/10 |
| Backpropagate | Add reward up the parent chain |

Default `num_iterations=12` (runner uses 10). Final answer: best rollout reward; steps: most-visited path.

**Returns:** `mcts_log`, `best_reward`, `nodes_explored`, `iterations`.

### 4.6 Hybrid (`reasoning/hybrid.py`) NEW

1. Run **Tree** (shallow: depth 3, 2 branches, top-1) to get a strong reasoning prefix.
2. Run **N=4 or 5** Self-Consistent chains continuing from that prefix.
3. Weighted clustered vote on final answers.

Combines exploration (Tree) with robustness (ensemble vote).

---

## 5. Evaluation pipeline

### 5.1 Correctness (`eval/evaluator.py`)

Strategies applied in order:

1. **numeric_tolerance:** First number in predicted vs expected, +/- 1% or 0.01.
2. **keyword_match:** 80% of expected keywords present.
3. **form_variation:** Normalized string similarity > 0.85.
4. **llm_judge:** If `--judge`, Groq scores 0-100; correct if >= 75.

### 5.2 Runner (`eval/runner_improved.py`)

```bash
# Full 55 questions, 6 layers (Linear, Self-Consistent, Tree, Graph, MCTS, Hybrid)
python eval/runner_improved.py

# Fast model, first 10 questions
python eval/runner_improved.py --fast --limit 10

# Skip expensive layers
python eval/runner_improved.py --no-mcts --no-hybrid

# Tune MCTS cost
python eval/runner_improved.py --mcts-iters 6

# Filter by category or difficulty
python eval/runner_improved.py --category Math --difficulty easy

# LLM judge for open-ended logic answers
python eval/runner_improved.py --judge
```

Output: `eval/results_improved_<model>.json` with per-question: `answer`, `correct`, `score`, `strategy`, `tokens`, `calls`, `time_s`, `category`, `difficulty`.

Delay between runs: 2 seconds (rate limit protection).

---

## 6. Dataset (55 questions)

Defined in `eval/questions.py`.

| Category | Count | Easy | Medium | Hard |
|----------|-------|------|--------|------|
| Math | 22 | 5 | 12 | 5 |
| Logic | 9 | 2 | 4 | 3 |
| Multi-hop | 7 | 1 | 4 | 2 |
| Probability | 6 | 2 | 3 | 1 |
| Adversarial | 6 | 1 | 3 | 2 |

**Sources tagged in each record:** `original`, `gsm8k-style`, `strategyqa-style`, `logiqa-style`, `adversarial`, `trick`.

**Fields:** `id`, `category`, `difficulty`, `question`, `answer`, `source`.

Examples:

- Q0 (Math/medium): Tank 3/5 full, add 120L, becomes 4/5. Answer: `600`.
- Q7 (Adversarial/hard): Bat and ball $1.10. Answer: `0.05`.
- Q43 (Adversarial/easy): Moses ark animals. Answer: `none` (trick: Noah).

Expand toward 100 by adding rows to `QUESTIONS` following the same schema.

---

## 7. Analysis and output figures

Run after evaluation:

```bash
python eval/analysis.py --results eval/results_improved_llama_3_3_70b_versatile.json
```

Outputs under `eval/figures/`:

### 7.1 `heatmap_layer_category.png`

**What it shows:** Rows = reasoning layers, columns = problem categories (Math, Logic, Multi-hop, Probability, Adversarial). Cell value = percent correct for that layer on all questions in that category. Color scale green (high) to red (low). Numeric label in each cell.

**How to interpret:** Identifies which layer wins per domain. Example pattern: Self-Consistent often leads on Math; Graph may lead on Multi-hop if decomposition helps; Adversarial column shows vulnerability to trick questions.

### 7.2 `efficiency_frontier.png`

**What it shows:** Scatter plot. X = average tokens per question for that layer. Y = overall accuracy percent. One point per layer, labeled.

**How to interpret:** Upper-left points are Pareto-efficient (accurate and cheap). Lower-right points waste tokens. Use to choose a layer for a fixed token budget.

### 7.3 `confidence_calibration.png`

**What it shows:** Subplot per layer. X = bins of evaluator `score` (0-0.5, 0.5-0.7, etc.). Y = whether answers were actually correct (0 or 100). Dashed diagonal = perfect calibration.

**How to interpret:** Points above diagonal = overconfident scoring; below = underconfident. Helps tune when to trust evaluator confidence vs run `--judge`.

### 7.4 `statistical_significance.txt`

**What it shows:** Per-layer accuracy with **95% confidence intervals** (binomial normal approximation). Section **Layer x Difficulty** with accuracy split easy / medium / hard per category row.

**How to interpret:** Overlapping CIs between two layers suggest difference may be noise until N is large enough. With 55 questions, CI width is roughly +/- 6-8 points at 70% accuracy.

### 7.5 `analysis_summary.txt`

Short index listing all generated artifacts and a preview of the statistical report.

### 7.6 Generated figures

After a benchmark, run `python eval/analysis.py --results eval/results_improved_<model>.json`.

Outputs in `eval/figures/`: `heatmap_layer_category.png`, `efficiency_frontier.png`, `confidence_calibration.png`, `statistical_significance.txt`, `analysis_summary.txt`.

### 7.7 Reasoning path visualization (planned extension)

Tree and Graph layers expose `steps` and partial `tree_structure` / `graph` in solve results. To plot branch trees, extend `runner_improved.py` to save `tree_structure` per question, then use NetworkX as described in `ANALYSIS_FRAMEWORK.md`.

---

## 8. Stored results and how to read them

### 8.1 Results JSON (`eval/results_improved_*.json`)

**Per-question fields:**

| Field | Meaning |
|-------|---------|
| `answer` | Model output or ERROR string |
| `correct` | Boolean from evaluator |
| `tokens` | Total tokens for that solve |
| `calls` | LLM API calls |
| `time_s` | Wall clock seconds |

### 8.2 Expected output after successful run

Console prints a summary table: question ID, category, per-layer Y/N and token count, then **ACCURACY BY LAYER** block.

Example (illustrative, not from a live run):

```
ACCURACY BY LAYER:
  Linear              38/55 (69.1%)
  Self-Consistent     47/55 (85.5%)
  Tree                42/55 (76.4%)
  Graph               40/55 (72.7%)
  MCTS                44/55 (80.0%)
  Hybrid              46/55 (83.6%)
```

Replace with your JSON after running with a valid API key.

### 8.3 Single-question demo (`main.py`)

```bash
python main.py
```

Runs all six layers on `QUESTIONS[0]` (tank capacity). Change index in `main.py` line selecting `QUESTIONS[n]`.

**Printed output per layer:**

- Numbered `steps` (truncated at 130 chars in console).
- `Final Answer`
- `stats`: `llm_calls`, `total_tokens`, `model`

---

## 9. Running the system end to end

### Step 1: Validate locally

```bash
python test_phase1.py
```

Checks imports, numeric scoring, form matching. API test optional if key present.

### Step 2: Smoke test one question

```bash
python main.py
```

### Step 3: Benchmark subset

```bash
python eval/runner_improved.py --limit 10 --fast
```

### Step 4: Full benchmark

```bash
python eval/runner_improved.py --mcts --hybrid
```

### Step 5: Analysis

```bash
python eval/analysis.py --results eval/results_improved_<your_model_file>.json
```

Open `eval/figures/*.png` and `statistical_significance.txt`.

### Step 6: Error analysis (manual)

For each layer, read 5 failures from JSON where `correct` is false. Categorize: arithmetic, logic, missing info, misinterpretation, hallucination, early stop, inconsistency. Template in `ANALYSIS_FRAMEWORK.md` section ANALYSIS 4.

---

## 10. Research notes and extensions

### 10.1 Priority implemented (from upgrade map)

1. Tree LLM scorer and early exit
2. Linear reflection and contradiction detection
3. Self-Consistent weighted vote and clustering
4. Graph cycle detection and consistency check
5. MCTS layer (UCB + rollout)
6. Hybrid Tree + Self-Consistent
7. Analysis scripts and 55-question dataset

### 10.2 Token cost guidance

| Layer | Relative cost | Notes |
|-------|---------------|-------|
| Linear | Medium | +2 calls/step with checks on |
| Self-Consistent | High | 8 chains + 8 scores + 8 extracts |
| Tree | High | 3 branches x depth x score each |
| Graph | Medium-High | Decompose + deps + consistency |
| MCTS | Very high | iterations x (expand + rollout) |
| Hybrid | Highest | Tree then partial Self-Consistent |

Use `--limit` and `--fast` during development.

### 10.3 Shared upgrades across layers

- **LLM-as-judge:** Inside layers (scoring) and eval (`--judge`).
- **Confidence:** Linear `step_scores`; Tree branch scores; MCTS `best_reward`.
- **Uncertainty propagation:** Future work: pass confidence into Graph node resolution.

### 10.4 Publication checklist

- [ ] Re-run eval with valid API key on full 55 questions
- [ ] Generate all figures via `analysis.py`
- [ ] Manual error analysis (5 per layer)
- [ ] Expand dataset toward 100 (see `DATASET_EXPANSION.md`)
- [ ] Add chi-square test (scipy) in `analysis.py` if scipy installed
- [ ] Save tree/graph structures in results JSON for path plots

### 10.5 Citing benchmarks

When adding GSM8K, StrategyQA, or LogiQA items, keep `source` field and note subset size in any paper. Current bank uses **style-aligned** questions, not full benchmark splits.

---

## Quick reference commands

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Demo all layers | `python main.py` |
| Eval 55 Q | `python eval/runner_improved.py` |
| Eval (6 layers) | `python eval/runner_improved.py` |
| Eval without MCTS | `python eval/runner_improved.py --no-mcts --no-hybrid` |
| Plots | `python eval/analysis.py --results <path>` |
| Math only | `python eval/runner_improved.py --category Math` |

---

## License and contact

Academic/research use. Set `GROQ_API_KEY` in `.env` (never commit; see `.gitignore`).

