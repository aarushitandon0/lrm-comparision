# EDI2: Language Reasoning Model Research Framework

A research framework for comparing multiple reasoning architectures on a structured question bank. Six reasoning layers wrap a Groq-hosted LLM and answer the same questions using different search and decomposition strategies. An evaluation runner scores predictions against gold answers and produces per-layer statistics.

---

## Table of Contents

1. [Project purpose](#1-project-purpose)
2. [Repository layout](#2-repository-layout)
3. [Setup and configuration](#3-setup-and-configuration)
4. [Reasoning layers](#4-reasoning-layers)
5. [Evaluation pipeline](#5-evaluation-pipeline)
6. [Dataset](#6-dataset)
7. [Running the system](#7-running-the-system)
8. [Research notes](#8-research-notes)

---

## 1. Project purpose

This project compares six reasoning strategies on the same set of questions using the same underlying LLM. The goal is to understand where each architecture succeeds and fails across problem categories and difficulty tiers.

Core design decisions:

- All layers share a single `BaseLLM` wrapper so token counts and API calls are measured uniformly.
- Weak heuristic scorers are replaced with LLM-as-judge scoring inside layers that require branch or chain quality assessment.
- Self-correction mechanisms (contradiction detection, cycle breaking, consistency passes) are built into individual layers rather than applied as a post-processing step.
- The evaluator supports multiple correctness strategies (numeric tolerance, keyword match, string similarity, LLM judge) so both exact and approximate answers are handled fairly.

---

## 2. Repository layout

```
edi2/
  base_llm.py                    # Groq client, retry logic, token tracking, per-call model override
  main.py                        # Run all layers on a single question for inspection
  requirements.txt
  reasoning/
    base.py                      # Abstract ReasoningLayer interface
    llm_utils.py                 # Score parsing, answer normalisation for clustering
    linear.py                    # Sequential chain with contradiction checks, confidence, reflection
    self_consistent.py           # N independent chains, quality-weighted clustered vote
    tree.py                      # Beam search with LLM scorer, early exit, UCB tracking
    graph.py                     # Decomposition, dependency resolution, cycle detection, consistency
    mcts.py                      # Monte Carlo Tree Search: select, expand, rollout, backpropagate
    hybrid.py                    # Tree prefix followed by Self-Consistent vote at leaf
  eval/
    questions.py                 # 55-question bank with category, difficulty, and source fields
    evaluator.py                 # Correctness strategies: numeric, keyword, form match, LLM judge
    runner_improved.py           # Benchmark runner with per-layer timing and token tracking
    results_improved_*.json      # Benchmark output (gitignored)
```

---

## 3. Setup and configuration

### 3.1 Install dependencies

```bash
pip install -r requirements.txt
```

Required packages: `groq`, `python-dotenv`, `matplotlib`, `numpy`.

### 3.2 API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

The key is loaded by `base_llm.py` via `python-dotenv`. Never commit `.env` to version control.

### 3.3 Models

Three model constants are defined in `base_llm.py`:

| Constant | Model string | Role |
|----------|--------------|------|
| `SMART_MODEL` | `llama-3.3-70b-versatile` | Default reasoning for all layers |
| `EVAL_MODEL` | `llama-3.1-8b-instant` | Fast rollouts inside MCTS simulations |
| `JUDGE_MODEL` | `llama-3.1-8b-instant` | Answer correctness judging in evaluator |

`BaseLLM.call(prompt, model=None)` accepts an optional model override. If not provided, `SMART_MODEL` is used. MCTS passes `EVAL_MODEL` during the simulate phase to reduce cost.

### 3.4 Retry policy

`BaseLLM` wraps every call in an exponential backoff loop with up to 6 attempts. It parses Groq rate-limit responses of the form `"try again in Xs"` and waits the specified duration before retrying.

---

## 4. Reasoning layers

All layers implement the interface defined in `base.py`:

```python
def solve(question: str) -> dict
```

The returned dict always contains `answer` (string), `steps` (list of strings), and `stats` (dict with `llm_calls`, `total_tokens`, `model`). Individual layers add extra keys documented below.

---

### 4.1 Linear

**File:** `reasoning/linear.py`

**Algorithm:** Generates up to 8 sequential reasoning steps. Each step prompt includes all prior steps as context. Generation stops when the model produces a `Final Answer:` token or the step limit is reached.

**Contradiction detection:** After each new step is generated, a separate LLM call asks whether the new step contradicts any prior step (YES/NO). On YES, the last step is popped and regenerated with a consistency instruction injected into the prompt. This costs one additional LLM call per step when enabled.

**Step confidence:** After each step, a second separate call rates logical validity on a scale of 1 to 10. Scores are stored in `step_scores` and averaged into `avg_confidence` in the return dict.

**Reflection:** A single final-pass call reviews the complete chain and produces a corrected answer if any errors are found. This costs one additional LLM call at the end of the chain.

**Config flags:** `enable_contradiction_check` and `enable_reflection` both default to `True`. Set to `False` to run a baseline with no self-correction.

**Extra return keys:** `step_scores`, `avg_confidence`, `backtrack_count`.

---

### 4.2 Self-Consistent

**File:** `reasoning/self_consistent.py`

**Algorithm:** Runs `N_CHAINS = 8` independent full reasoning chains in parallel (sequential API calls). Extracts a short final answer from each chain. Scores each chain's coherence 1 to 10 with an LLM call. Drops chains with quality below 3.0 as outliers. Clusters remaining answers by normalised form (`600`, `600.0`, and `six hundred` all map to `num:600.0`). Selects the cluster whose member chains have the highest summed quality score.

This addresses two weaknesses of basic majority voting: chains with poor reasoning no longer count equally, and formatting variation no longer splits votes across the same answer.

**Extra return keys:** `chains` (list of per-chain dicts with `answer`, `quality`), `clusters` (dict of normalised form to list of answers), `outliers_dropped` (count).

---

### 4.3 Tree

**File:** `reasoning/tree.py`

**Algorithm:** Depth-limited beam search over reasoning steps.

At each depth level, the layer generates `BRANCH_COUNT = 3` candidate next steps from the current best state. Each candidate is scored 1 to 10 by an LLM call asking for logical validity and progress toward the answer. The top `TOP_K = 2` candidates are kept and become the parents for the next depth level. Final answers are only accepted from depth `MIN_DEPTH = 3` or deeper.

**LLM scorer:** Replaces the original heuristic scorer (which used step length and keyword presence). The scorer prompt is: "Rate this reasoning step 1 to 10 for logical validity and progress toward the final answer. Respond with a number only."

**Early exit:** If any candidate at any depth scores 8.5 or above and contains `Final Answer:`, the layer returns immediately without exploring further branches. The return dict includes `early_exit: true` and `exit_score`.

**UCB tracking:** Each node stores `visits` and `total_reward` for potential UCB-based selection in future extensions.

**Extra return keys:** `early_exit`, `exit_score`, `tree_structure`.

---

### 4.4 Graph

**File:** `reasoning/graph.py`

**Algorithm:**

1. **Decompose:** The LLM breaks the question into 3 to 4 sub-questions (nodes).
2. **Infer dependencies:** A second call produces a `depends_on` mapping between nodes.
3. **Cycle detection:** A DFS check identifies any cycles in the dependency graph. Cycles are broken by removing the dependency from the first node encountered in the cycle.
4. **Topological sort:** Nodes are ordered so each node's dependencies are resolved before it is. Nodes with more dependents are resolved earlier.
5. **Resolve:** Each node is answered with context from its resolved dependencies injected into the prompt.
6. **Conclude:** A global conclusion call synthesises the resolved nodes into a final answer. If the conclusion is marked insufficient, one new node is spawned (up to `MAX_NODES = 8`).
7. **Consistency check:** A final LLM call asks whether the conclusion follows logically from all resolved nodes. On INCONSISTENT, the flagged node IDs are re-resolved.

**Extra return keys:** `graph` (list of resolved node dicts with `id`, `question`, `answer`, `depends_on`), `is_consistent`.

---

### 4.5 MCTS

**File:** `reasoning/mcts.py`

**Algorithm:** Monte Carlo Tree Search over reasoning steps. Each node in the tree represents one reasoning step. Default `num_iterations = 12` (runner uses 10).

| Phase | Action |
|-------|--------|
| Select | Walk from root, at each node choose the child with the highest UCB score until an unvisited node or leaf is reached |
| Expand | Generate `K = 2` child reasoning steps from the current node using `SMART_MODEL` |
| Simulate | Roll out from the expanded child to a final answer or max depth using `EVAL_MODEL` (cheaper) |
| Score | LLM rates the rollout answer's validity 1 to 10; reward = score / 10 |
| Backpropagate | Add reward to `total_reward` and increment `visits` for each node on the path to root |

UCB formula: `(total_reward / visits) + C * sqrt(ln(parent_visits) / visits)` where `C = 1.41`.

Final answer is taken from the rollout with the highest reward. The returned steps follow the most-visited path from root to leaf.

**Extra return keys:** `mcts_log`, `best_reward`, `nodes_explored`, `iterations`.

---

### 4.6 Hybrid

**File:** `reasoning/hybrid.py`

**Algorithm:**

1. Run Tree with reduced parameters (depth 3, 2 branches, top-1 kept) to produce a strong reasoning prefix.
2. Run 4 to 5 Self-Consistent chains that begin from the Tree prefix rather than from the raw question.
3. Apply weighted clustered vote (same mechanism as Self-Consistent) to the chains' final answers.

The Tree phase provides a high-quality starting point; the Self-Consistent phase adds robustness against single-chain errors.

**Extra return keys:** `tree_prefix`, `chains`, `clusters`.

---

## 5. Evaluation pipeline

### 5.1 Correctness strategies

`eval/evaluator.py` applies strategies in order and returns the first match:

1. **numeric_tolerance:** Extracts the first number from both prediction and expected answer. Correct if within 1% relative tolerance or 0.01 absolute tolerance.
2. **keyword_match:** Tokenises the expected answer. Correct if 80% or more of expected keywords appear in the prediction.
3. **form_variation:** Normalises both strings (lowercase, strip punctuation, collapse whitespace). Correct if normalised similarity exceeds 0.85.
4. **llm_judge:** If `--judge` flag is set, sends both strings to `JUDGE_MODEL` for a 0 to 100 correctness score. Correct if score >= 75.

The `strategy` field in results JSON records which strategy produced the verdict.

### 5.2 Runner

**File:** `eval/runner_improved.py`

The runner iterates over all questions and all layers, calls `layer.solve(question)`, passes the answer to the evaluator, and writes per-question results to a JSON file. A 2-second delay is inserted between layer runs on the same question to stay within Groq rate limits.

**CLI flags:**

```bash
# Full benchmark, all 55 questions, all 6 layers
python eval/runner_improved.py

# Use EVAL_MODEL for all layers, first 10 questions only
python eval/runner_improved.py --fast --limit 10

# Skip MCTS and Hybrid (fastest benchmark)
python eval/runner_improved.py --no-mcts --no-hybrid

# Reduce MCTS iterations
python eval/runner_improved.py --mcts-iters 6

# Filter by category or difficulty
python eval/runner_improved.py --category Math --difficulty easy

# Enable LLM judge for open-ended answers
python eval/runner_improved.py --judge
```

**Output file:** `eval/results_improved_<model>.json`

**Per-question fields in results:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Model output or ERROR string |
| `correct` | boolean | Verdict from evaluator |
| `score` | float | Evaluator confidence (0 to 1) |
| `strategy` | string | Which correctness strategy matched |
| `tokens` | integer | Total tokens consumed for this solve |
| `calls` | integer | Number of LLM API calls |
| `time_s` | float | Wall clock seconds |
| `category` | string | Question category |
| `difficulty` | string | easy, medium, or hard |

---

## 6. Dataset

**File:** `eval/questions.py`

55 questions across five categories and three difficulty tiers.

| Category | Total | Easy | Medium | Hard |
|----------|-------|------|--------|------|
| Math | 22 | 5 | 12 | 5 |
| Logic | 9 | 2 | 4 | 3 |
| Multi-hop | 7 | 1 | 4 | 2 |
| Probability | 6 | 2 | 3 | 1 |
| Adversarial | 6 | 1 | 3 | 2 |

Each question record contains: `id`, `category`, `difficulty`, `question`, `answer`, `source`.

The `source` field tags where each question style originates: `original`, `gsm8k-style`, `strategyqa-style`, `logiqa-style`, `adversarial`, or `trick`. Questions are style-aligned with published benchmarks, not direct copies from benchmark test splits.

Selected examples:

| ID | Category | Difficulty | Question summary | Expected answer |
|----|----------|------------|-----------------|-----------------|
| Q0 | Math | medium | Tank 3/5 full, add 120L, becomes 4/5. Capacity? | 600 |
| Q7 | Adversarial | hard | Bat and ball cost $1.10. Bat costs $1 more. Ball costs? | 0.05 |
| Q43 | Adversarial | easy | How many animals did Moses take on the ark? | none |

To expand the dataset, add rows to the `QUESTIONS` list in `eval/questions.py` following the same schema.

---

## 7. Running the system

### Step 1: Validate imports locally

```bash
python test_phase1.py
```

Checks that all modules import correctly and that the numeric and form-match evaluator strategies work without an API call.

### Step 2: Smoke test on one question

```bash
python main.py
```

Runs all six layers on `QUESTIONS[0]` and prints truncated steps, final answer, and stats per layer. To test a different question, change the index on the `QUESTIONS[n]` line in `main.py`.

### Step 3: Benchmark a subset

```bash
python eval/runner_improved.py --limit 10 --fast
```

### Step 4: Full benchmark

```bash
python eval/runner_improved.py
```

### Step 5: Benchmark without expensive layers

```bash
python eval/runner_improved.py --no-mcts --no-hybrid
```

---

## 8. Research notes

### 8.1 Token cost by layer

| Layer | Relative cost | Notes |
|-------|---------------|-------|
| Linear | Medium | +2 LLM calls per step when checks are enabled |
| Self-Consistent | High | 8 chains plus 8 quality scores plus 8 answer extractions |
| Tree | High | 3 branches x depth x 1 score call each |
| Graph | Medium-High | Decompose, dependency inference, consistency check |
| MCTS | Very high | iterations x (expand call + rollout call + score call) |
| Hybrid | Highest | Full Tree run followed by partial Self-Consistent |

Use `--limit` and `--fast` during development to avoid exhausting API quota.

### 8.2 Disabling self-correction in Linear

To isolate the contribution of contradiction detection and reflection, set both flags to `False` in `linear.py` or pass config on construction:

```python
layer = Linear(llm, enable_contradiction_check=False, enable_reflection=False)
```

This produces a clean baseline matching the original single-pass chain behaviour.

### 8.3 Extending the dataset

The dataset is designed to grow toward 100 questions. Add entries to `QUESTIONS` in `eval/questions.py`. Maintain the `source` field accurately, especially for any questions derived from GSM8K, StrategyQA, or LogiQA, so benchmark provenance is clear in any write-up.

### 8.4 Adding a chi-square significance test

If `scipy` is available, add the following to the analysis script to test whether layer accuracy differences are statistically significant beyond the binomial confidence intervals:

```python
from scipy.stats import chi2_contingency
# Build a 2 x N contingency table: [correct_counts], [incorrect_counts] per layer
# chi2_contingency returns (chi2, p, dof, expected)
```

A p-value below 0.05 with 55 questions and the expected accuracy spread should be achievable for the largest observed differences.

### 8.5 Saving tree and graph structures for path visualisation

Tree and Graph layers return `tree_structure` and `graph` keys in their solve dicts. To plot these, modify `runner_improved.py` to persist these fields in the results JSON alongside the standard fields, then use NetworkX to build and render the graph:

```python
import networkx as nx
G = nx.DiGraph()
for node in result["graph"]:
    G.add_node(node["id"], label=node["question"][:40])
    for dep in node["depends_on"]:
        G.add_edge(dep, node["id"])
nx.draw(G, with_labels=True)
```

---

## Quick reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Single question demo | `python main.py` |
| Benchmark 55 questions | `python eval/runner_improved.py` |
| Benchmark without MCTS | `python eval/runner_improved.py --no-mcts --no-hybrid` |
| Math category only | `python eval/runner_improved.py --category Math` |
| Hard questions only | `python eval/runner_improved.py --difficulty hard` |
| Fast dev run | `python eval/runner_improved.py --limit 10 --fast` |

---

## License

Academic and research use. Set `GROQ_API_KEY` in `.env` and ensure it is listed in `.gitignore` before pushing.
