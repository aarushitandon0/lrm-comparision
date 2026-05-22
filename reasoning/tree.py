import math
from reasoning.base import ReasoningLayer
from reasoning.llm_utils import parse_score
from base_llm import BaseLLM

MAX_DEPTH = 4
BRANCH_COUNT = 3
TOP_K = 2
MIN_DEPTH = 3
EARLY_EXIT_THRESHOLD = 8.5
UCB_C = 1.41


class TreeReasoning(ReasoningLayer):
    def __init__(
        self,
        llm: BaseLLM,
        max_depth: int = MAX_DEPTH,
        branch_count: int = BRANCH_COUNT,
        top_k: int = TOP_K,
        min_depth: int = MIN_DEPTH,
        early_exit_threshold: float = EARLY_EXIT_THRESHOLD,
    ):
        super().__init__(llm)
        self.max_depth = max_depth
        self.branch_count = branch_count
        self.top_k = top_k
        self.min_depth = min_depth
        self.early_exit_threshold = early_exit_threshold

    def _score_step(self, question: str, steps_so_far: list, candidate: str) -> float:
        history = "\n".join(steps_so_far) if steps_so_far else "(none)"
        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning so far:\n{history}\n\n"
            f"Candidate next step:\n{candidate}\n\n"
            f"Rate this reasoning step 1-10 for logical validity and progress "
            f"toward the answer. Respond with just a number."
        )
        response = self.llm.call(prompt).strip()
        return parse_score(response, 1, 10)

    def _ucb_select_branch(self, branches: list, parent_visits: int) -> dict:
        def ucb(branch):
            visits = branch.get("visits", 1)
            avg = branch["cumulative_score"] / max(1, len(branch["steps"]))
            if visits == 0:
                return float("inf")
            explore = UCB_C * math.sqrt(math.log(parent_visits + 1) / visits)
            return avg + explore

        return max(branches, key=ucb)

    def _expand(self, question: str, steps_so_far: list, allow_final: bool) -> list:
        if allow_final:
            final_instruction = (
                "If ready, include 'Final Answer: <value>' as one option."
            )
        else:
            final_instruction = "Do NOT give a final answer yet."

        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning so far:\n" + "\n".join(steps_so_far) + "\n\n"
            f"Generate {self.branch_count} DIFFERENT next steps.\n"
            f"{final_instruction}\n"
            f"Format:\nOption 1: <step>\nOption 2: <step>\nOption 3: <step>"
        )
        response = self.llm.call(prompt)
        candidates = []
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("option"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    candidates.append(parts[1].strip())
        return candidates or [response.strip()]

    def solve(self, question: str) -> dict:
        self.llm.reset_stats()
        all_steps = []
        active_branches = [{"steps": [], "cumulative_score": 0.0, "visits": 0}]
        final_answer = None
        best_final = {"answer": None, "score": -1}
        early_exit = False
        tree_structure = {"id": "root", "children": []}

        parent_visits = 0

        for depth in range(1, self.max_depth + 1):
            next_branches = []
            allow_final = depth >= self.min_depth
            parent_visits += 1

            ordered = sorted(
                active_branches,
                key=lambda b: b["cumulative_score"] / max(1, len(b["steps"]) + 1),
                reverse=True,
            )

            for branch in ordered:
                branch["visits"] = branch.get("visits", 0) + 1
                candidates = self._expand(question, branch["steps"], allow_final)
                scored_candidates = []

                for candidate in candidates:
                    score = self._score_step(question, branch["steps"], candidate)
                    cumulative = branch["cumulative_score"] + score

                    if score >= self.early_exit_threshold:
                        ans = self._extract_final_answer(candidate)
                        if ans:
                            all_steps.append(
                                f"[EARLY EXIT depth {depth}] score={score:.1f} "
                                f"answer={ans}"
                            )
                            return {
                                "answer": ans,
                                "steps": all_steps,
                                "early_exit": True,
                                "exit_score": score,
                                "tree_structure": tree_structure,
                                "stats": self.llm.get_stats(),
                            }

                    if allow_final:
                        answer = self._extract_final_answer(candidate)
                        if answer:
                            all_steps.append(
                                f"[depth {depth}] score={score:.1f} "
                                f"FINAL: {candidate[:100]}"
                            )
                            if cumulative > best_final["score"]:
                                best_final = {"answer": answer, "score": cumulative}
                            continue

                    all_steps.append(
                        f"[depth {depth}] score={score:.1f} "
                        f"total={cumulative:.1f} {candidate[:100]}"
                    )
                    scored_candidates.append({
                        "steps": branch["steps"] + [candidate],
                        "cumulative_score": cumulative,
                        "last_score": score,
                        "visits": 0,
                    })

                scored_candidates.sort(
                    key=lambda x: x["cumulative_score"], reverse=True
                )
                kept = scored_candidates[: self.top_k]
                for p in scored_candidates[self.top_k :]:
                    all_steps.append(
                        f"[depth {depth}] score={p['last_score']:.1f} PRUNED"
                    )
                next_branches.extend(kept)

            active_branches = next_branches
            if not active_branches:
                break

        final_answer = best_final["answer"]
        if not final_answer and active_branches:
            active_branches.sort(
                key=lambda x: x["cumulative_score"], reverse=True
            )
            best = active_branches[0]
            conclude = (
                f"Question: {question}\n\n"
                f"Reasoning:\n" + "\n".join(best["steps"]) + "\n\n"
                f"Final numerical answer only:"
            )
            final_answer = self.llm.call(conclude).strip()
            all_steps.append(f"[fallback] {final_answer}")

        return {
            "answer": final_answer or "No answer produced.",
            "steps": all_steps,
            "early_exit": early_exit,
            "tree_structure": tree_structure,
            "stats": self.llm.get_stats(),
        }
