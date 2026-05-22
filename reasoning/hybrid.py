"""Hybrid: Tree exploration at branches, Self-Consistent voting at leaves."""

from reasoning.base import ReasoningLayer
from reasoning.tree import TreeReasoning
from reasoning.self_consistent import SelfConsistentReasoning
from base_llm import BaseLLM


class HybridTreeConsistentReasoning(ReasoningLayer):
    """
    Explore reasoning branches with Tree search, then run N independent
    completion chains from the best leaf and majority-vote (weighted) the answer.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tree_depth: int = 3,
        n_chains: int = 5,
        branch_count: int = 2,
    ):
        super().__init__(llm)
        self.tree = TreeReasoning(
            llm,
            max_depth=tree_depth,
            branch_count=branch_count,
            top_k=1,
            min_depth=2,
        )
        self.voter = SelfConsistentReasoning(llm, n_chains=n_chains)

    def solve(self, question: str) -> dict:
        self.llm.reset_stats()
        steps = []

        tree_result = self.tree.solve(question)
        steps.append("[Tree phase] Best exploration path selected")

        prefix_steps = []
        for s in tree_result.get("steps", []):
            if s.startswith("[depth") and "PRUNED" not in s and "FINAL" not in s:
                if "]" in s:
                    chunk = s.split("]", 2)
                    if len(chunk) >= 3:
                        prefix_steps.append(chunk[-1].strip())

        if not prefix_steps:
            prefix_steps = tree_result.get("steps", [])[-3:]

        steps.append(f"[Tree prefix] {len(prefix_steps)} steps retained")
        for i, p in enumerate(prefix_steps[:5], 1):
            steps.append(f"  prefix {i}: {p[:100]}")

        chains = []
        answers_with_scores = []

        for i in range(self.voter.n_chains):
            prompt = self._build_step_prompt(question, prefix_steps)
            response = self.llm.call(prompt)
            extract_prompt = (
                f"Given this reasoning:\n{response}\n\n"
                f"What is the final answer? Reply with ONLY the answer value."
            )
            clean_answer = self.llm.call(extract_prompt).strip()
            score = self.voter._score_chain_quality(question, response, clean_answer)
            chains.append({
                "chain": i + 1,
                "response": response,
                "answer": clean_answer,
                "quality_score": score,
            })
            answers_with_scores.append((clean_answer, score))

        clusters = self.voter._cluster_answers(answers_with_scores)
        best_key = max(clusters.keys(), key=lambda k: clusters[k]["score"])
        final_answer = clusters[best_key]["answers"][0]

        steps.append(f"[Vote phase] {self.voter.n_chains} chains from tree leaf")
        for c in chains:
            steps.append(
                f"  Chain {c['chain']}: {c['answer'][:50]} "
                f"(quality {c['quality_score']:.1f})"
            )
        steps.append(f"[Selected] {final_answer} (cluster {best_key})")

        return {
            "answer": final_answer,
            "steps": steps,
            "tree_answer": tree_result.get("answer"),
            "chains": chains,
            "clusters": clusters,
            "stats": self.llm.get_stats(),
        }
