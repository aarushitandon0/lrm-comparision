from difflib import SequenceMatcher
from reasoning.base import ReasoningLayer
from reasoning.llm_utils import parse_score, normalize_answer_for_clustering
from base_llm import BaseLLM

N_CHAINS = 8
OUTLIER_QUALITY_THRESHOLD = 3.0


class SelfConsistentReasoning(ReasoningLayer):
    """
    N independent chains with LLM quality scoring, answer clustering,
    weighted voting, and outlier chain removal.
    """

    def __init__(self, llm: BaseLLM, n_chains: int = N_CHAINS):
        super().__init__(llm)
        self.n_chains = n_chains

    def _score_chain_quality(self, question: str, reasoning: str, answer: str) -> float:
        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Final answer: {answer}\n\n"
            f"Rate logical coherence 1-10. Reply with ONLY a single integer 1-10."
        )
        response = self.llm.call(prompt).strip()
        return parse_score(response, 1, 10)

    def _normalize_answer(self, answer: str) -> str:
        return normalize_answer_for_clustering(answer)

    def _cluster_answers(self, answers_with_scores: list) -> dict:
        clusters = {}

        for answer, score in answers_with_scores:
            normalized = self._normalize_answer(answer)
            best_cluster = None
            best_similarity = 0.85

            for cluster_key in clusters:
                similarity = SequenceMatcher(None, normalized, cluster_key).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_key

            if best_cluster:
                clusters[best_cluster]["answers"].append(answer)
                clusters[best_cluster]["weights"].append(score)
                clusters[best_cluster]["score"] += score
            else:
                clusters[normalized] = {
                    "answers": [answer],
                    "weights": [score],
                    "score": score,
                }

        return clusters

    def _drop_outliers(self, chains: list) -> list:
        return [
            c for c in chains
            if c.get("quality_score", 5) >= OUTLIER_QUALITY_THRESHOLD
        ]

    def solve(self, question: str) -> dict:
        self.llm.reset_stats()
        all_chains = []

        for i in range(self.n_chains):
            prompt = self._build_step_prompt(question, [])
            response = self.llm.call(prompt)
            extract_prompt = (
                f"Given this reasoning:\n{response}\n\n"
                f"What is the final answer? Reply with ONLY the answer value."
            )
            clean_answer = self.llm.call(extract_prompt).strip()
            all_chains.append({
                "chain": i + 1,
                "response": response,
                "answer": clean_answer,
            })

        for chain in all_chains:
            chain["quality_score"] = self._score_chain_quality(
                question, chain["response"], chain["answer"]
            )

        filtered = self._drop_outliers(all_chains)
        dropped = len(all_chains) - len(filtered)
        vote_chains = filtered if filtered else all_chains

        answers_with_scores = [
            (c["answer"], c["quality_score"]) for c in vote_chains
        ]
        clusters = self._cluster_answers(answers_with_scores)

        def cluster_key_score(k):
            c = clusters[k]
            avg_q = sum(c["weights"]) / len(c["weights"])
            return (c["score"], avg_q, len(c["answers"]))

        best_cluster_key = max(clusters.keys(), key=cluster_key_score)
        final_answer = clusters[best_cluster_key]["answers"][0]

        steps = [f"Generated {self.n_chains} chains, dropped {dropped} outliers"]
        for chain in all_chains:
            marker = " (outlier)" if chain not in vote_chains else ""
            steps.append(
                f"Chain {chain['chain']}: {chain['answer'][:50]} "
                f"quality={chain['quality_score']:.1f}{marker}"
            )
        steps.append(f"Clusters: {len(clusters)}")
        for ck, cl in sorted(clusters.items(), key=lambda x: x[1]["score"], reverse=True):
            steps.append(
                f"  '{ck}': n={len(cl['answers'])} weighted={cl['score']:.1f}"
            )
        steps.append(f"Selected: {final_answer}")

        return {
            "answer": final_answer,
            "steps": steps,
            "chains": all_chains,
            "clusters": clusters,
            "outliers_dropped": dropped,
            "stats": self.llm.get_stats(),
        }
