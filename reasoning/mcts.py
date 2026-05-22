"""Monte Carlo Tree Search reasoning layer."""

import math
import re
from reasoning.base import ReasoningLayer
from reasoning.llm_utils import parse_score
from base_llm import BaseLLM, EVAL_MODEL

DEFAULT_ITERATIONS = 4
DEFAULT_EXPAND_K = 2
UCB_C = 1.41


class MCTSNode:
    def __init__(self, steps: list, parent=None):
        self.steps = steps
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.last_score = 0.0
        self.last_answer = None

    def ucb_value(self, c: float = UCB_C) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value_sum / self.visits
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c * math.sqrt(math.log(parent_visits + 1) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = UCB_C) -> "MCTSNode":
        return max(self.children, key=lambda ch: ch.ucb_value(c))

    def backpropagate(self, reward: float):
        node = self
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent


class MCTSReasoning(ReasoningLayer):
    """
    MCTS reasoning: Select (UCB) -> Expand (K children) -> Simulate (cheap rollout)
    -> Backpropagate (quality signal).
    """

    def __init__(
        self,
        llm: BaseLLM,
        num_iterations: int = DEFAULT_ITERATIONS,
        max_depth: int = 8,
        expand_k: int = DEFAULT_EXPAND_K,
        rollout_model: str = EVAL_MODEL,
        expand_model: str | None = None,
    ):
        super().__init__(llm)
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.expand_k = expand_k
        self.rollout_model = rollout_model
        self.expand_model = expand_model or rollout_model

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children:
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return unvisited[0]
            node = node.best_child()
        return node

    def _expand_children(self, node: MCTSNode, question: str) -> list[MCTSNode]:
        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning so far:\n" + "\n".join(node.steps) + "\n\n"
            f"Generate {self.expand_k} DIFFERENT next reasoning steps.\n"
            f"Format:\nOption 1: <step>\nOption 2: <step>"
        )
        response = self.llm.call(prompt, model=self.expand_model)
        candidates = []
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("option"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    candidates.append(parts[1].strip())
        if not candidates:
            candidates = [response.strip()]

        children = []
        for cand in candidates[: self.expand_k]:
            child = MCTSNode(steps=node.steps + [cand], parent=node)
            node.children.append(child)
            children.append(child)
        return children

    def _score_rollout(self, question: str, steps: list, answer: str) -> float:
        if not answer or answer == "No answer":
            return 0.0
        history = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning:\n{history}\n\n"
            f"Proposed answer: {answer}\n\n"
            f"Rate logical validity and likelihood of correctness from 1-10.\n"
            f"Reply with ONLY a single integer 1-10."
        )
        response = self.llm.call(prompt, model=self.rollout_model)
        return parse_score(response, 1, 10) / 10.0

    def _simulate(self, node: MCTSNode, question: str) -> tuple[float, str]:
        steps = list(node.steps)
        answer = self._extract_final_answer(steps[-1]) if steps else None
        depth = len(steps)

        while not answer and depth < self.max_depth:
            prompt = self._build_step_prompt(question, steps)
            response = self.llm.call(prompt, model=self.rollout_model)
            steps.append(response)
            depth += 1
            answer = self._extract_final_answer(response)

        if not answer and steps:
            conclude = (
                f"Question: {question}\n\n"
                f"Reasoning:\n" + "\n".join(steps) + "\n\n"
                f"Final answer only, no explanation:"
            )
            answer = self.llm.call(conclude, model=self.rollout_model).strip()

        answer = answer or "No answer"
        reward = self._score_rollout(question, steps, answer)
        node.last_answer = answer
        node.last_score = reward
        return reward, answer

    def _most_visited_path(self, root: MCTSNode) -> list[str]:
        path_steps = []
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            if node.steps:
                path_steps.append(node.steps[-1])
        return path_steps

    def _count_nodes(self, node: MCTSNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)

    @staticmethod
    def _clean_short_answer(text: str) -> str:
        extracted = ReasoningLayer._extract_final_answer(text)
        if extracted:
            return extracted.split("\n")[0].strip()[:80]
        first_line = text.strip().split("\n")[0].strip()
        nums = re.findall(r"-?\d+\.?\d*", first_line)
        if nums and len(first_line) < 120:
            return nums[0]
        return first_line[:80]

    def solve(self, question: str) -> dict:
        self.llm.reset_stats()
        root = MCTSNode(steps=[])
        log = []
        best_answer = None
        best_reward = -1.0

        for it in range(self.num_iterations):
            leaf = self._select(root)

            if len(leaf.steps) < self.max_depth:
                children = self._expand_children(leaf, question)
                sim_node = children[0] if children else leaf
            else:
                sim_node = leaf

            reward, answer = self._simulate(sim_node, question)
            sim_node.backpropagate(reward)

            if reward > best_reward:
                best_reward = reward
                best_answer = answer

            log.append(
                f"[iter {it+1}] depth={len(sim_node.steps)} "
                f"reward={reward:.2f} answer={str(answer)[:40]}"
            )

        path = self._most_visited_path(root)
        if not best_answer:
            best_answer = "No answer found in MCTS exploration"
        else:
            best_answer = self._clean_short_answer(best_answer)

        return {
            "answer": best_answer,
            "steps": path if path else log[:8],
            "mcts_log": log,
            "best_reward": best_reward,
            "iterations": self.num_iterations,
            "nodes_explored": self._count_nodes(root),
            "stats": self.llm.get_stats(),
        }
