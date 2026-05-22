from abc import ABC, abstractmethod
from base_llm import BaseLLM


class ReasoningLayer(ABC):
    """
    Every reasoning layer must inherit this class and implement solve().
    This shared interface lets us swap layers and compare them fairly.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @abstractmethod
    def solve(self, question: str) -> dict:
        """
        Takes a question string.
        Returns a dict with AT LEAST:
            {
                "answer"  : str,        # final answer
                "steps"   : list[str],  # intermediate reasoning steps
                "stats"   : dict,       # llm_calls, total_tokens
            }
        """
        pass

    def _build_step_prompt(self, question: str, steps_so_far: list[str]) -> str:
        """Helper: builds a prompt that includes all reasoning steps so far."""
        history = "\n".join(
            [f"Step {i+1}: {s}" for i, s in enumerate(steps_so_far)]
        )
        if history:
            return (
                f"Question: {question}\n\n"
                f"Reasoning so far:\n{history}\n\n"
                f"What is the next reasoning step? "
                f"If you have enough information, write 'Final Answer: <answer>'"
            )
        else:
            return (
                f"Question: {question}\n\n"
                f"Think step by step. "
                f"Write your first reasoning step, or if simple enough, "
                f"write 'Final Answer: <answer>'"
            )

    @staticmethod
    def _extract_final_answer(text: str) -> str | None:
        """Pulls out the answer if the LLM wrote 'Final Answer: ...'"""
        lower = text.lower()
        if "final answer:" in lower:
            idx = lower.index("final answer:") + len("final answer:")
            return text[idx:].strip()
        return None