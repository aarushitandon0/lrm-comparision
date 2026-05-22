from reasoning.base import ReasoningLayer
from reasoning.llm_utils import parse_score
from base_llm import BaseLLM

MAX_STEPS = 8


class LinearReasoning(ReasoningLayer):
    """
    Sequential reasoning with mid-chain contradiction detection,
    per-step confidence scores, and end-of-chain reflection.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int = MAX_STEPS,
        enable_contradiction_check: bool = True,
        enable_reflection: bool = True,
    ):
        super().__init__(llm)
        self.max_steps = max_steps
        self.enable_contradiction_check = enable_contradiction_check
        self.enable_reflection = enable_reflection

    def _detect_contradiction(self, question: str, steps: list, new_step: str) -> bool:
        if len(steps) < 1:
            return False

        history = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        prompt = (
            f"Question: {question}\n\n"
            f"Previous steps:\n{history}\n\n"
            f"New proposed step:\n{new_step}\n\n"
            f"Does this new step contradict any previous step? "
            f"Reply with ONLY: YES or NO"
        )
        response = self.llm.call(prompt).strip().upper()
        return "YES" in response

    def _score_step_confidence(self, question: str, step: str) -> float:
        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning step: {step}\n\n"
            f"Rate confidence in this step's logical validity (1-10).\n"
            f"Reply with ONLY a single integer 1-10."
        )
        response = self.llm.call(prompt).strip()
        return parse_score(response, 1, 10)

    def _reflection_pass(self, question: str, steps: list) -> str:
        history = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        prompt = (
            f"Question: {question}\n\n"
            f"Complete reasoning chain:\n{history}\n\n"
            f"Review each step above. Identify any logical errors and produce "
            f"a corrected final answer.\n"
            f"If correct, confirm. End with: Final Answer: <value>"
        )
        return self.llm.call(prompt).strip()

    def solve(self, question: str) -> dict:
        self.llm.reset_stats()
        steps = []
        step_scores = []
        answer = None
        backtrack_count = 0
        retries_at_step = 0
        max_retries = 2

        step_num = 0
        while step_num < self.max_steps:
            prompt = self._build_step_prompt(question, steps)
            response = self.llm.call(prompt)

            if (
                self.enable_contradiction_check
                and steps
                and self._detect_contradiction(question, steps, response)
            ):
                if steps:
                    steps.pop()
                    if step_scores:
                        step_scores.pop()
                backtrack_count += 1
                retries_at_step += 1
                if retries_at_step <= max_retries:
                    guidance = (
                        f"Question: {question}\n\n"
                        f"Valid steps so far:\n"
                        + "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
                        + "\n\nNext step must be logically consistent. "
                        f"No contradictions."
                    )
                    response = self.llm.call(guidance)
                else:
                    retries_at_step = 0
                    step_num += 1
                    continue

            confidence = self._score_step_confidence(question, response)
            steps.append(response)
            step_scores.append(confidence)
            retries_at_step = 0
            step_num += 1

            answer = self._extract_final_answer(response)
            if answer:
                break

        if not answer:
            answer = steps[-1] if steps else "No answer produced."

        if self.enable_reflection and steps:
            reflection_answer = self._reflection_pass(question, steps)
            steps.append(f"[REFLECTION]: {reflection_answer[:200]}")
            reflected = self._extract_final_answer(reflection_answer)
            if reflected:
                answer = reflected

        avg_conf = sum(step_scores) / len(step_scores) if step_scores else 0.0

        return {
            "answer": answer,
            "steps": steps,
            "step_scores": step_scores,
            "avg_confidence": avg_conf,
            "backtrack_count": backtrack_count,
            "stats": self.llm.get_stats(),
        }
