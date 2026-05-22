"""
evaluator.py — Smart correctness evaluation with multiple strategies.

Implements:
1. Numeric tolerance for math answers
2. LLM-as-judge for logic/open-ended answers  
3. Multiple valid answer forms recognition
"""

import re
from base_llm import BaseLLM, JUDGE_MODEL, JUDGE_TEMP, JUDGE_TOKENS


class CorrectnesEvaluator:
    """Multi-strategy evaluator for answer correctness."""
    
    def __init__(self, llm: BaseLLM = None, api_key: str = None):
        """Initialize with optional LLM for judge-based evaluation."""
        if llm is not None:
            self.judge_llm = llm
        elif api_key is not None:
            self.judge_llm = BaseLLM(api_key=api_key, model=JUDGE_MODEL, temperature=JUDGE_TEMP)
        else:
            self.judge_llm = None
        
        self.evaluation_count = 0
    
    def evaluate(self, predicted: str, expected: str, question: str = "", category: str = "") -> dict:
        """
        Evaluate if predicted answer is correct.
        
        Returns:
            {
                "correct": bool,
                "score": float (0.0-1.0),
                "confidence": str ("high", "medium", "low"),
                "strategy_used": str,
                "reason": str,
                "predicted_clean": str,
                "expected_clean": str,
            }
        """
        self.evaluation_count += 1
        
        # Try numeric evaluation first (most reliable)
        numeric_result = self._try_numeric(predicted, expected)
        if numeric_result is not None:
            return {
                **numeric_result,
                "strategy_used": "numeric_tolerance",
            }
        
        # Try keyword matching (fast, fragile)
        keyword_result = self._try_keyword_match(predicted, expected)
        if keyword_result["score"] > 0.9:
            return {
                **keyword_result,
                "strategy_used": "keyword_match",
            }
        
        # Try semantic/form variations
        form_result = self._try_form_variations(predicted, expected)
        if form_result["score"] > 0.8:
            return {
                **form_result,
                "strategy_used": "form_variation",
            }
        
        # Fallback: Use LLM judge if available
        if self.judge_llm is not None:
            judge_result = self._llm_judge(predicted, expected, question, category)
            return {
                **judge_result,
                "strategy_used": "llm_judge",
            }
        
        # Last resort: return medium confidence keyword match
        return {
            **keyword_result,
            "strategy_used": "keyword_match_fallback",
        }
    
    def _try_numeric(self, predicted: str, expected: str) -> dict | None:
        """Try to parse as numbers and check with tolerance."""
        try:
            # Extract first number from predicted
            pred_nums = re.findall(r'-?\d+\.?\d*', predicted.lower())
            if not pred_nums:
                return None
            pred_val = float(pred_nums[0])
            
            # Extract first number from expected
            exp_nums = re.findall(r'-?\d+\.?\d*', expected.lower())
            if not exp_nums:
                return None
            exp_val = float(exp_nums[0])
            
            # Check with tolerance
            tolerance = max(abs(exp_val) * 0.01, 0.01)  # 1% or 0.01, whichever is larger
            is_correct = abs(pred_val - exp_val) <= tolerance
            
            return {
                "correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "confidence": "high",
                "reason": f"Numeric match: {pred_val} vs {exp_val} (tolerance: ±{tolerance:.2f})",
                "predicted_clean": str(pred_val),
                "expected_clean": str(exp_val),
            }
        except (ValueError, IndexError):
            return None
    
    def _try_keyword_match(self, predicted: str, expected: str) -> dict:
        """Check if all keywords from expected appear in predicted."""
        pred = predicted.lower().strip()
        exp = expected.lower().strip()
        
        # Split into keywords
        keywords = [k.strip() for k in exp.split() if len(k.strip()) > 2]
        
        if not keywords:
            # If no keywords (very short answer), do exact match
            is_correct = pred == exp
            score = 1.0 if is_correct else 0.0
        else:
            # Count how many keywords match
            matches = sum(1 for kw in keywords if kw in pred)
            score = matches / len(keywords)
            is_correct = score >= 0.8  # 80% of keywords must match
        
        return {
            "correct": is_correct,
            "score": score,
            "confidence": "medium" if 0.5 < score < 0.9 else ("high" if score >= 0.9 else "low"),
            "reason": f"Keyword match: {score*100:.0f}% of keywords found",
            "predicted_clean": pred[:100],
            "expected_clean": exp[:100],
        }
    
    def _try_form_variations(self, predicted: str, expected: str) -> dict:
        """Try to match different valid forms of the same answer."""
        pred_clean = self._normalize_answer(predicted)
        exp_clean = self._normalize_answer(expected)
        
        # Exact match on normalized form
        if pred_clean == exp_clean:
            return {
                "correct": True,
                "score": 1.0,
                "confidence": "high",
                "reason": "Exact match on normalized form",
                "predicted_clean": pred_clean,
                "expected_clean": exp_clean,
            }
        
        # Partial match
        score = self._string_similarity(pred_clean, exp_clean)
        is_correct = score > 0.85
        
        return {
            "correct": is_correct,
            "score": score,
            "confidence": "medium" if 0.7 < score < 0.85 else ("low" if score < 0.7 else "high"),
            "reason": f"Normalized form similarity: {score*100:.0f}%",
            "predicted_clean": pred_clean,
            "expected_clean": exp_clean,
        }
    
    def _llm_judge(self, predicted: str, expected: str, question: str = "", category: str = "") -> dict:
        """Use LLM as a judge to score correctness."""
        if self.judge_llm is None:
            raise ValueError("No judge LLM available")
        
        prompt = f"""You are evaluating whether a predicted answer is correct.

Question: {question}
Category: {category}

Expected answer (may have multiple valid forms): {expected}
Predicted answer: {predicted}

Score this answer's correctness from 0-100:
- 100: Completely correct, exact or valid alternative form
- 75-99: Correct with minor differences (wording, rounding)
- 50-74: Partially correct, missing details or has minor errors
- 1-49: Significantly incorrect, wrong core value
- 0: Completely wrong or nonsensical

Reply with ONLY:
SCORE: <number>
REASON: <brief explanation>"""

        try:
            response = self.judge_llm.call(prompt)
            
            # Parse score
            score_match = re.search(r'SCORE:\s*(\d+)', response)
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response)
            
            score = int(score_match.group(1)) / 100.0 if score_match else 0.5
            reason = reason_match.group(1).strip() if reason_match else "LLM judge evaluation"
            
            is_correct = score >= 0.75
            confidence = "high" if score >= 0.9 else ("medium" if score >= 0.7 else "low")
            
            return {
                "correct": is_correct,
                "score": score,
                "confidence": confidence,
                "reason": reason,
                "predicted_clean": predicted[:100],
                "expected_clean": expected[:100],
            }
        except Exception as e:
            print(f"⚠️  Judge evaluation failed: {e}")
            return {
                "correct": False,
                "score": 0.0,
                "confidence": "low",
                "reason": f"Judge error: {str(e)[:50]}",
                "predicted_clean": predicted[:100],
                "expected_clean": expected[:100],
            }
    
    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer form for comparison.
        
        Handles:
        - Case insensitivity
        - Whitespace normalization
        - Common alternative phrasings ("has" vs ":")
        - Plural/singular variations
        """
        # Lowercase and strip
        norm = answer.lower().strip()
        
        # Normalize common separators: ":" or "has" → space
        norm = re.sub(r'[:\s]+', ' ', norm)
        norm = re.sub(r'\s+has\s+', ' ', norm)
        
        # Remove common non-essential words
        stopwords = ['the', 'a', 'an', 'and', 'or', 'is', 'are', 'be']
        words = norm.split()
        words = [w for w in words if w not in stopwords and len(w) > 1]
        norm = ' '.join(words)
        
        # Remove extra spaces
        norm = ' '.join(norm.split())
        
        return norm
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Compute simple string similarity (0.0-1.0)."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Levenshtein-like distance (simplified)
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        # Count matching characters in order
        matches = 0
        j = 0
        for i in range(len(longer)):
            if j < len(shorter) and longer[i] == shorter[j]:
                matches += 1
                j += 1
        
        return matches / len(longer)
    
    def get_stats(self) -> dict:
        """Return evaluator statistics."""
        if self.judge_llm:
            judge_stats = self.judge_llm.get_stats()
        else:
            judge_stats = {}
        
        return {
            "evaluations_run": self.evaluation_count,
            "judge_llm_stats": judge_stats,
        }
