"""Shared helpers for parsing LLM scores and normalizing answers."""

import re

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}


def parse_score(response: str, low: float = 1, high: float = 10) -> float:
    """Extract first numeric score in range [low, high] from LLM response."""
    for token in response.replace(",", ".").split():
        cleaned = "".join(c for c in token if c.isdigit() or c == ".")
        if not cleaned:
            continue
        try:
            score = float(cleaned)
            if low <= score <= high:
                return score
        except ValueError:
            continue
    return (low + high) / 2


def words_to_number(text: str) -> str | None:
    """Convert phrases like 'six hundred' to '600' when possible."""
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    if not words:
        return None

    total = 0
    current = 0
    matched = False

    for w in words:
        if w not in NUMBER_WORDS:
            continue
        matched = True
        val = NUMBER_WORDS[w]
        if val == 100:
            current = max(1, current) * 100
        elif val == 1000:
            current = max(1, current) * 1000
            total += current
            current = 0
        elif val >= 20:
            current += val
        else:
            current += val

    if not matched:
        return None

    total += current
    return str(int(total)) if total == int(total) else str(total)


def normalize_answer_for_clustering(answer: str) -> str:
    """Normalize answer for clustering across format variants."""
    normalized = answer.lower().strip()
    normalized = re.sub(r"[^\w.\-\s/]", "", normalized)

    word_num = words_to_number(normalized)
    if word_num is not None:
        return f"num:{word_num}"

    if "/" in normalized:
        parts = normalized.split("/")
        if len(parts) == 2:
            try:
                return f"num:{float(parts[0]) / float(parts[1])}"
            except ValueError:
                pass

    nums = re.findall(r"-?\d+\.?\d*", normalized)
    if nums:
        try:
            val = float(nums[0])
            return f"num:{val}"
        except ValueError:
            pass

    return " ".join(normalized.split())
