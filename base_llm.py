import os
import time
import re
from groq import Groq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

EVAL_MODEL = "llama-3.1-8b-instant"
SMART_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL = "llama-3.1-8b-instant"
ACTIVE_MODEL = SMART_MODEL

TEMPERATURE = 0.7
JUDGE_TEMP = 0.3
MAX_TOKENS = 1024
JUDGE_TOKENS = 256
MAX_RETRIES = 6
BACKOFF_BASE = 2
MAX_WAIT_SECONDS = 90  # do not sleep longer than this on rate limits


class GroqQuotaExceeded(Exception):
    """Daily token quota (TPD) or non-recoverable rate limit."""

    def __init__(self, message: str, retry_after_s: int | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class BaseLLM:
    def __init__(self, api_key: str, model: str = ACTIVE_MODEL, temperature: float = TEMPERATURE):
        self.client = Groq(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.call_count = 0
        self.token_count = 0
        self.raw_outputs = []

    def call(self, prompt: str, max_retries: int = MAX_RETRIES, model: str | None = None) -> str:
        use_model = model or self.model_name
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=use_model,
                    temperature=self.temperature,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.call_count += 1
                self.token_count += response.usage.total_tokens

                output = response.choices[0].message.content.strip()
                self.raw_outputs.append({
                    "model": use_model,
                    "prompt": prompt[:200],
                    "output": output[:500],
                    "tokens": response.usage.total_tokens,
                })
                return output

            except Exception as e:
                err = str(e)

                if "401" in err or "authentication" in err.lower():
                    print("\n  AUTH ERROR: Missing or invalid GROQ_API_KEY in .env")
                    raise

                if self._is_daily_quota(err):
                    wait = self._parse_wait(err)
                    msg = (
                        "Groq daily token limit (TPD) reached. "
                        f"Quota resets after ~{wait}s. "
                        "Re-run with: --no-mcts --resume  OR wait and use --resume. "
                        "See https://console.groq.com/settings/billing"
                    )
                    raise GroqQuotaExceeded(msg, retry_after_s=wait) from e

                wait = self._parse_wait(err)
                if wait and attempt < max_retries - 1:
                    wait = min(wait, MAX_WAIT_SECONDS)
                    print(
                        f"\n  Rate limited ({wait}s wait, "
                        f"attempt {attempt + 1}/{max_retries})...",
                        flush=True,
                    )
                    time.sleep(wait)
                elif attempt < max_retries - 1:
                    backoff = min(BACKOFF_BASE ** attempt, MAX_WAIT_SECONDS)
                    print(f"\n  Error (attempt {attempt + 1}/{max_retries}): {err[:80]}")
                    print(f"  Retrying in {backoff}s...", flush=True)
                    time.sleep(backoff)
                else:
                    print(f"\n  All {max_retries} attempts failed: {err[:120]}")
                    raise

    @staticmethod
    def _is_daily_quota(error_msg: str) -> bool:
        lower = error_msg.lower()
        return (
            "tokens per day" in lower
            or "tpd" in lower
            or ("limit" in lower and "used" in lower and "requested" in lower)
        )

    def _parse_wait(self, error_msg: str) -> int | None:
        match = re.search(
            r"try again in\s*(?:(\d+)m)?(\d+(?:\.\d+)?)s",
            error_msg,
            re.IGNORECASE,
        )
        if match:
            mins = int(match.group(1) or 0)
            secs = float(match.group(2) or 0)
            return int(mins * 60 + secs) + 2

        if "429" in error_msg or "rate_limit" in error_msg.lower():
            return 30
        return None

    def reset_stats(self):
        self.call_count = 0
        self.token_count = 0
        self.raw_outputs = []

    def get_stats(self) -> dict:
        return {
            "llm_calls": self.call_count,
            "total_tokens": self.token_count,
            "model": self.model_name,
        }

    def get_raw_outputs(self) -> list:
        return self.raw_outputs
