"""One tiny API call to see if Groq daily quota has reset."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from base_llm import BaseLLM, EVAL_MODEL, GroqQuotaExceeded


def main():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("FAIL: GROQ_API_KEY missing in .env")
        return 1

    print("Testing Groq with smallest model (llama-3.1-8b-instant)...")
    llm = BaseLLM(api_key=key, model=EVAL_MODEL)
    try:
        out = llm.call("Reply with exactly: OK", max_retries=1)
        print(f"SUCCESS: API works. Response: {out[:40]}")
        print(f"Tokens used: {llm.token_count}")
        print("\nYou can run:")
        print("  python eval/runner_improved.py --limit 10 --no-mcts --fast --resume")
        return 0
    except GroqQuotaExceeded as e:
        print(f"BLOCKED: {e}")
        print("\nWait for daily quota to reset (often ~10 min or midnight UTC).")
        print("Check: https://console.groq.com")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
