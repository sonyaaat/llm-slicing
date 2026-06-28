"""Step 1 — Target LLM Inference.

Runs the target model (gpt-4o-mini) on the 500-question MedQA sample using a
chain-of-thought prompt, then assigns each question a binary error label M
(0 = model correct, 1 = model incorrect) by comparing the predicted option
letter to the ground-truth answer_idx.

Usage:
    python src/run_inference.py                # full run on all 500 questions
    python src/run_inference.py --limit 5      # quick dry run on first 5

Behaviour:
    - temperature=0 for reproducibility
    - saves each result immediately to results/inference_results.jsonl
    - resumable: skips questions whose orig_index is already in the output file
    - prints final accuracy and error count

Outputs:
    results/inference_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config import OPENAI_API_KEY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "medqa_500.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "inference_results.jsonl"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 512  # room for short reasoning + final answer
MAX_RETRIES = 4

SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE multiple-choice questions. "
    "Think step by step briefly, then give your final answer. "
    "End your response with a line in exactly this format: 'Answer: X' "
    "where X is one of A, B, C, or D."
)


def build_user_prompt(question: str, options: dict) -> str:
    opts = "\n".join(f"{letter}. {options[letter]}" for letter in sorted(options))
    return f"{question}\n\nOptions:\n{opts}\n\nReason briefly, then end with 'Answer: X'."


def parse_answer_letter(text: str) -> str | None:
    """Extract the predicted option letter (A-D) from the model's response."""
    if not text:
        return None
    # Preferred: explicit 'Answer: X' (take the last occurrence).
    matches = re.findall(r"answer\s*[:\-]?\s*\(?([ABCD])\)?", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Fallback: a standalone letter near the end of the text.
    tail_matches = re.findall(r"\b([ABCD])\b", text)
    if tail_matches:
        return tail_matches[-1].upper()
    return None


def call_model(client: OpenAI, question: str, options: dict) -> str:
    """Call the API with simple exponential-backoff retries."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, options)},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - retry on any transient API error
            last_exc = exc
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {MAX_RETRIES} retries: {last_exc}")


def load_done_indices() -> set:
    """Return orig_index values already present in the output file (for resuming)."""
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["orig_index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Run target LLM inference (Step 1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N questions (for a dry run).")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        questions = questions[: args.limit]

    done = load_done_indices()
    todo = [q for q in questions if q["orig_index"] not in done]
    print(f"Loaded {len(questions)} questions; {len(done)} already done; {len(todo)} to process.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(OUTPUT_PATH, "a") as out:
        for q in tqdm(todo, desc="Inference", unit="q"):
            raw = call_model(client, q["question"], q["options"])
            pred = parse_answer_letter(raw)
            gold = q["answer_idx"]
            # M = 1 means the model erred (including unparseable answers).
            M = 0 if (pred is not None and pred == gold) else 1
            record = {
                "orig_index": q["orig_index"],
                "predicted": pred,
                "gold": gold,
                "M": M,
                "meta_info": q.get("meta_info"),
                "raw_response": raw,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    # Summary over the full output file.
    all_results = []
    with open(OUTPUT_PATH) as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    n = len(all_results)
    errors = sum(r["M"] for r in all_results)
    correct = n - errors
    unparsed = sum(1 for r in all_results if r["predicted"] is None)
    acc = correct / n if n else 0.0

    print("\n--- Inference summary ---")
    print(f"Total answered:  {n}")
    print(f"Correct (M=0):   {correct}")
    print(f"Incorrect (M=1): {errors}")
    print(f"Unparseable:     {unparsed}")
    print(f"Accuracy:        {acc:.3f} ({acc*100:.1f}%)")
    print(f"Results saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
