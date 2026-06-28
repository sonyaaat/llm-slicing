"""Download the MedQA-USMLE 4-option test split, save it locally, and draw a 500-question sample.

Usage:
    python src/load_dataset.py

Outputs:
    data/medqa_test_full.jsonl   full test split
    data/medqa_500.jsonl         500-question sample (random_state=42)
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULL_PATH = DATA_DIR / "medqa_test_full.jsonl"
SAMPLE_PATH = DATA_DIR / "medqa_500.jsonl"

SAMPLE_SIZE = 500
RANDOM_STATE = 42


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GBaker/MedQA-USMLE-4-options (test split) ...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

    df = ds.to_pandas()

    # Preserve the original index as an explicit, stable question ID.
    df = df.reset_index().rename(columns={"index": "orig_index"})

    total = len(df)
    print(f"Total questions loaded: {total}")

    # Save the full test split.
    df.to_json(FULL_PATH, orient="records", lines=True, force_ascii=False)
    print(f"Saved full test set -> {FULL_PATH}")

    # Sample exactly 500 questions, no stratification/filtering, reproducible.
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    sample.to_json(SAMPLE_PATH, orient="records", lines=True, force_ascii=False)
    print(f"Saved sample of {len(sample)} questions -> {SAMPLE_PATH}")

    # Print one full example so the data can be visually confirmed.
    example = sample.iloc[0].to_dict()
    print("\n--- Example record (first row of sample) ---")
    print(f"orig_index: {example.get('orig_index')}")
    print(f"question:   {example.get('question')}")
    print(f"options:    {example.get('options')}")
    print(f"answer:     {example.get('answer')}")
    print(f"answer_idx: {example.get('answer_idx')}")
    print("--- end example ---")


if __name__ == "__main__":
    main()
