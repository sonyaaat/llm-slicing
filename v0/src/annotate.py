"""v0 — Phase 1: Annotation of MedQA-500 with the human-designed taxonomy (taxonomy_v0.json).

This is INDEPENDENT of the v2 (automated HiBug2) taxonomy: it neither reads nor
reuses any v2 result. It only consumes the shared, taxonomy-agnostic input
`data/medqa_500.jsonl`. The error label M is NOT used here — annotation is blind
to model correctness (it is joined in later, in the analysis phase).

For each question:
  - deterministic attributes (no LLM):
      * usmle_step           -> mapped from the `meta_info` field
      * vignette_length_level -> binned from question word count
  - semantic attributes (gpt-4o, JSON mode, temperature 0): every remaining
    taxonomy attribute; the model must pick values only from the closed
    vocabulary, using "unclear" when uncertain.

Each returned value is validated against taxonomy_v0.json; out-of-vocabulary
values are replaced with "unclear" (categorical) or null (boolean) and counted.

Usage:
    python v0/src/annotate.py                 # full run on all 500
    python v0/src/annotate.py --limit 5       # dry run on first 5

Outputs (all under v0/):
    v0/results/annotations_v0.jsonl
    v0/results/annotation_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --- Paths (v0 is self-contained; only the input dataset is shared) ---------- #
V0_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = V0_DIR.parent
INPUT_PATH = PROJECT_ROOT / "data" / "medqa_500.jsonl"
TAXONOMY_PATH = V0_DIR / "taxonomy_v0.json"
OUTPUT_PATH = V0_DIR / "results" / "annotations_v0.jsonl"
SUMMARY_PATH = V0_DIR / "results" / "annotation_summary.json"
ENV_PATH = PROJECT_ROOT / ".env"

# --- Model config ------------------------------------------------------------ #
MODEL = "gpt-4o"
TEMPERATURE = 0
MAX_TOKENS = 500
MAX_RETRIES = 4

# --- Deterministic attributes (handled in code, not by the LLM) -------------- #
DETERMINISTIC = {"usmle_step", "vignette_length_level"}
# Word-count thresholds for vignette_length_level (short / medium / long).
LEN_SHORT_MAX = 60
LEN_MEDIUM_MAX = 120


def load_taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text())


def map_usmle_step(meta_info: str) -> str:
    """Map the dataset `meta_info` field to the taxonomy usmle_step vocabulary."""
    if meta_info == "step1":
        return "step1"
    if meta_info in ("step2&3", "step2_3", "step2", "step3"):
        return "step2_3"
    return "step1"  # dataset only contains step1 / step2&3


def bin_vignette_length(question: str) -> tuple[str, int]:
    """Bin the question by word count. Note: 'no_vignette' is NOT auto-detected
    (it needs semantics); see the report limitations."""
    wc = len(question.split())
    if wc <= LEN_SHORT_MAX:
        level = "short"
    elif wc <= LEN_MEDIUM_MAX:
        level = "medium"
    else:
        level = "long"
    return level, wc


def build_llm_attributes(taxonomy: dict) -> dict:
    """Return {attr: allowed_values_or_'boolean'} for LLM-annotated attributes only."""
    return {k: v for k, v in taxonomy.items() if k not in DETERMINISTIC}


def format_question(q: dict) -> str:
    opts = "\n".join(f"{k}. {q['options'][k]}" for k in sorted(q["options"]))
    return f"{q['question']}\n\nOptions:\n{opts}\nCorrect answer: {q['answer_idx']}"


def build_prompt(q: dict, llm_attrs: dict) -> str:
    lines = []
    for attr, spec in llm_attrs.items():
        if spec == "boolean":
            lines.append(f"- {attr}: true or false")
        else:
            lines.append(f"- {attr}: one of {spec}")
    attr_block = "\n".join(lines)
    keys = ", ".join(f'"{a}"' for a in llm_attrs)
    return (
        "Annotate the following USMLE question along a FIXED taxonomy.\n"
        "For each attribute, choose EXACTLY ONE value from its allowed list "
        "(do not invent values). Use \"unclear\" when a categorical attribute is "
        "genuinely uncertain or not applicable. Boolean attributes must be true "
        "or false.\n\n"
        f"Question:\n{format_question(q)}\n\n"
        f"Attributes and allowed values:\n{attr_block}\n\n"
        "Before answering, verify every value is from the allowed list.\n"
        f"Return ONLY a JSON object with exactly these keys: {keys}."
    )


def call_llm(client: OpenAI, prompt: str) -> dict:
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content":
                        "You are a medical education expert annotating USMLE "
                        "questions along a fixed taxonomy. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001 - retry transient/parse errors
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {last_exc}")


def validate(raw: dict, llm_attrs: dict, invalid_counter: Counter) -> dict:
    """Coerce each LLM value to the closed vocabulary; log out-of-vocab values."""
    clean = {}
    for attr, spec in llm_attrs.items():
        val = raw.get(attr)
        if spec == "boolean":
            if isinstance(val, bool):
                clean[attr] = val
            else:
                clean[attr] = "unclear"
                invalid_counter[attr] += 1
        else:
            if isinstance(val, str) and val in spec:
                clean[attr] = val
            else:
                clean[attr] = "unclear" if "unclear" in spec else "unclear"
                invalid_counter[attr] += 1
    return clean


def load_done() -> set:
    done = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["orig_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="v0 Phase 1: annotate MedQA-500.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv(dotenv_path=ENV_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy()
    llm_attrs = build_llm_attributes(taxonomy)

    questions = [json.loads(l) for l in INPUT_PATH.read_text().splitlines() if l.strip()]
    if args.limit is not None:
        questions = questions[: args.limit]

    done = load_done()
    todo = [q for q in questions if q["orig_index"] not in done]
    print(f"Taxonomy: {len(taxonomy)} attributes "
          f"({len(DETERMINISTIC)} deterministic, {len(llm_attrs)} via LLM).")
    print(f"Questions: {len(questions)} total, {len(done)} done, {len(todo)} to annotate.")

    client = OpenAI()
    invalid_counter: Counter = Counter()

    with open(OUTPUT_PATH, "a") as out:
        for q in tqdm(todo, desc="Annotate", unit="q"):
            level, wc = bin_vignette_length(q["question"])
            record = {"orig_index": q["orig_index"],
                      "usmle_step": map_usmle_step(q.get("meta_info", "")),
                      "vignette_length_level": level,
                      "word_count": wc}
            raw = call_llm(client, build_prompt(q, llm_attrs))
            record.update(validate(raw, llm_attrs, invalid_counter))
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    # ---- Summary over the full output file ---- #
    rows = [json.loads(l) for l in OUTPUT_PATH.read_text().splitlines() if l.strip()]
    dist: dict = defaultdict(Counter)
    for r in rows:
        for attr in taxonomy:
            dist[attr][str(r.get(attr))] += 1

    summary = {
        "n_annotated": len(rows),
        "invalid_replaced": dict(invalid_counter),
        "distributions": {a: dict(c) for a, c in dist.items()},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nAnnotated {len(rows)} questions.")
    print(f"Invalid values replaced with 'unclear': {sum(invalid_counter.values())} "
          f"{dict(invalid_counter) if invalid_counter else ''}")
    print("\nDistribution preview:")
    for attr in taxonomy:
        top = dist[attr].most_common(4)
        preview = " | ".join(f"{v}:{n}" for v, n in top)
        print(f"  {attr}: {preview}")
    print(f"\nSaved -> {OUTPUT_PATH}")
    print(f"Saved -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
