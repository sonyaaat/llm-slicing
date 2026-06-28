"""Step 2 — Taxonomy Generation (HiBug2-style contrastive, incremental).

For each of N independent runs:
  - draw random (error M=1, success M=0) question pairs
  - process pairs one by one, showing the analyzer (GPT-4o) the taxonomy
    accumulated so far; it either REUSES an existing attribute or proposes a
    NEW one
  - stop when PATIENCE consecutive pairs add nothing new (saturation), or at
    MAX_PAIRS

Then merge the N run taxonomies:
  - normalize synonymous attribute names into canonical clusters (GPT-4o)
  - count in how many distinct runs each canonical attribute appeared

All model-facing text (prompts, attribute names, definitions) is in English.

Usage:
    python src/generate_taxonomy.py                       # full: 5 runs
    python src/generate_taxonomy.py --runs 1 --max-pairs 3 --no-normalize  # cheap dry run

Outputs:
    results/taxonomy_run_1..N.json
    results/taxonomy_merged.json
    results/saturation_log.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from openai import OpenAI

from config import OPENAI_API_KEY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "medqa_500.jsonl"
INFERENCE_PATH = PROJECT_ROOT / "results" / "inference_results.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
MERGED_PATH = RESULTS_DIR / "taxonomy_merged.json"
SATURATION_PATH = RESULTS_DIR / "saturation_log.json"

MODEL = "gpt-4o"
GEN_TEMPERATURE = 0.3
NORM_TEMPERATURE = 0.0
MAX_TOKENS = 600
NORM_MAX_TOKENS = 4000  # normalization must emit many clusters in one JSON object
MAX_RETRIES = 4

DEFAULT_RUNS = 5
DEFAULT_MAX_PAIRS = 50
DEFAULT_PATIENCE = 6  # stop a run after this many consecutive pairs with no new attribute

PAIR_SYSTEM_PROMPT = (
    "You are analyzing why an LLM fails on USMLE medical multiple-choice "
    "questions. You compare a question the model got WRONG with one it got "
    "RIGHT and identify a characteristic OF THE QUESTION ITSELF (not of the "
    "answer) that could explain the difference. You return JSON only."
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data() -> tuple[list, list]:
    """Return (errors, successes): question dicts with M=1 and M=0 respectively."""
    with open(QUESTIONS_PATH) as f:
        questions = {json.loads(l)["orig_index"]: json.loads(l)
                     for l in f if l.strip()}
    errors, successes = [], []
    with open(INFERENCE_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = questions.get(rec["orig_index"])
            if q is None:
                continue
            (errors if rec["M"] == 1 else successes).append(q)
    return errors, successes


def format_question(q: dict) -> str:
    opts = "\n".join(f"{k}. {q['options'][k]}" for k in sorted(q["options"]))
    return f"{q['question']}\n\nOptions:\n{opts}\nCorrect answer: {q['answer_idx']}"


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def call_json(client: OpenAI, system: str, user: str, temperature: float,
              max_tokens: int = MAX_TOKENS) -> dict:
    """Call the model expecting a JSON object response, with retries."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001 - retry on transient/parse errors
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API/JSON call failed after {MAX_RETRIES} retries: {last_exc}")


def build_pair_prompt(q_err: dict, q_succ: dict, taxonomy: list) -> str:
    if taxonomy:
        tax_lines = "\n".join(
            f"- {a['name']} | {a['definition']} | {a['value_type']}" for a in taxonomy
        )
    else:
        tax_lines = "(empty)"
    return (
        "Question A — the model answered INCORRECTLY:\n"
        f"{format_question(q_err)}\n\n"
        "Question B — the model answered CORRECTLY:\n"
        f"{format_question(q_succ)}\n\n"
        "Current taxonomy of attributes (you may reuse these):\n"
        f"{tax_lines}\n\n"
        "Task:\n"
        "1. Identify ONE characteristic OF THE QUESTION ITSELF (not the answer) "
        "that could explain why A is harder/different than B.\n"
        "2. If an existing attribute already covers it, return its exact name in "
        "\"reused_attribute\".\n"
        "3. Otherwise propose a NEW attribute:\n"
        "   - binary attributes use the prefix \"is_\" (e.g. is_multi_step_reasoning)\n"
        "   - give a clear, observable definition\n"
        "   - before adding, self-check that it does not semantically overlap an "
        "existing attribute.\n"
        "Return ONLY JSON of the form:\n"
        '{"reused_attribute": "name_or_null", '
        '"new_attribute": {"name": "...", "definition": "...", "value_type": "binary"} or null}'
    )


# --------------------------------------------------------------------------- #
# One generation run
# --------------------------------------------------------------------------- #
def run_once(client: OpenAI, run_id: int, errors: list, successes: list,
             max_pairs: int, patience: int) -> tuple[list, dict]:
    rng = random.Random(run_id)
    errs = errors[:]
    succs = successes[:]
    rng.shuffle(errs)
    rng.shuffle(succs)
    n_pairs = min(max_pairs, len(errs), len(succs))

    taxonomy: list = []
    names_lower = set()
    no_new = 0
    history = []  # per-pair: number of attributes after that pair

    for i in range(n_pairs):
        result = call_json(
            client, PAIR_SYSTEM_PROMPT,
            build_pair_prompt(errs[i], succs[i], taxonomy),
            GEN_TEMPERATURE,
        )
        new_attr = result.get("new_attribute")
        added = False
        if isinstance(new_attr, dict) and new_attr.get("name"):
            name = str(new_attr["name"]).strip()
            if name.lower() not in names_lower:
                taxonomy.append({
                    "name": name,
                    "definition": str(new_attr.get("definition", "")).strip(),
                    "value_type": str(new_attr.get("value_type", "binary")).strip(),
                })
                names_lower.add(name.lower())
                added = True

        no_new = 0 if added else no_new + 1
        history.append({"pair": i + 1, "n_attributes": len(taxonomy), "added": added})
        if no_new >= patience:
            break

    stop_reason = ("saturated" if no_new >= patience else "max_pairs_reached")
    log = {
        "run_id": run_id,
        "pairs_processed": len(history),
        "n_attributes": len(taxonomy),
        "stop_reason": stop_reason,
        "history": history,
    }
    return taxonomy, log


# --------------------------------------------------------------------------- #
# Cross-run merge + normalization
# --------------------------------------------------------------------------- #
def normalize_and_merge(client: OpenAI, run_taxonomies: list, normalize: bool) -> list:
    """Cluster synonymous attributes, then count distinct runs per canonical attribute."""
    # Collect unique (name -> definition) across all runs, plus run membership per name.
    name_to_def = {}
    name_to_runs: dict = {}
    for run_id, tax in enumerate(run_taxonomies, start=1):
        for a in tax:
            name = a["name"]
            name_to_def.setdefault(name, a.get("definition", ""))
            name_to_runs.setdefault(name, set()).add(run_id)

    unique_names = sorted(name_to_def)

    # Default clustering: each name is its own cluster (used when --no-normalize).
    clusters = [{"canonical_name": n, "canonical_definition": name_to_def[n],
                 "value_type": "binary", "members": [n]} for n in unique_names]

    if normalize and len(unique_names) > 1:
        listing = "\n".join(f"- {n}: {name_to_def[n]}" for n in unique_names)
        prompt = (
            "Below is a list of attribute names with definitions, collected from "
            "several independent runs. Some may be synonyms (the same underlying "
            "concept named differently). Group semantically-equivalent attributes "
            "into clusters. Each cluster gets one canonical name (prefer the "
            "clearest is_-prefixed name) and a merged definition. Keep distinct "
            "concepts in separate clusters — do NOT over-merge.\n\n"
            f"Attributes:\n{listing}\n\n"
            "Return ONLY JSON of the form:\n"
            '{"clusters": [{"canonical_name": "...", "canonical_definition": "...", '
            '"value_type": "binary", "members": ["name1", "name2"]}]}'
        )
        result = call_json(client, "You normalize and deduplicate attribute taxonomies. "
                                    "Return JSON only.", prompt, NORM_TEMPERATURE,
                           max_tokens=NORM_MAX_TOKENS)
        got = result.get("clusters")
        if isinstance(got, list) and got:
            clusters = got

    # Count distinct runs each canonical attribute appeared in (union over members).
    merged = []
    for c in clusters:
        members = [m for m in c.get("members", []) if m in name_to_runs]
        runs = set()
        for m in members:
            runs |= name_to_runs[m]
        if not runs:
            continue
        merged.append({
            "name": c.get("canonical_name"),
            "definition": c.get("canonical_definition", ""),
            "value_type": c.get("value_type", "binary"),
            "members": members,
            "runs_appeared_in": sorted(runs),
            "count": len(runs),
        })
    merged.sort(key=lambda x: x["count"], reverse=True)
    return merged


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate error taxonomy (Step 2).")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip the cross-run synonym normalization step.")
    parser.add_argument("--stability-threshold", type=int, default=3,
                        help="Preview which attributes survive Step 3 (count >= threshold).")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip generation; merge existing taxonomy_run_*.json files.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)

    run_taxonomies = []
    if args.merge_only:
        run_files = sorted(RESULTS_DIR.glob("taxonomy_run_*.json"))
        for rf in run_files:
            run_taxonomies.append(json.loads(rf.read_text()))
        print(f"Merge-only: loaded {len(run_taxonomies)} existing run files.")
    else:
        errors, successes = load_data()
        print(f"Loaded {len(errors)} error (M=1) and {len(successes)} success (M=0) questions.\n")
        logs = []
        for run_id in range(1, args.runs + 1):
            run_path = RESULTS_DIR / f"taxonomy_run_{run_id}.json"
            tax, log = run_once(client, run_id, errors, successes,
                                args.max_pairs, args.patience)
            run_path.write_text(json.dumps(tax, indent=2, ensure_ascii=False))
            run_taxonomies.append(tax)
            logs.append(log)
            print(f"Run {run_id}: {len(tax)} attributes "
                  f"(stopped after {log['pairs_processed']} pairs — {log['stop_reason']})")
        SATURATION_PATH.write_text(json.dumps(logs, indent=2, ensure_ascii=False))

    merged = normalize_and_merge(client, run_taxonomies, normalize=not args.no_normalize)
    MERGED_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    raw_unique = len({a["name"] for tax in run_taxonomies for a in tax})
    print(f"\nUnique attributes before normalization: {raw_unique}")
    print(f"Unique attributes after normalization:  {len(merged)}")

    print("\n--- Merged taxonomy (sorted by stability) ---")
    print(f"{'count':>5}  {'attribute':<34} runs")
    for a in merged:
        print(f"{a['count']:>5}  {a['name']:<34} {a['runs_appeared_in']}")

    survive = [a for a in merged if a["count"] >= args.stability_threshold]
    print(f"\n>>> Will SURVIVE Step 3 (count >= {args.stability_threshold}): {len(survive)} attributes")
    print(f">>> Will be DISCARDED as unstable:    {len(merged) - len(survive)} attributes")
    print(f"\nSaved -> {MERGED_PATH}")


if __name__ == "__main__":
    main()
