"""v0 — Phase 2: Validity of the annotation (no error label M used).

Checks the QUALITY of the Phase-1 annotation before any correlation analysis:
  1. Coverage        — share of non-"unclear" values per attribute.
  2. Distribution    — value counts, dead (0) / rare (<10) values, dominance skew.
  3. Reliability     — re-annotate a fixed subset with GPT-4o and compute
                       self-agreement (Cohen's kappa + raw agreement) per attribute.

Then it assigns a rule-based verdict per attribute (keep / weak / drop), writes a
cleaned taxonomy, a markdown + JSON report, and four figures.

Independent of v2. Uses only v0 artifacts + the shared dataset.

Usage:
    python v0/src/validate.py                 # full (re-annotates 50 for reliability)
    python v0/src/validate.py --reliability-n 0   # skip reliability re-annotation

Outputs (under v0/):
    results/validity_report.md
    results/validity_report.json
    results/taxonomy_v0_cleaned.json
    results/figures/{coverage,skew,value_distributions,reliability_kappa}.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

import annotate  # sibling module (v0/src/annotate.py)

V0_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = V0_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
ANNOTATIONS_PATH = V0_DIR / "results" / "annotations_v0.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "medqa_500.jsonl"
REPORT_MD = V0_DIR / "results" / "validity_report.md"
REPORT_JSON = V0_DIR / "results" / "validity_report.json"
CLEANED_TAXONOMY = V0_DIR / "results" / "taxonomy_v0_cleaned.json"
FIG_DIR = V0_DIR / "results" / "figures"

RELIABILITY_N = 50
RARE_THRESHOLD = 10          # values used fewer than this are "rare"
SKEW_DROP = 0.95             # top value >= this share -> degenerate -> drop
SKEW_WEAK = 0.85             # top value >= this share -> weak
KAPPA_WEAK = 0.40            # below this -> unreliable
BOOL_MIN_MINORITY = 0.02     # boolean minority class below this -> drop

# --- Palette (from the validated reference, light surface) ------------------- #
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

VERDICT_COLOR = {"keep": GOOD, "weak": WARNING, "drop": CRITICAL}


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_rows() -> list:
    return [json.loads(l) for l in ANNOTATIONS_PATH.read_text().splitlines() if l.strip()]


def is_unclear(v) -> bool:
    return v in ("unclear", None, "None")


# --------------------------------------------------------------------------- #
# Reliability: re-annotate a fixed subset and compare
# --------------------------------------------------------------------------- #
def reliability(rows: list, taxonomy: dict, llm_attrs: dict, n: int) -> dict:
    if n <= 0:
        return {}
    questions = {json.loads(l)["orig_index"]: json.loads(l)
                 for l in QUESTIONS_PATH.read_text().splitlines() if l.strip()}
    subset = rows[:n]
    client = OpenAI()
    inv: Counter = Counter()

    run1, run2 = {a: [] for a in taxonomy}, {a: [] for a in taxonomy}
    for r in tqdm(subset, desc="Reliability re-annotate", unit="q"):
        q = questions[r["orig_index"]]
        raw = annotate.call_llm(client, annotate.build_prompt(q, llm_attrs))
        clean = annotate.validate(raw, llm_attrs, inv)
        # deterministic re-computation
        level, _ = annotate.bin_vignette_length(q["question"])
        clean["usmle_step"] = annotate.map_usmle_step(q.get("meta_info", ""))
        clean["vignette_length_level"] = level
        for a in taxonomy:
            run1[a].append(str(r.get(a)))
            run2[a].append(str(clean.get(a)))

    out = {}
    for a in taxonomy:
        agree = sum(x == y for x, y in zip(run1[a], run2[a])) / len(subset)
        if a in annotate.DETERMINISTIC:
            out[a] = {"kappa": 1.0, "agreement": 1.0, "note": "deterministic"}
            continue
        try:
            k = cohen_kappa_score(run1[a], run2[a])
            if k != k:  # nan (single category)
                k = 1.0 if agree == 1.0 else 0.0
        except Exception:
            k = 1.0 if agree == 1.0 else 0.0
        out[a] = {"kappa": round(float(k), 3), "agreement": round(agree, 3), "note": ""}
    return out


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def verdict_for(attr, spec, dist, coverage, kappa) -> tuple[str, str]:
    total = sum(dist.values())
    non_unclear = {k: v for k, v in dist.items() if not is_unclear(k)}
    top_share = (max(non_unclear.values()) / total) if non_unclear else 0.0

    if spec == "boolean":
        minority = min(dist.get("True", 0), dist.get("False", 0)) / total
        if minority < BOOL_MIN_MINORITY:
            return "drop", f"minority class {minority*100:.1f}% (too rare to slice)"
    else:
        if top_share >= SKEW_DROP:
            return "drop", f"degenerate: top value {top_share*100:.0f}%"

    reasons = []
    if coverage < 0.80:
        reasons.append(f"low coverage {coverage*100:.0f}%")
    if top_share >= SKEW_WEAK:
        reasons.append(f"skewed: top value {top_share*100:.0f}%")
    if kappa is not None and kappa < KAPPA_WEAK:
        reasons.append(f"unreliable κ={kappa:.2f}")
    if reasons:
        return "weak", "; ".join(reasons)
    return "keep", "good coverage, spread and reliability"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_coverage(stats):
    attrs = list(stats)
    vals = [stats[a]["coverage"] * 100 for a in attrs]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = range(len(attrs))
    ax.barh(list(y), vals, color=BLUE, height=0.62, zorder=3)
    ax.axvline(80, color=CRITICAL, lw=1.5, ls="--", zorder=2)
    ax.text(80, len(attrs) - 0.3, " 80% target", color=CRITICAL, fontsize=8, va="center")
    for i, v in enumerate(vals):
        ax.text(v - 1, i, f"{v:.0f}%", va="center", ha="right", color="white", fontsize=7.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(attrs, fontsize=8)
    ax.set_xlim(0, 105); ax.invert_yaxis()
    ax.set_title("Coverage per attribute (% non-'unclear')", color=INK, fontsize=11, loc="left")
    ax.xaxis.grid(True, color=GRID, zorder=0); _style_ax(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "coverage.png"); plt.close(fig)


def fig_skew(stats):
    attrs = sorted(stats, key=lambda a: stats[a]["top_share"], reverse=True)
    vals = [stats[a]["top_share"] * 100 for a in attrs]
    colors = [VERDICT_COLOR[stats[a]["verdict"]] for a in attrs]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = range(len(attrs))
    ax.barh(list(y), vals, color=colors, height=0.62, zorder=3)
    ax.axvline(90, color=INK2, lw=1.2, ls="--", zorder=2)
    ax.text(90, -0.5, "90% degenerate", color=INK2, fontsize=8, va="center")
    for i, (a, v) in enumerate(zip(attrs, vals)):
        ax.text(v + 1, i, f"{v:.0f}%  ({stats[a]['verdict']})", va="center", ha="left",
                color=INK2, fontsize=7.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(attrs, fontsize=8)
    ax.set_xlim(0, 120); ax.invert_yaxis()
    ax.set_title("Dominance skew — share of most-frequent value\n(color = verdict: green keep / amber weak / red drop)",
                 color=INK, fontsize=11, loc="left")
    ax.xaxis.grid(True, color=GRID, zorder=0); _style_ax(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "skew.png"); plt.close(fig)


def fig_distributions(stats, taxonomy):
    attrs = list(taxonomy)
    ncol, nrow = 4, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 13), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for idx, a in enumerate(attrs):
        ax = axes[idx // ncol][idx % ncol]
        dist = stats[a]["dist"]
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        yy = range(len(labels))
        ax.barh(list(yy), vals, color=BLUE, height=0.7, zorder=3)
        ax.set_yticks(list(yy)); ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(a, color=INK, fontsize=9, loc="left")
        ax.xaxis.grid(True, color=GRID, zorder=0); _style_ax(ax)
    for j in range(len(attrs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Value distributions per attribute", color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_DIR / "value_distributions.png"); plt.close(fig)


def fig_reliability(stats, rel):
    attrs = [a for a in stats if a not in annotate.DETERMINISTIC]
    attrs = sorted(attrs, key=lambda a: rel[a]["kappa"])
    vals = [rel[a]["kappa"] for a in attrs]

    def kcolor(k):
        if k < 0.40: return CRITICAL
        if k < 0.60: return WARNING
        return GOOD
    colors = [kcolor(v) for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = range(len(attrs))
    ax.barh(list(y), vals, color=colors, height=0.62, zorder=3)
    for x0, lab in [(0.40, "weak"), (0.60, "good")]:
        ax.axvline(x0, color=MUTED, lw=1, ls="--", zorder=2)
        ax.text(x0, -0.5, lab, color=MUTED, fontsize=7.5, ha="center")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", ha="left", color=INK2, fontsize=7.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(attrs, fontsize=8)
    ax.set_xlim(0, 1.05); ax.invert_yaxis()
    ax.set_title("Reliability — self-agreement (Cohen's κ, 2 runs on 50 questions)",
                 color=INK, fontsize=11, loc="left")
    ax.xaxis.grid(True, color=GRID, zorder=0); _style_ax(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "reliability_kappa.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="v0 Phase 2: validity checks.")
    parser.add_argument("--reliability-n", type=int, default=RELIABILITY_N)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy = annotate.load_taxonomy()
    llm_attrs = annotate.build_llm_attributes(taxonomy)
    rows = load_rows()
    n = len(rows)

    rel = reliability(rows, taxonomy, llm_attrs, args.reliability_n)

    stats = {}
    for attr, spec in taxonomy.items():
        dist = Counter(str(r.get(attr)) for r in rows)
        non_unclear = {k: v for k, v in dist.items() if not is_unclear(k)}
        coverage = sum(non_unclear.values()) / n
        top_share = (max(non_unclear.values()) / n) if non_unclear else 0.0
        allowed = ([] if spec == "boolean" else spec)
        used = set(non_unclear)
        dead = [v for v in allowed if v not in used and v != "unclear"]
        rare = {k: v for k, v in non_unclear.items() if v < RARE_THRESHOLD}
        kappa = rel[attr]["kappa"] if rel else None
        vdt, reason = verdict_for(attr, spec, dist, coverage, kappa)
        stats[attr] = {
            "type": "boolean" if spec == "boolean" else "categorical",
            "coverage": round(coverage, 3),
            "top_value": (max(non_unclear, key=non_unclear.get) if non_unclear else None),
            "top_share": round(top_share, 3),
            "dead_values": dead,
            "rare_values": rare,
            "kappa": kappa,
            "agreement": rel[attr]["agreement"] if rel else None,
            "verdict": vdt,
            "reason": reason,
            "dist": dict(dist),
        }

    # cleaned taxonomy: drop 'drop' attributes; strip dead values from the rest
    cleaned = {}
    for attr, spec in taxonomy.items():
        if stats[attr]["verdict"] == "drop":
            continue
        if spec == "boolean":
            cleaned[attr] = "boolean"
        else:
            cleaned[attr] = [v for v in spec if v not in stats[attr]["dead_values"]]
    CLEANED_TAXONOMY.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))

    # figures
    fig_coverage(stats); fig_skew(stats); fig_distributions(stats, taxonomy)
    if rel:
        fig_reliability(stats, rel)

    # JSON report
    report = {"n_annotated": n, "reliability_n": args.reliability_n,
              "attributes": {a: {k: v for k, v in s.items() if k != "dist"}
                             for a, s in stats.items()}}
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Markdown report
    lines = ["# v0 — Phase 2: Validity Report\n",
             f"Annotated questions: **{n}**. Reliability subset: **{args.reliability_n}** "
             "(re-annotated independently).\n",
             "## Verdict per attribute\n",
             "| attribute | type | coverage | top value (share) | dead values | κ | verdict | reason |",
             "|---|---|---|---|---|---|---|---|"]
    order = {"keep": 0, "weak": 1, "drop": 2}
    for a in sorted(stats, key=lambda a: order[stats[a]["verdict"]]):
        s = stats[a]
        kd = "—" if s["kappa"] is None else (f"{s['kappa']:.2f}" +
             (" (det.)" if a in annotate.DETERMINISTIC else ""))
        dead = ", ".join(s["dead_values"]) or "—"
        lines.append(f"| {a} | {s['type']} | {s['coverage']*100:.0f}% | "
                     f"{s['top_value']} ({s['top_share']*100:.0f}%) | {dead} | {kd} | "
                     f"{s['verdict']} | {s['reason']} |")
    counts = Counter(s["verdict"] for s in stats.values())
    lines += ["\n## Summary\n",
              f"- ✅ keep: **{counts['keep']}**   ⚠️ weak: **{counts['weak']}**   "
              f"❌ drop: **{counts['drop']}**",
              f"- Cleaned taxonomy written to `taxonomy_v0_cleaned.json` "
              f"({len([a for a in stats if stats[a]['verdict']!='drop'])} attributes).",
              "\n## Figures\n",
              "- `figures/coverage.png` — coverage per attribute",
              "- `figures/skew.png` — dominance skew (share of top value), colored by verdict",
              "- `figures/value_distributions.png` — value distribution per attribute",
              "- `figures/reliability_kappa.png` — self-agreement (Cohen's κ)",
              "\n## Caveats\n",
              "- **Reliability is measured at temperature 0**, so both annotation "
              "passes are near-deterministic; the high κ mainly reflects the "
              "model's determinism, not genuine robustness. A stricter estimate "
              "would re-annotate at temperature > 0 or against human labels.",
              "- Annotation is LLM-based (no human gold set yet); κ here is "
              "model-vs-model self-agreement, not model-vs-human.",
              "- Verdicts use fixed thresholds (drop ≥95% dominance / boolean "
              "minority <2%; weak ≥85% or κ<0.40) — reasonable but adjustable."]
    REPORT_MD.write_text("\n".join(lines))

    # console
    print(f"\n=== Phase 2 validity ({n} questions) ===")
    print(f"{'attribute':<42}{'cov':>5}{'topShare':>10}{'κ':>7}  verdict")
    for a in sorted(stats, key=lambda a: order[stats[a]["verdict"]]):
        s = stats[a]
        kd = "  —  " if s["kappa"] is None else f"{s['kappa']:.2f}"
        print(f"{a:<42}{s['coverage']*100:>4.0f}%{s['top_share']*100:>9.0f}%{kd:>7}  {s['verdict']}")
    print(f"\nkeep {counts['keep']} | weak {counts['weak']} | drop {counts['drop']}")
    print(f"Saved report -> {REPORT_MD}")
    print(f"Saved figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
