"""v0 — Phase 3: Usefulness analysis (which attributes relate to model errors).

Joins the Phase-1 annotations with the error label M (from Step 1 inference) and,
for every attribute in the cleaned taxonomy, measures how strongly it is
associated with errors:

  - error rate per value (+ lift vs baseline, low-support flag)
  - NMI(attribute, M)               — strength of association
  - permutation test                — p-value (raw, NO multiple-comparison correction)
  - Cramer's V                      — effect size for categorical association

Then it assigns a verdict per attribute (useful / not useful) and writes tables,
figures, and a report. No LLM calls — pure statistics on existing data.

Independent of v2. Uses only v0 annotations + the shared inference labels.

Usage:
    python v0/src/analyze.py

Outputs (under v0/):
    results/usefulness_report.md
    results/usefulness_report.json
    results/figures/{attribute_importance,error_rates}.png
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score

V0_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = V0_DIR.parent
ANNOTATIONS_PATH = V0_DIR / "results" / "annotations_v0.jsonl"
CLEANED_TAXONOMY = V0_DIR / "results" / "taxonomy_v0_cleaned.json"
INFERENCE_PATH = PROJECT_ROOT / "results" / "inference_results.jsonl"
REPORT_MD = V0_DIR / "results" / "usefulness_report.md"
REPORT_JSON = V0_DIR / "results" / "usefulness_report.json"
FIG_DIR = V0_DIR / "results" / "figures"

N_PERM = 2000
P_THRESHOLD = 0.05
MIN_SUPPORT = 10        # values with fewer questions are flagged low-support
V_WEAK, V_SMALL, V_MODERATE = 0.10, 0.20, 0.30

# --- Palette (light surface) ---------------------------------------------- #
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, GOOD, WARNING, CRITICAL = "#2a78d6", "#0ca30c", "#fab219", "#d03b3b"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def effect_label(v: float) -> str:
    if v < V_WEAK: return "negligible"
    if v < V_SMALL: return "small"
    if v < V_MODERATE: return "moderate"
    return "strong"


def cramers_v(values: list, M: list) -> float:
    cats = sorted(set(values))
    table = np.array([[sum(1 for v, m in zip(values, M) if v == c and m == mm)
                       for mm in (0, 1)] for c in cats], dtype=float)
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def permutation_p(values: list, M: np.ndarray, obs_nmi: float, rng) -> float:
    ge = 0
    for _ in range(N_PERM):
        if normalized_mutual_info_score(values, rng.permutation(M)) >= obs_nmi:
            ge += 1
    return (ge + 1) / (N_PERM + 1)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy = json.loads(CLEANED_TAXONOMY.read_text())

    ann = {json.loads(l)["orig_index"]: json.loads(l)
           for l in ANNOTATIONS_PATH.read_text().splitlines() if l.strip()}
    Mmap = {json.loads(l)["orig_index"]: json.loads(l)["M"]
            for l in INFERENCE_PATH.read_text().splitlines() if l.strip()}

    idx = [i for i in ann if i in Mmap]
    M = np.array([Mmap[i] for i in idx])
    n = len(idx)
    baseline = M.mean()
    rng = np.random.default_rng(42)

    print(f"Joined {n} questions | baseline error rate {baseline*100:.1f}% "
          f"({int(M.sum())} errors)\n")

    results = {}
    for attr in taxonomy:
        values = [str(ann[i].get(attr)) for i in idx]

        # per-value error rates
        by_val = defaultdict(lambda: [0, 0])  # value -> [count, errors]
        for v, m in zip(values, M):
            by_val[v][0] += 1
            by_val[v][1] += int(m)
        per_value = []
        for v, (cnt, err) in sorted(by_val.items(), key=lambda x: -x[1][1] / x[1][0]):
            rate = err / cnt
            per_value.append({"value": v, "count": cnt, "errors": err,
                              "error_rate": round(rate, 3),
                              "lift": round(rate / baseline, 2),
                              "low_support": cnt < MIN_SUPPORT})

        nmi = normalized_mutual_info_score(values, M)
        p = permutation_p(values, M, nmi, rng)
        v = cramers_v(values, M)
        useful = (p < P_THRESHOLD) and (v >= V_WEAK)
        results[attr] = {"nmi": round(float(nmi), 4), "p_value": round(p, 4),
                         "cramers_v": round(v, 3), "effect": effect_label(v),
                         "useful": useful, "per_value": per_value}
        flag = "USEFUL" if useful else "not useful"
        print(f"{attr:<44} V={v:.3f} ({effect_label(v):<10}) p={p:.4f}  -> {flag}")

    useful_attrs = [a for a in results if results[a]["useful"]]
    print(f"\nUseful attributes: {len(useful_attrs)}/{len(taxonomy)}")

    # ---------------- Figures ---------------- #
    _fig_importance(results, baseline)
    _fig_error_rates(results, baseline, useful_attrs)

    # ---------------- Reports ---------------- #
    REPORT_JSON.write_text(json.dumps(
        {"n": n, "baseline_error_rate": round(float(baseline), 4),
         "attributes": results}, indent=2, ensure_ascii=False))
    _write_md(results, baseline, n, useful_attrs)
    print(f"\nSaved -> {REPORT_MD}\nSaved figures -> {FIG_DIR}")


def _fig_importance(results, baseline):
    attrs = sorted(results, key=lambda a: results[a]["cramers_v"])
    vals = [results[a]["cramers_v"] for a in attrs]
    colors = [BLUE if results[a]["useful"] else MUTED for a in attrs]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = range(len(attrs))
    ax.barh(list(y), vals, color=colors, height=0.62, zorder=3)
    for i, a in enumerate(attrs):
        tag = "significant" if results[a]["useful"] else f"n.s. (p={results[a]['p_value']:.2f})"
        ax.text(vals[i] + 0.004, i, f"{vals[i]:.2f}  {tag}", va="center", fontsize=7.3, color=INK2)
    ax.set_yticks(list(y)); ax.set_yticklabels(attrs, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.5 + 0.05); ax.invert_yaxis()
    ax.set_title("Attribute importance — association with errors (Cramer's V)\n"
                 "blue = statistically significant (p<0.05); gray = not",
                 color=INK, fontsize=11, loc="left")
    ax.xaxis.grid(True, color=GRID, zorder=0); _style(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "attribute_importance.png"); plt.close(fig)


def _fig_error_rates(results, baseline, useful_attrs):
    show = sorted(useful_attrs, key=lambda a: results[a]["cramers_v"], reverse=True)[:6]
    if not show:
        return
    ncol = 2
    nrow = (len(show) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, attr in enumerate(show):
        ax = axes[k // ncol][k % ncol]
        pv = [d for d in results[attr]["per_value"] if not d["low_support"]]
        pv = sorted(pv, key=lambda d: d["error_rate"])
        labels = [f"{d['value']} (n={d['count']})" for d in pv]
        rates = [d["error_rate"] * 100 for d in pv]

        def c(r):
            if r > baseline * 100 * 1.15: return CRITICAL
            if r < baseline * 100 * 0.85: return GOOD
            return BLUE
        colors = [c(r) for r in rates]
        yy = range(len(labels))
        ax.barh(list(yy), rates, color=colors, height=0.66, zorder=3)
        ax.axvline(baseline * 100, color=INK2, lw=1.3, ls="--", zorder=4)
        for i, r in enumerate(rates):
            ax.text(r + 0.5, i, f"{r:.0f}%", va="center", fontsize=7, color=INK2)
        ax.set_yticks(list(yy)); ax.set_yticklabels(labels, fontsize=7.2)
        ax.invert_yaxis()
        ax.set_title(f"{attr}  (V={results[attr]['cramers_v']:.2f})", color=INK, fontsize=9.5, loc="left")
        ax.xaxis.grid(True, color=GRID, zorder=0); _style(ax)
    for j in range(len(show), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Error rate by value  (dashed line = baseline {baseline*100:.0f}%; "
                 "red = worse, green = better)", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "error_rates.png"); plt.close(fig)


def _write_md(results, baseline, n, useful_attrs):
    lines = ["# v0 — Phase 3: Usefulness Report\n",
             f"Questions: **{n}** | baseline error rate: **{baseline*100:.1f}%**. "
             "No multiple-comparison correction applied.\n",
             "## Attribute ranking (association with errors)\n",
             "| attribute | Cramer's V | effect | p-value | verdict |",
             "|---|---|---|---|---|"]
    for a in sorted(results, key=lambda a: results[a]["cramers_v"], reverse=True):
        r = results[a]
        vd = "✅ useful" if r["useful"] else "— not useful"
        lines.append(f"| {a} | {r['cramers_v']:.3f} | {r['effect']} | {r['p_value']:.4f} | {vd} |")
    lines += [f"\n**Useful attributes: {len(useful_attrs)}/{len(results)}**\n",
              "## Error rate by value (useful attributes)\n"]
    for a in sorted(useful_attrs, key=lambda a: results[a]["cramers_v"], reverse=True):
        lines.append(f"### {a}")
        lines.append("| value | n | error rate | lift vs baseline |")
        lines.append("|---|---|---|---|")
        for d in results[a]["per_value"]:
            note = " ⚠️low-n" if d["low_support"] else ""
            lines.append(f"| {d['value']}{note} | {d['count']} | "
                         f"{d['error_rate']*100:.0f}% | {d['lift']}× |")
        lines.append("")
    lines += ["## Figures\n",
              "- `figures/attribute_importance.png` — which attributes relate to errors",
              "- `figures/error_rates.png` — error rate per value for useful attributes"]
    REPORT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
