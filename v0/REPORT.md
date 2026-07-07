# v0 — Full Report: Human Taxonomy Annotation & Error Analysis

**Project:** LLM Slicing (MSCI720) — identifying where and why a target LLM fails
on MedQA-USMLE.

**What v0 is:** a self-contained, human-designed-taxonomy track. The taxonomy was
authored by hand (blind to model correctness); the pipeline then annotates all
questions with it and tests, statistically, which attributes are associated with
the model's errors. v0 is **independent of the automated (v2) taxonomy** — it
reuses no v2 result. The only shared inputs are the dataset sample and the target
model's error labels.

---

## 0. Setup and inputs

- **Dataset:** MedQA-USMLE, 4-option test split; a fixed sample of **500 questions**
  (`random_state=42`), each with a stable `orig_index`.
- **Target model (studied model):** `gpt-4o-mini`, run earlier with chain-of-thought
  at `temperature=0`. For each question it produced a binary **error label M**
  (0 = correct, 1 = wrong).
  - **Baseline: accuracy 78.4% → error rate 21.6% (108 / 500 errors).**
- **Analyzer model (annotator):** `gpt-4o`, `temperature=0`, used only to apply the
  taxonomy — never to invent it.
- **Taxonomy (`taxonomy_v0.json`):** 14 attributes authored by hand —
  10 categorical (with closed value lists) + 4 boolean. Every categorical
  attribute includes an `unclear` escape value.

Folder layout:
```
v0/
├── taxonomy_v0.json              # the human taxonomy
├── src/
│   ├── annotate.py               # Phase 1
│   ├── validate.py               # Phase 2
│   └── analyze.py                # Phase 3
├── results/
│   ├── annotations_v0.jsonl      # Phase 1 output (500 × 14 tags)
│   ├── annotation_summary.json
│   ├── validity_report.{md,json} # Phase 2
│   ├── taxonomy_v0_cleaned.json  # 13 attributes after Phase 2
│   ├── usefulness_report.{md,json} # Phase 3
│   └── figures/*.png
├── PHASE1_REPORT.md
└── REPORT.md                     # this document
```

---

## Phase 1 — Annotation

**Goal:** assign every one of the 500 questions a value for each of the 14 taxonomy
attributes, blind to M.

**How it was done (`src/annotate.py`):**

1. **Deterministic attributes (computed in code, no LLM)** — for reliability and
   zero cost:
   - `usmle_step`: mapped directly from the dataset `meta_info` field
     (`step2&3` → `step2_3`).
   - `vignette_length_level`: binned from question word count
     (short ≤ 60, medium ≤ 120, long > 120); the raw `word_count` is stored too.
2. **Semantic attributes (12) via GPT-4o** — one API call per question, JSON mode,
   `temperature=0`. The prompt lists each attribute with its **closed** value list
   and requires the model to pick exactly one allowed value (or `unclear`), with a
   **self-check** instruction to verify every value is in the list before answering.
3. **Code-side validation** — each returned value is checked against
   `taxonomy_v0.json`; any out-of-vocabulary value is coerced to `unclear` and
   counted.
4. **Incremental, resumable save** to `annotations_v0.jsonl` (skips already-done
   `orig_index`).

**Results:**
- **500 / 500 questions annotated** on all 14 attributes.
- Only **10 of ~6000 values (0.17%)** came back out-of-vocabulary and were coerced
  to `unclear` (9 in `knowledge_domain`, 1 in `cognitive_level`).
- **Coverage high:** 12/14 attributes at 100%; `knowledge_domain` 98%,
  `body_system` 99%.
- Cost ≈ $1–2, ~15 min.

---

## Phase 2 — Validity (quality of the annotation, no M used)

**Goal:** before correlating anything with errors, verify the tags themselves are
trustworthy.

**How it was done (`src/validate.py`):** three checks per attribute.

1. **Coverage** — share of non-`unclear` values.
2. **Distribution / degeneracy** — value counts; flag **dead** values (never used),
   **rare** values (< 10 uses), and **dominance skew** (share of the most frequent
   value). A highly skewed attribute barely discriminates between questions.
3. **Reliability (self-agreement)** — re-annotate a fixed subset of **50 questions**
   independently and compute **Cohen's kappa** (and raw agreement) between the two
   passes, per attribute.

**Verdict rule (per attribute):**
- **drop** if categorical top-value share ≥ 95%, or boolean minority class < 2%;
- **weak** if top share ≥ 85%, or coverage < 80%, or κ < 0.40;
- otherwise **keep**.

**Results — 11 keep / 2 weak / 1 drop → cleaned taxonomy = 13 attributes:**
- **Dropped (1):** `requires_calculation_or_formula` — only **2 / 500** True
  (MedQA has almost no calculation items); too rare to analyze.
- **Weak (2):** `distractor_similarity_level` (86% "medium"; "low" used once) and
  `contains_negation_or_exception` (87% False) — kept but flagged.
- **Coverage:** all attributes 98–100%.
- **Reliability:** κ ranged **0.83–1.00** (high).
  - ⚠️ **Caveat:** both passes ran at `temperature=0`, so they are near-deterministic;
    the high κ mostly reflects the model's determinism, not genuine robustness. A
    stricter estimate would use `temperature > 0` or human labels. Reliability here
    is model-vs-model, not model-vs-human.
- **Dead values removed** from the cleaned taxonomy: `other` (clinical_task_type),
  `multidomain` (knowledge_domain), `no_vignette` (vignette_length_level — cannot be
  detected from word count alone), `rare_or_edge_case_knowledge` (domain_specificity).

**Figures:** `coverage.png`, `skew.png`, `value_distributions.png`,
`reliability_kappa.png`.

---

## Phase 3 — Usefulness (which attributes relate to errors)

**Goal:** the core question — for each of the 13 cleaned attributes, is it
associated with the model's errors, and on which values does the model fail most?

**How it was done (`src/analyze.py`, pure statistics — no LLM):**

1. **Join** annotations with the error label M by `orig_index` (500 questions;
   baseline error rate 21.6%).
2. **Error rate per value** — for each attribute value: count, error count, error
   rate, and **lift** (rate ÷ baseline). Values with < 10 questions are flagged
   **low-support** and excluded from the error-rate figures.
3. **Association strength** — `NMI(attribute, M)` (normalized mutual information).
4. **Significance** — a **permutation test** (2000 shuffles of M): p-value =
   fraction of random shuffles whose NMI ≥ the observed NMI. **No multiple-comparison
   correction** was applied (by choice).
5. **Effect size** — **Cramér's V** from the attribute×M contingency table.
6. **Verdict:** *useful* if p < 0.05 **and** Cramér's V ≥ 0.10.

**Results — 5 of 13 attributes are useful.** Ranked by effect size:

| attribute | Cramér's V | p-value | verdict |
|---|---|---|---|
| body_system | 0.205 | 0.037 | ✅ useful |
| reasoning_load | 0.168 | 0.0005 | ✅ useful |
| distractor_similarity_level | 0.154 | 0.004 | ✅ useful |
| domain_specificity_level | 0.143 | 0.0015 | ✅ useful |
| vignette_length_level | 0.111 | 0.038 | ✅ useful |
| knowledge_domain | 0.184 | 0.41 | — not useful (n.s.) |
| evidence_type | 0.163 | 0.058 | — not useful |
| clinical_task_type | 0.145 | 0.60 | — not useful (n.s.) |
| cognitive_level | 0.097 | 0.08 | — not useful |
| requires_risk_benefit… | 0.097 | 0.03 | — effect too small |
| requires_guideline… | 0.064 | 0.17 | — not useful |
| contains_negation… | 0.050 | 0.31 | — not useful |
| usmle_step | 0.007 | 0.91 | — not useful |

**Where the model fails most (error rate by value, baseline 21.6%):**

- **reasoning_load** — direct **8%** → two_step 18% → multi_step **30%** (≈4× harder).
- **body_system** — worst: multisystem **39%**, respiratory 33%, neurologic/renal 28%;
  best: dermatology **0%**, psychiatry 4%, gastrointestinal 17%.
- **distractor_similarity_level** — medium 19% → high **37%** (≈2×).
- **domain_specificity_level** — common_core 14% → specialized **26%** (≈2×).
- **vignette_length_level** — short 10% → medium 20% → long **26%**.

**Figures:** `attribute_importance.png` (ranking, significant in blue),
`error_rates.png` (error rate per value vs baseline; red = worse, green = better).

---

## Overall findings

The human taxonomy produced a clear, interpretable **map of the target model's
weaknesses**:

> `gpt-4o-mini` errs more on questions that require **more reasoning steps**, have
> **highly similar answer options**, rely on **specialized (vs common) knowledge**,
> are **longer**, or concern **multisystem / respiratory / neurologic** topics; it is
> strongest on dermatology and psychiatry items.

These patterns are (a) intuitive and medically sensible, (b) statistically
significant (p < 0.05), and (c) resting on well-supported values (the driving
categories each contain ~20–430 questions, not 1–2). This directly supports RQ1
(the taxonomy is meaningful) and RQ3 (significant error-linked slices exist).

**Comparison to v2 (automated taxonomy):** v2 yielded only 2 stable attributes and
was polluted by over-granular, topic-specific tags. The hand-designed v0 taxonomy is
substantially cleaner and more informative (13 valid attributes, 5 error-linked).

---

## Limitations

1. **LLM annotation, not human.** Tags were assigned by GPT-4o. The taxonomy is
   human (blind to M), but the labeling is not; no human gold set has been created.
2. **Reliability caveat.** Self-agreement was measured at `temperature=0`, so it
   reflects determinism more than true robustness. Model-vs-human agreement (Cohen's
   kappa on a ~30-item gold set) is not yet measured.
3. **Rare categories kept in the statistics.** Low-support values (< 10) are flagged
   and excluded from figures, but the significance tests still included them. For
   many-category attributes (`clinical_task_type` 13, `knowledge_domain` 17), sparse
   cells reduce statistical power and can inflate Cramér's V — so their "not useful"
   verdict is not fully reliable. (A merge-rare-into-"other" pass would fix this.)
4. **`no_vignette` not detectable** from word count; that value is effectively unused
   (the signal lives in `evidence_type = no_patient_vignette`).
5. **No multiple-comparison correction** (deliberately omitted): with 13 tests, some
   borderline p-values may be optimistic.
6. **Scope:** a single target model (`gpt-4o-mini`), a single 500-question sample,
   one annotation pass.

---

## What was achieved (summary)

- A 3-phase, reproducible v0 pipeline (annotate → validate → analyze), fully
  separate from v2.
- 500 questions annotated on a 14-attribute human taxonomy; validated down to 13
  trustworthy attributes; 5 shown to be significantly linked to model errors.
- A concrete, defensible characterization of the model's failure modes, with tables
  and figures ready for write-up.
```
Status:  Phase 1 ✅   Phase 2 ✅   Phase 3 ✅
Optional next: Phase 4 (slices = combinations of the 5 useful attributes),
               human gold set for model-vs-human reliability.
```
