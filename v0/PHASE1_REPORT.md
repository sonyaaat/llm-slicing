# v0 — Phase 1: Annotation — Report

**Goal:** annotate all 500 MedQA questions with the human-designed taxonomy
(`taxonomy_v0.json`), independently of the v2 (automated) taxonomy, so that the
attributes can later be validated (Phase 2) and tested for usefulness against the
error label M (Phase 3).

**Status: ✅ completed.** 500/500 questions annotated across all 14 attributes.

---

## Folder structure (v0/)

```
v0/
├── taxonomy_v0.json              # the human taxonomy (copied in; self-contained)
├── src/
│   └── annotate.py               # Phase 1 annotation script
├── results/
│   ├── annotations_v0.jsonl      # 500 annotated questions (main output)
│   └── annotation_summary.json   # coverage + value distributions
└── PHASE1_REPORT.md              # this report
```

Independence from v2: this code neither reads nor reuses any v2 artifact. The
only shared input is the taxonomy-agnostic dataset `data/medqa_500.jsonl`. The
error label M is **not** used in Phase 1 (annotation is blind to correctness).

---

## Steps implemented

1. **Loaded the taxonomy** (`taxonomy_v0.json`): 14 attributes — 10 categorical
   (with closed value lists) + 4 boolean.
2. **Deterministic attributes (no LLM)** — computed in code for reliability:
   - `usmle_step` — mapped from the dataset `meta_info` field (`step2&3` → `step2_3`).
   - `vignette_length_level` — binned from question word count
     (short ≤ 60, medium ≤ 120, long > 120); `word_count` stored for transparency.
3. **Semantic attributes (12) via GPT-4o** — one call per question, JSON mode,
   `temperature=0`. The prompt lists every attribute with its allowed values and
   forces a choice from the closed vocabulary, allowing `unclear`.
4. **Self-check** — the prompt asks the model to verify every value is in the
   allowed list before returning.
5. **Validation in code** — each returned value is checked against
   `taxonomy_v0.json`; out-of-vocabulary values are replaced with `unclear`
   (categorical) and counted.
6. **Incremental, resumable save** — each result written immediately to
   `annotations_v0.jsonl`; reruns skip already-annotated `orig_index`.
7. **Summary** — coverage and per-value distributions written to
   `annotation_summary.json` and printed.

**Config:** model `gpt-4o`, `temperature=0`, ~500 sequential calls (~15 min),
cost ≈ $1–2.

---

## Did it achieve what was needed?

Yes, for Phase 1 (annotation):
- ✅ All 500 questions annotated on all 14 attributes.
- ✅ Fully independent of v2.
- ✅ Closed-vocabulary enforced; only **10 / 6000 values (0.17%)** came back
  out-of-vocabulary and were safely coerced to `unclear`.
- ✅ High coverage: 12/14 attributes at 100%, `knowledge_domain` 98%,
  `body_system` 99%.
- ✅ Deterministic attributes computed without LLM (reliable, free).

Phase 1 produces the **annotated dataset**; it does not yet judge which
attributes are useful — that is Phase 3.

---

## Limitations / what did not work perfectly

1. **`no_vignette` is never assigned** in `vignette_length_level` (0 cases).
   Because the level is derived deterministically from word count, absence of a
   clinical vignette cannot be detected this way. The signal is not lost — it
   appears in `evidence_type = no_patient_vignette` (22 cases) — but the
   `no_vignette` value of this attribute is effectively dead.
2. **Near-degenerate attributes (likely weak in Phase 3):**
   - `requires_calculation_or_formula`: only **2 True** — too rare to slice on.
   - `distractor_similarity_level`: `low` used only **1** time; the model
     defaulted heavily to `medium` (429/500), suggesting weak discrimination on
     this subjective attribute.
   - `domain_specificity_level`: only 2 of 4 values used
     (`rare_or_edge_case_knowledge` never assigned).
   - `cognitive_level`: dominated by `scenario_based_problem_solving` (389/500).
3. **Minor vocabulary confusion:** 9 of the 10 invalid values were in
   `knowledge_domain` (the model occasionally returned a body-system-like value),
   coerced to `unclear`.
4. **Annotation is LLM-based, not human.** Reliability (self-agreement /
   Cohen's kappa) is **not yet measured** — that is Phase 2. No human gold set
   was labeled in this phase.
5. **Single pass, temperature 0.** Reproducible, but no cross-run agreement
   estimate yet.

---

## Brief results analysis

- **Well-behaved, discriminative attributes** (good spread, usable):
  `clinical_task_type` (13 values, top = diagnosis 111), `knowledge_domain`
  (17 values), `body_system` (15 values, fairly even), `reasoning_load`
  (two_step 258 / multi_step 193 / direct 49), `evidence_type`, and
  `usmle_step` (267/233).
- **Balanced booleans** (usable): `requires_risk_benefit...` (114 True),
  `requires_guideline...` (92 True), `contains_negation_or_exception` (63 True).
- **Skewed / likely-uninformative** (candidates for removal or value-merging in
  Phase 2): `requires_calculation_or_formula`, `distractor_similarity_level`,
  `domain_specificity_level`, `cognitive_level`.

Overall the annotation is clean and high-coverage; a few attributes are already
visibly skewed and will need pruning before or during the statistical analysis.

---

## Next (not in this phase)

- **Phase 2 — Validity:** coverage recap, dead/rare value handling, and
  reliability (annotate a subset twice → self-agreement κ).
- **Phase 3 — Usefulness:** join with M, compute error-rate per value, NMI +
  permutation test, Cramér's V → verdict per attribute.
