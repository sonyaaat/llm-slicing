# LLM Slicing - Implementation Plan

MSCI720 project | Supervisors: Lukasz Golab, Jarek Szlichta | Deadline: Aug 8

---

## Step 0 - Data Preparation
- Source: MedQA-USMLE test split, 4-option version (1,273 questions total)
- Sample **500 questions** via simple random sampling, fixed seed (e.g. `random_state=42`)

## Step 1 - Target LLM Inference
- Run target LLM (GPT-4o mini) on all 500 questions
- Compare predicted answer to ground truth → binary error label **M** (0 = correct, 1 = incorrect)

## Step 2 - Taxonomy Generation
- Analyzer LLM (GPT-4o), contrastive approach: compare error (M=1) vs. success (M=0) question pairs
- 5 independent runs on random sub-samples
- Naming rules: binary attributes → `is_` prefix, no semantic overlap between attributes
- Accumulate attributes into one shared list across runs

## Step 3 - Pruning #1: Stability Filter
- Keep attributes that appear in **≥3 of 5** taxonomy-generation runs
- Discard one-off / unstable attributes

## Step 4 - Annotation
- Fix a **closed vocabulary** of possible values per attribute before annotating
- Annotate all 500 questions with the analyzer LLM
- Include in the prompt: `not_applicable` tag,  self-check step (model verifies its own tags for overlap)

## Step 5 - Annotation Quality Check
- **Coverage**: % of questions with ≥1 non-`not_applicable` tag (target >80%)
- **Per-attribute coverage**: flag attributes <5–10% usage for removal
- **Reliability**: TODO

## Step 6 - Pruning #2: Statistical Verification
- Compute NMI(attribute, error label M) for each surviving attribute TODO

## Step 7 - Slicing

## Step 8 - Validation & Evaluation

---

**Research questions covered:**
RQ1 → meaningful taxonomy?
RQ2 → reliable annotation?
RQ3 → significant slices found?

---

# Progress Log

| Step | Status |
|------|--------|
| 0 - Data Preparation |  Done |
| 1 - Target LLM Inference |  Done (accuracy 78.4%, 108 errors) |
| 2 - Taxonomy Generation |  First run done; needs prompt iteration |
| ... | Not started |


## Step 1 - Target LLM Inference

- ran `gpt-4o-mini` (CoT, `temperature=0`) on all 500;
  assigned binary error label **M** (0 = correct, 1 = wrong).
- **Result:** accuracy **78.4%** — 392 correct, **108 errors**.

**Prompt used (chain-of-thought, answer label forced):**

*System:*
```
You are a medical expert answering USMLE multiple-choice questions. Think step
by step briefly, then give your final answer. End your response with a line in
exactly this format: 'Answer: X' where X is one of A, B, C, or D.
```
*User:*
```
{question}

Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Reason briefly, then end with 'Answer: X'.
```
The predicted letter is parsed from the `Answer: X` line and compared to the
gold `answer_idx` to set M.

## Step 2 - Taxonomy Generation


**What was done:**
1. Split 500 questions into error (M=1) / success (M=0) pools.
2. 5 independent runs; each processes random (error, success) pairs, asking
   GPT-4o to reuse an existing attribute or propose a new one, accumulating a
   taxonomy. Runs stop on saturation (6 pairs with no new attribute).
3. Cross-run merge: cluster synonymous names, count in how many runs each
   attribute appears; keep `count ≥ 3` (Step 3).
4. Params: `gpt-4o`, gen temp 0.3, 5 runs, max 50 pairs, patience 6.

### Prompts used

**Prompt A — contrastive pair (one call per pair, builds the taxonomy).**
The model sees one wrong + one correct question and the taxonomy accumulated so
far, then either reuses an attribute or proposes a new one.

*System:*
```
You are analyzing why an LLM fails on USMLE medical multiple-choice questions.
You compare a question the model got WRONG with one it got RIGHT and identify a
characteristic OF THE QUESTION ITSELF (not of the answer) that could explain the
difference. You return JSON only.
```
*User (template, filled per pair):*
```
Question A — the model answered INCORRECTLY:
{wrong question + options + correct answer}

Question B — the model answered CORRECTLY:
{correct question + options + correct answer}

Current taxonomy of attributes (you may reuse these):
{list of "name | definition | value_type", or "(empty)"}

Task:
1. Identify ONE characteristic OF THE QUESTION ITSELF (not the answer) that
   could explain why A is harder/different than B.
2. If an existing attribute already covers it, return its exact name in
   "reused_attribute".
3. Otherwise propose a NEW attribute:
   - binary attributes use the prefix "is_" (e.g. is_multi_step_reasoning)
   - give a clear, observable definition
   - before adding, self-check that it does not semantically overlap an
     existing attribute.
Return ONLY JSON of the form:
{"reused_attribute": "name_or_null",
 "new_attribute": {"name": "...", "definition": "...", "value_type": "binary"} or null}
```

**Prompt B — cross-run normalization (one call, after all 5 runs).**
Clusters synonymous attribute names from the five runs into canonical
attributes before stability counting.

*System:*
```
You normalize and deduplicate attribute taxonomies. Return JSON only.
```
*User:*
```
Below is a list of attribute names with definitions, collected from several
independent runs. Some may be synonyms (the same underlying concept named
differently). Group semantically-equivalent attributes into clusters. Each
cluster gets one canonical name (prefer the clearest is_-prefixed name) and a
merged definition. Keep distinct concepts in separate clusters — do NOT
over-merge.

Attributes:
{list of "name: definition"}

Return ONLY JSON of the form:
{"clusters": [{"canonical_name": "...", "canonical_definition": "...",
"value_type": "binary", "members": ["name1", "name2"]}]}
```

**Result:** 42 attributes → 38 after normalization; only **2** passed ≥3 of 5
(`is_multi_step_reasoning` 5/5, `is_mechanism_of_action` 3/5).

**Limitations:**
- Attributes too granular / topic-specific , so each appears once and is  unstable.
- All attributes forced binary (`is_`); no categorical attributes generated
  (Prompt A only offers an `is_` / binary template).
- Normalization was too cautious: it only merged attributes with almost the
same name but did not merge attributes that mean the same thing under different names.

**Proposals for next iteration:**
- Revise prompt to elicit general cognitive/structural difficulty axes, not
  medical topics.
- Allow **categorical** attributes with a closed value set (not only binary);
  forbid free-text. This folds many one-off topic attributes into single stable
  axes.
- Strengthen normalization to merge into higher-level attributes.
- Target ≈8-12 stable attributes.
