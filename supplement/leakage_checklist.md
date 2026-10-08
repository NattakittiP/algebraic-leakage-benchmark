# POLRC — Paired Outcome Leakage Reporting Checklist (v1.0)

Standalone version of the 12-item audit instrument described in the paper
(main text, POLRC section; full table and worked examples in Additional file 1, Section S1).

## How to use

- Answer each item **Yes = 1 / No = 0 / N/A**.
- Any **No** is a demonstrable leakage risk and needs explicit justification in the study report.
- **Score** = proportion of *applicable* items answered Yes.
- **Pass:** ≥ 83.3% of applicable items (10/12 when all items apply).
- N/A is allowed only when an item is structurally inapplicable (e.g. B5 when no oversampling is used).
  Items that cannot be determined from the report are recorded as such and excluded from the denominator.

Items A1–A3 address Tier 1 (algebraically deterministic) leakage, B1–B6 address Tier 2
(preprocessing) leakage, and C1–C3 address evaluation and reproducibility.

---

## Domain A — Data generation (formula audit)

**A1. Formula listing and mathematical coupling** (L1, L2)
Have you listed every formula used to derive labels and features? Is any predictor a component
of the outcome formula (e.g. the baseline *A*)? If yes, justify its inclusion and report the
coupling; if the predictors jointly determine the outcome, exclude the component.
- Pass: feature list annotated; no algebraic antecedent in the predictor set.
- Fail: TG4h included when the label is derived from TCR = f(TG0h, TG4h).

**A2. Post-baseline component excluded** (L1)
If using a clearance/response formula *Y = h(A, B)*, is the post-baseline component *B* excluded
from the predictors?
- Pass: `peak_cgm` excluded when *Y* = `peak_cgm` − `pre_meal_cgm`.
- Fail: `peak_cgm` included in the feature set.

**A3. Temporal ordering** (L1)
Are all predictors measured before the outcome? Are post-treatment values excluded (unless the
response itself is the target)?
- Pass: temporal diagram provided; no post-treatment features.
- Fail: a post-discharge status field included in mortality prediction.

## Domain B — Label construction and preprocessing

**B1. Fold-sealed label threshold** (L5)
Is the label threshold computed only from training-fold data, with no global threshold in the
clean pipeline?
- Pass: Q1 computed inside each CV fold; test-fold labels assigned after the split.
- Fail: global Q1 computed on the full dataset before splitting.

**B2. Threshold reported** (L5)
Is the threshold value reported numerically, and is the fold-to-fold threshold SD < 10% of the
mean threshold?
- Pass: "Q1 = 28.4 mg/dL; fold SD = 1.2 mg/dL (4.2%)".

**B3. Fold-sealed scaling** (L3)
Is the scaler fitted only on training-fold data, with explicit code documentation?
- Pass: `scaler.fit(X_train)` inside the CV loop.
- Fail: `scaler.fit(X_all)` before splitting.

**B4. Fold-sealed winsorisation / clipping** (L4)
Are winsorisation or clipping bounds computed from training data only?
- Pass: 1st–99th percentile bounds from the training fold.
- Fail: bounds computed on the entire dataset.

**B5. Fold-sealed oversampling** (L6)
Is oversampling (SMOTE, etc.) applied only inside each training fold, after the train/test split?
- Pass: SMOTE in the pipeline after the split.
- Fail: SMOTE applied to the full dataset before CV.

**B6. Fold-sealed feature selection** (L7)
If feature selection is performed, is it based only on training-fold data?
- Pass: univariate filter fitted on `X_train` inside the loop.
- Fail: features selected by correlation on the full dataset.

## Domain C — Evaluation and reproducibility

**C1. Fold-sealed evaluation design**
Are all preprocessing parameters (scaler, imputer, selector) fitted exclusively within each
training fold? Are ≥ 5 outer folds and ≥ 10 random seeds used?
- Pass: fold-sealed pipeline; 5-fold × 30 seeds reported.

**C2. Metrics beyond AUROC**
Are PR-AUC and Brier score reported alongside AUROC? Is the calibration slope reported?
(A slope far above 1 under in-distribution, near-deterministic prediction is a leakage warning sign.)
- Pass: AUROC + PR-AUC + Brier + calibration slope all reported.
- Fail: AUROC only; a slope of 3.6 left unreported.

**C3. Reproducibility**
Is the code publicly available? Can the results be reproduced with a single command? Are all
random seeds and package versions pinned?
- Pass: public repository + `bash run_all.sh` + `requirements.txt`.
- Fail: code "available upon request"; seeds not stated.

---

## Scoring sheet

| Item | Yes | No | N/A | Notes |
|------|-----|----|-----|-------|
| A1 | ☐ | ☐ | ☐ | |
| A2 | ☐ | ☐ | ☐ | |
| A3 | ☐ | ☐ | ☐ | |
| B1 | ☐ | ☐ | ☐ | |
| B2 | ☐ | ☐ | ☐ | |
| B3 | ☐ | ☐ | ☐ | |
| B4 | ☐ | ☐ | ☐ | |
| B5 | ☐ | ☐ | ☐ | |
| B6 | ☐ | ☐ | ☐ | |
| C1 | ☐ | ☐ | ☐ | |
| C2 | ☐ | ☐ | ☐ | |
| C3 | ☐ | ☐ | ☐ | |

**Score** = Yes / (12 − N/A) = ____ %  →  Pass if ≥ 83.3%
