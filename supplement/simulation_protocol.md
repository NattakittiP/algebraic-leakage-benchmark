# Pre-Registration Simulation Protocol

**Study Title:** Synthetic Benchmark for Detecting Definitional Leakage in Machine-Learning Pipelines for Paired Biomedical Outcomes

**Protocol Version:** 1.0  
**Registration Date:** Prior to data generation

---

## 1. Study Objective

To demonstrate, via a reproducible synthetic benchmark, that:

1. Including a variable mathematically embedded in the outcome (TG4h, which is derived from TCR) causes severe AUC inflation (definitional leakage).
2. Clean, fold-sealed machine-learning pipelines achieve at-chance AUROC (~0.48–0.52) in the null scenario, matching the original clinical manuscript's results.
3. Eight distinct leakage types can be systematically benchmarked and ranked by their degree of performance inflation.
4. WBV (whole blood viscosity) has no detectable independent effect on triglyceride clearance in the null scenario (and a large effect in the positive-control scenario).

---

## 2. Central Thesis Statement

> "This study is not a clinical test of WBV physiology, but a reproducible synthetic benchmark showing how definitional leakage can distort machine-learning performance and interpretation in paired biomedical outcomes."

---

## 3. Data Generator Specification

### 3.1 Generation Order (pre-specified; deviations prohibited)

1. Sample Age, Sex, BMI, Hct, TP, HDL, LDL from truncated Normal distributions (parameters in Section 3.2)
2. Compute WBV = 0.12 × Hct + 0.17 × (TP − 2.07)
3. Sample TG0h from a shifted LogNormal distribution
4. **Generate TCR first** (primary latent response) using scenario-specific equation
5. **Derive TG4h = TG0h × (1 − TCR/100)** — AFTER TCR is generated

**Deviation from this order is prohibited.** Reversing steps 4 and 5 constitutes definitional leakage and is the key benchmark violation being studied.

### 3.2 Population Parameters (null scenario)

| Variable | Distribution | Mean | SD | Range |
|----------|-------------|------|----|-------|
| Age | TruncNormal | 53.0 | 10.0 | [18, 75] |
| Sex | Bernoulli | p_male = 0.485 | — | {0, 1} |
| BMI | TruncNormal | 24.0 | 3.0 | [16, 33] |
| Hct (%) | TruncNormal | 41.7 | 3.75 | [30, 55] |
| TP (g/dL) | TruncNormal | 6.88 | 0.62 | [5.0, 8.9] |
| HDL (mg/dL) | TruncNormal | 50.5 | 9.7 | [25, 87] |
| LDL (mg/dL) | TruncNormal | 131.0 | 28.0 | [70, 215] |
| TG0h (mg/dL) | ShiftedLogNormal | ~700 | ~250 | [350, 1750] |
| TCR (%) | TruncNormal | 52.2 | 18.6 | [−10, 99.9] |

### 3.3 Sample Size

Primary analysis: n = 1500 synthetic records per scenario.  
Sensitivity: n ∈ {300, 750, 1500, 3000}.

---

## 4. Scenarios (Pre-specified)

| Scenario | TCR Generation | Expected Clean AUROC |
|----------|---------------|---------------------|
| Null | TCR ⊥ all predictors | 0.48–0.52 |
| Weak signal | β_age, β_HDL, β_BMI (small) | 0.55–0.60 |
| Moderate signal | β_TG0h, β_BMI (moderate) | 0.65–0.75 |
| WBV positive control | β_WBV = 3.5 SD/SD | 0.80+ |

---

## 5. Leakage Types (Pre-specified Taxonomy)

| # | Leakage Type | Mechanism | Expected AUC Inflation |
|---|-------------|-----------|----------------------|
| 1 | Clean (no leakage) | Baseline | 0 (by definition) |
| 2 | TG4h leakage | TG4h added as predictor | Large (+0.40+) |
| 3 | TCR leakage | TCR added as predictor | Maximal (~1.00) |
| 4 | Global scaling | Scaler fit on full data | Small (+0.02–0.05) |
| 5 | Global winsorisation | Winsorize on full data | Small (+0.02–0.05) |
| 6 | Global label threshold | Q1 TCR from full data | Small (+0.01–0.03) |
| 7 | SMOTE before CV | Oversample before split | Moderate (+0.05–0.15) |
| 8 | Feature selection leak | Select on full data | Moderate (+0.03–0.10) |
| 9 | Combined | TG4h + global prep + SMOTE | Maximal |

---

## 6. Primary Outcomes (Pre-specified)

- **AUC Inflation** = AUC_leaky − AUC_clean (per leakage type × model)
- **Attribution Distortion** = rank shift of each feature in SHAP importance
- **False Attribution Rate** = P(WBV ranked top-3 SHAP | null scenario)

---

## 7. Seed Strategy

- Seed 42: development and testing
- Seed 2026: **main analysis** (all primary results use this seed)
- Seeds 1–100: robustness sweep

---

## 8. Confirmatory Rules

A result is considered pre-specified confirmatory if:
- Null scenario clean AUROC: all 4 models within [0.44, 0.56]
- TG4h leakage AUROC: > 0.85
- TCR leakage AUROC: > 0.95
- WBV positive-control AUROC: > 0.75
- Null scenario WBV SHAP rank: ≥ 4 (not in top-3)
- WBV positive-control WBV SHAP rank: ≤ 3 (in top-3)
