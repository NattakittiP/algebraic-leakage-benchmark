# Benchmark Card — TCR Leakage Benchmark

**Version:** 1.0.0  
**Date:** 2026  
**Format:** Following Mitchell et al. (2019) Model Card conventions

---

## Benchmark Overview

**Name:** Synthetic Paired-Outcome Leakage Benchmark (TCR-Leakage-Bench)

**Purpose:** Demonstrate and quantify how definitional leakage — specifically including a mathematically-derived intermediate variable (TG4h) as a predictor — inflates machine-learning performance metrics and distorts feature attribution in paired biomedical outcome settings.

**Central Claim:** In the null scenario (WBV has no effect on TCR), a clean ML pipeline achieves AUROC ≈ 0.48–0.52 (at-chance). Including TG4h as a predictor inflates AUROC to ≥ 0.90 purely through mathematical reconstruction, not predictive validity.

---

## Data Characteristics

| Property | Value |
|----------|-------|
| Data type | Fully synthetic (no real patient data) |
| Generator | Parametric truncated-Normal / shifted-LogNormal |
| Primary seed | 2026 |
| Sample size | n = 1500 (primary); 300–3000 (sensitivity) |
| Features | 9 clean predictors (see below) |
| Label | Binary: low_TCR = (TCR ≤ Q1 of training fold) |
| Scenarios | 4 (null, weak, moderate, WBV-positive) |

### Clean predictors (9)

Age, Sex, BMI, Haematocrit (Hct), Total Protein (TP), Whole Blood Viscosity (WBV), HDL-cholesterol, LDL-cholesterol, Baseline triglyceride (TG0h)

### Excluded from clean pipeline (leakage variables)

- **TG4h** (post-heparin triglyceride) — mathematically embedded in outcome TCR
- **TCR** (triglyceride clearance rate) — IS the outcome variable

---

## Benchmark Tasks

| Task | Models | Metric |
|------|--------|--------|
| Clean baseline | LR, RF, XGB, SVM | AUROC (5×5 nested CV) |
| Leakage type 1–8 | LR, RF, XGB, SVM | AUROC + AUC Inflation |
| SHAP attribution | RF, XGB | Attribution Distortion, FAR |
| Scenario sensitivity | LR | AUROC mean ± SD (100 seeds) |
| Domain shift | RF | Calibration slope, ECE |

---

## Intended Uses

- Education: teaching ML practitioners about leakage types in biomedical data
- Methods benchmarking: testing leakage-detection algorithms
- Reproducibility research: demonstrating one-command reproducible pipelines

## Out-of-scope Uses

- Clinical decision making (data is synthetic, not real patient data)
- Drug or therapy efficacy claims
- Any clinical guidance about triglyceride clearance in real patients

---

## Ethical Considerations

- No real patient data is used at any stage
- All synthetic records are explicitly labelled as synthetic
- Clinical language (patients, hospital, EMR) has been purged from all code and documentation
- Results should not be used to make claims about WBV physiology in real patients

---

## Known Limitations

1. Synthetic data cannot capture all real-world confounding structures
2. The null scenario is by construction — real-world clean AUROC may be higher
3. SMOTE interaction effects depend on k-neighbours parameter
4. SHAP computation is sampled (n=300) for speed — full-dataset SHAP may differ slightly

---

## Citation

See CITATION.cff in the repository root.
