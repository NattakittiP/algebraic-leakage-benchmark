# Paired Outcome Leakage Reporting Checklist (POLRC)

**Version:** 1.0  
**Applies to:** Machine-learning studies using paired biomedical measurements (pre/post, baseline/endpoint)

Use this checklist when reporting, reviewing, or implementing ML pipelines with paired outcomes.  
Answer YES/NO for each item. Any NO requires explicit justification in the manuscript.

---

## Section A: Data Generation / Feature Engineering

**A1. Outcome derivation check**  
☐ Have you listed the mathematical formula used to derive each feature?  
☐ Is any predictor a direct algebraic function of the outcome variable?  
*If YES: that predictor constitutes definitional leakage and must be excluded from clean predictors.*

**A2. TCR / ratio variables**  
☐ If using a clearance rate, reduction rate, or any formula of the form (baseline − endpoint) / baseline, is the intermediate "endpoint" value excluded from predictors?  
*Example: TCR = (TG0h − TG4h)/TG0h × 100 → including TG4h reconstructs TCR exactly.*

**A3. Temporal contamination**  
☐ Are all predictors temporally prior to or simultaneous with the baseline measurement?  
☐ Are no post-treatment measurements used as predictors (except when studying treatment response)?

---

## Section B: Label / Target Construction

**B4. Threshold derivation**  
☐ Is the label threshold (e.g., Q1 of TCR) computed ONLY from training-fold data?  
☐ Is the global threshold (computed before any split) explicitly NOT used in the clean pipeline?

**B5. Label encoding documentation**  
☐ Is the label threshold value reported (not just "Q1")?  
☐ Is the threshold verified to be stable across folds (SD < 10% of mean threshold)?

---

## Section C: Preprocessing

**B6. Scaler fold-sealing**  
☐ Is the StandardScaler (or similar) fitted ONLY on training fold data?  
☐ Is there explicit code documentation confirming this?

**B7. Winsorisation fold-sealing**  
☐ Are winsorisation percentile bounds computed from training data only?

**B8. SMOTE / oversampling timing**  
☐ Is oversampling applied ONLY after the train/test split (inside each fold)?  
☐ Is SMOTE applied BEFORE the validation fold is separated? (This is the leak — it must NOT happen.)

**B9. Feature selection timing**  
☐ If feature selection is performed, is the selection based ONLY on training-fold data?  
☐ Is there a separate test confirming no test-set information was used?

---

## Section D: Model Evaluation

**B10. Nested CV documentation**  
☐ Is nested cross-validation used (separate inner and outer folds)?  
☐ Are outer-fold AUROC values reported separately before averaging?  
☐ Is the number of folds justified (recommend outer ≥ 5, inner ≥ 5)?

**B11. Multiple metrics**  
☐ Are PR-AUC and Brier score reported in addition to AUROC?  
☐ Is calibration assessed (ECE or calibration slope)?

---

## Section E: Transparency and Reproducibility

**B12. Reproducibility package**  
☐ Is the code publicly available with a DOI (e.g., Zenodo)?  
☐ Can results be reproduced with a single command (e.g., `bash run_all.sh`)?  
☐ Is a pinned `environment.yml` or `requirements.txt` provided?  
☐ Is the random seed explicitly stated for all analyses?

---

## Scoring

- **12/12**: Fully compliant — no known leakage concerns
- **10–11/12**: Minor gaps — address in revision
- **< 10/12**: Substantial concerns — major revision required before publication

---

*This checklist was developed as part of the TCR Leakage Benchmark (BMC Bioinformatics, 2026).*
