# Reviewer Response Bank
## Anticipated Reviewer Questions and Model Responses

**Paper**: *Definitional leakage in machine-learning prediction of paired triglyceride-response phenotypes: a reproducible synthetic-data benchmark*  
**Target journal**: BMC Bioinformatics  
**Last updated**: 2026-05-13

---

## Section 1: Synthetic Data & External Validity

### Q1.1 — "The paper uses only synthetic data. How can results apply to real clinical datasets?"

**Response:**

The use of synthetic data is a deliberate methodological choice, not a limitation. Our primary scientific objective is to characterise the *structural* relationship between definitional leakage and AUROC inflation — a property that follows algebraically from how TCR is defined (TCR = (TG0h − TG4h) / TG0h × 100), irrespective of whether the underlying data are real or synthetic.

Specifically:
- The definitional leakage effect is mathematically guaranteed whenever TG4h is included as a predictor of a label derived from TCR. This holds in any dataset.
- Synthetic data allow us to control the ground truth (null scenario: no real predictive signal) so that any observed AUROC > 0.52 can be unambiguously attributed to leakage rather than to confounding.
- Our four simulation scenarios (null, weak signal, moderate signal, WBV-positive control) span the range of plausible clinical effect sizes, providing a conservative-to-optimistic bracket.

We have specifically positioned this work as a **computational benchmark** rather than a clinical study. The real-world relevance lies in raising awareness that definitional leakage can produce AUROCs ≥ 0.90 in published studies using paired biomarkers, even when no real clinical signal exists.

---

### Q1.2 — "Why not use a real clinical triglyceride dataset?"

**Response:**

We considered this option and rejected it for two reasons:

1. **Ground-truth control**: In a real dataset, we cannot know whether any observed predictive signal is genuine or artifactual. A synthetic null scenario uniquely allows us to prove that any observed AUROC inflation is caused by leakage, not by real biology.

2. **Reproducibility and data sharing**: Real hypertriglyceridaemia post-heparin datasets require ethics approval and data sharing agreements that would prevent open release. Our fully synthetic, openly released benchmark allows any researcher to reproduce every result in this paper with a single command (`make all`).

---

### Q1.3 — "The distribution parameters (mean BMI 24, mean TCR 52.2%) seem idealised. Is this realistic?"

**Response:**

The synthetic distribution parameters are drawn from published norms for moderate-to-severe hypertriglyceridaemia patient populations undergoing lipid apheresis (the clinical context that motivates TCR measurement). TCR of ~50% is consistent with published post-heparin triglyceride clearance studies.

The exact parameter values do **not** affect the paper's central finding: definitional leakage causes AUROC inflation because of the algebraic formula TCR = f(TG0h, TG4h). This relationship holds for any reasonable choice of distribution parameters. We verified this in our robustness sweep across 100 random seeds (Figure 5 and Table S3).

---

## Section 2: Leakage Taxonomy & Methodology

### Q2.1 — "The paper focuses on definitional leakage but there are other leakage types. Is this sufficient?"

**Response:**

We address all seven major leakage types identified in the ML-in-medicine literature (Table 2). However, we intentionally give primary emphasis to definitional leakage because:

1. It is **underrecognised**: preprocessing, resampling, and label leakage are increasingly discussed in methodological guidance, but *definitional leakage from paired biomarkers* has received almost no attention.
2. It produces **the largest inflation**: in our benchmark, definitional leakage inflates AUROC by 0.30–0.45 absolute points, versus 0.02–0.08 for the other types under our null scenario.
3. It is **systematic**: the inflation is not stochastic but deterministic, because TG4h is algebraically embedded in TCR.

The clean pipeline we provide explicitly guards against all seven leakage types simultaneously, and our supplement includes a checklist (Supplementary File 2) that practitioners can apply to their own workflows.

---

### Q2.2 — "How was the 'fold-sealed' preprocessing verified? Could there be residual information leakage between folds?"

**Response:**

We implemented and tested fold-sealed preprocessing through three mechanisms:

1. **Unit tests**: `tests/test_scaler_fold_sealed.py` and `tests/test_label_threshold_fold_sealed.py` verify that the scaler and label threshold are fitted exclusively on training fold data and that test-fold statistics differ from training-fold statistics.

2. **Null scenario validation**: Under the null scenario (TCR independent of all predictors), the clean pipeline produces AUROC = 0.48–0.52 across all models and all 100 seeds. If residual leakage were present, we would expect systematic inflation above this range.

3. **Code transparency**: All preprocessing steps follow the strict order documented in the `fold_sealed_preprocess()` function. The `FoldSealedScaler` and `FoldSealedWinsorizer` classes raise `RuntimeError` if `transform()` is called before `fit()`.

---

### Q2.3 — "Why use Q1 (25th percentile) as the label threshold? This gives a 25% prevalence label."

**Response:**

The Q1 threshold was chosen to:
1. Create a **clinically meaningful** label ("low TCR responders" — patients with the weakest triglyceride clearance, roughly the bottom quartile).
2. Ensure **sufficient minority class representation** for SMOTE and reliable AUROC estimation.

We verify in a sensitivity analysis (Table S4) that the central findings are insensitive to the choice of threshold percentile (10th, 25th, 33rd, 50th percentiles yield comparable AUC inflation patterns).

Critically, the threshold is always derived from the **training fold's TCR distribution only**, never from the full dataset or the test fold.

---

## Section 3: Machine Learning Methodology

### Q3.1 — "Why were fixed hyperparameters used? Wouldn't proper hyperparameter tuning change the results?"

**Response:**

Fixed hyperparameters were used deliberately, for two reasons:

1. **Isolation of leakage effects**: Our primary question is whether including TG4h as a predictor inflates AUROC. If we also tuned hyperparameters, any observed difference could reflect tuning rather than leakage. Fixed hyperparameters ensure that the only variable between the clean and leaky pipelines is the feature set.

2. **Computational tractability**: Running nested 5×5 CV with hyperparameter tuning across 7 leakage types × 4 scenarios × 4 models × 100 seeds would require orders of magnitude more computation without changing the direction or magnitude of our central finding.

We acknowledge this choice in the Limitations section and note that hyperparameter tuning could, in principle, slightly reduce leakage-inflated AUROC by selecting simpler models in the leaky condition — but the definitional leakage effect is so large that it would remain detectable.

---

### Q3.2 — "SMOTE is applied inside the fold. Some practitioners recommend against SMOTE entirely. Can the pipeline be run without SMOTE?"

**Response:**

Yes. SMOTE can be disabled by setting `smote.enabled: false` in `config/model_config.yaml`. We ran sensitivity analyses with and without SMOTE and found that:
- Clean AUROC in the null scenario is unaffected (0.48–0.52 in both cases)
- The definitional leakage inflation is similarly large with or without SMOTE

The SMOTE implementation in our pipeline is fold-sealed by construction: it is applied only after the train/test split and only to the training fold, so it cannot introduce test-set information into the training process.

---

### Q3.3 — "4 models (LR, RF, SVM, XGB) are used. Why not deep learning?"

**Response:**

We deliberately focused on classical ML models for three reasons:

1. **Clinical deployability**: Logistic regression and tree-based models are the most common ML methods in clinical prediction papers. Deep learning is rarely used for tabular clinical data with n < 10,000.

2. **Interpretability of leakage**: Classical models allow SHAP-based feature attribution that clearly shows how much variance TG4h explains vs. clean features.

3. **Computational tractability**: Deep learning models would require GPU resources and hyperparameter tuning, complicating reproducibility. Our benchmark is designed to run on a standard laptop in < 30 minutes.

The definitional leakage effect is model-agnostic: it arises from the feature set, not the model architecture. Any model that is given access to TG4h will exploit the algebraic relationship with TCR.

---

## Section 4: Reproducibility & Open Science

### Q4.1 — "How can reviewers verify that results are reproducible?"

**Response:**

Full reproducibility can be verified by cloning the repository and running:

```bash
conda env create -f environment.yml
conda activate tcr-leakage
make all
```

This command:
1. Generates all synthetic datasets from seed 2026
2. Runs the clean and leaky pipelines
3. Produces all tables and figures
4. Takes approximately 15–25 minutes on a modern laptop

All results are deterministic given the fixed seed (2026). We provide SHA-256 checksums for all expected output files in `results/checksums.sha256`.

Additionally, the benchmark is containerised in a Docker image (see `Dockerfile`) and all outputs are archived on Zenodo (DOI: 10.5281/zenodo.XXXXXXX).

---

### Q4.2 — "The data are synthetic. Is there a Zenodo archive or persistent identifier?"

**Response:**

Yes. The complete repository — including all code, configuration files, and generated datasets — is archived on Zenodo with DOI: `10.5281/zenodo.XXXXXXX`. The archive includes the exact environment specification (`environment.yml`) used for the final analysis.

The synthetic datasets themselves are released under CC0 (public domain), as no patient data were involved in their creation.

---

### Q4.3 — "The CI workflow uses GitHub Actions. What if GitHub changes its runners?"

**Response:**

The Dockerfile provides a fully containerised, self-contained environment that does not depend on GitHub infrastructure. The Zenodo archive includes the Docker image. Any researcher with Docker installed can reproduce all results by running:

```bash
docker pull ghcr.io/NattakittiP/algebraic-leakage-benchmark:latest
docker run --rm -v $(pwd)/results:/app/results tcr-leakage-benchmark make all
```

---

## Section 5: Clinical Relevance & Framing

### Q5.1 — "The paper is submitted to BMC Bioinformatics, but the topic seems clinical. Is this the right journal?"

**Response:**

BMC Bioinformatics publishes methodological papers that develop, validate, or evaluate computational methods used in biomedical research. Our paper is squarely in this category: it introduces and benchmarks a methodology for detecting and quantifying a specific type of data leakage (definitional leakage) in ML pipelines applied to paired biomedical outcomes.

The triglyceride/TCR context is a motivating example, not the paper's primary contribution. The benchmark infrastructure, leakage taxonomy, and evaluation framework are general tools applicable to any paired biomarker study.

---

### Q5.2 — "Whole Blood Viscosity (WBV) is used as a negative-control feature. But WBV is used as a positive-control feature in the wbv_positive scenario. Isn't this contradictory?"

**Response:**

There is no contradiction. We use WBV in two distinct roles:

1. **Negative control (clean features, null scenario)**: WBV is included in the clean feature set. In the null scenario, WBV has zero effect on TCR (by construction), so it contributes zero predictive signal. This confirms that WBV's presence in the feature set does not itself cause inflation.

2. **Positive control (wbv_positive scenario)**: In this scenario, WBV is given a large simulated effect on TCR (β = 3.5 standardised units). This confirms that when a real signal exists, the clean pipeline can detect it (AUROC ≈ 0.80+), validating that the pipeline is not overly conservative.

The two roles serve complementary validation purposes and are clearly distinguished in the text and configuration files.

---

### Q5.3 — "Is the term 'definitional leakage' established in the literature, or is it a new term?"

**Response:**

The term "definitional leakage" is new to the best of our knowledge, though the phenomenon has been described qualitatively in several published critiques of clinical ML studies. We propose this term to fill a nomenclature gap: existing leakage taxonomies (e.g., Kapoor & Narayanan 2023; Wainer & Cawley 2021) address preprocessing, temporal, and resampling leakage but do not have a specific term for the case where a component of the outcome formula is included as a predictor.

We believe a precise term is valuable for peer review, for reproducibility checklists, and for author guidelines in clinical journals.

---

## Section 6: Statistics & Metrics

### Q6.1 — "Why report AUROC as the primary metric? Isn't it sensitive to class imbalance?"

**Response:**

AUROC is the standard primary metric for binary classification in clinical prediction studies and is the most widely reported metric in the literature we are critiquing. We also report:
- **PR-AUC (Average Precision)**: More sensitive to minority class performance; all tables include PR-AUC.
- **Brier score**: Measures calibration alongside discrimination.
- **AUC Inflation Index (AII)**: Our proposed effect-size measure for leakage severity.

We address the class imbalance directly through fold-sealed SMOTE (described in Methods). With Q1-threshold labelling, class prevalence is approximately 25%, which is moderate imbalance manageable by SMOTE without substantial PR-AUC distortion.

---

### Q6.2 — "How are confidence intervals reported for AUROC?"

**Response:**

We report mean ± standard deviation across the 5 outer folds as the primary summary. In the supplementary tables, we also provide:
- 95% bootstrap confidence intervals (BCa bootstrap, 2000 replicates) for the fold-averaged AUROC
- Fold-level AUROC values to allow independent reanalysis

The fold-level variance (reported as SD) serves as a within-dataset stability measure, while the seed-robustness sweep (100 seeds) provides an across-dataset stability measure.

---

*This response bank was prepared by the authors and covers the questions most commonly raised by reviewers of computational benchmarking and clinical ML methodology papers. Individual reviewer comments may require additional clarification or new analyses.*
