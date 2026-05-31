# Algebraic Definitional Leakage in Paired Biomedical Prediction Models
## A Reproducible Audit Framework and Benchmark
## A Reproducible Synthetic-Data Benchmark

[![CI](https://github.com/NattakittiP/algebraic-leakage-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/NattakittiP/algebraic-leakage-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Reproducible](https://img.shields.io/badge/reproducibility-seeds%20fixed-green.svg)](#reproducibility)

---

## Overview

This repository provides a **fully reproducible synthetic-data benchmark** for evaluating how *definitional leakage* inflates machine-learning performance when predicting paired biomedical measurement outcomes.

The motivating use case is **triglyceride clearance rate (TCR)** — a phenotype derived from baseline (TG0h) and post-challenge (TG4h) triglyceride measurements:

```
TCR = (TG0h − TG4h) / TG0h × 100
```

Because TG4h is algebraically embedded in TCR, including TG4h as a predictor constitutes **definitional leakage**: the model exploits the outcome's own formula rather than learning clinically meaningful signal. This benchmark quantifies that inflation and tests eight leakage types under controlled simulation.

**Paper**: *"Algebraic Definitional Leakage in Paired Biomedical Prediction Models and a Reproducible Audit Framework"*

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/NattakittiP/algebraic-leakage-benchmark.git
cd tcr-leakage-benchmark
conda env create -f environment.yml
conda activate tcr-benchmark

# 2. Run the complete pipeline (one command)
make all
# or
bash run_all.sh
```

Expected runtime: **~90–180 minutes** (full) or **~5–10 minutes** (quick smoke test) on a modern laptop (4 CPU cores, 8 GB RAM).

```bash
# Quick smoke test only (~5–10 min, n=300, 5 seeds)
bash run_all.sh --quick
```

---

## Repository Structure

```
tcr-leakage-benchmark/
├── .github/
│   └── workflows/
│       └── ci.yml                              # GitHub Actions CI
├── config/
│   ├── generator_null.yaml                     # Null scenario parameters
│   ├── generator_weak_signal.yaml              # Weak signal parameters
│   ├── generator_moderate_signal.yaml          # Moderate signal parameters
│   ├── generator_wbv_positive.yaml             # WBV positive-control parameters
│   └── model_config.yaml                       # ML hyperparameters & CV settings
├── data/                                       # Generated synthetic datasets
│   ├── paired_tcr_null_v1_seed2026.csv
│   ├── paired_tcr_weak_signal_v1_seed2026.csv
│   ├── paired_tcr_moderate_signal_v1_seed2026.csv
│   ├── paired_tcr_wbv_positive_v1_seed2026.csv
│   └── paired_tcr_domain_shift_v1_seed2027.csv
├── notebooks/
│   ├── 01_generate_synthetic_data.ipynb
│   ├── 02_validate_generator.ipynb
│   ├── 03_clean_pipeline.ipynb
│   ├── 04_leakage_benchmark.ipynb
│   ├── 05_scenario_sensitivity.ipynb
│   └── 06_figures.ipynb
├── results/
│   ├── tables/                                 # CSV result tables (auto-generated)
│   └── figures/                                # Publication figures (auto-generated)
├── src/
│   ├── generate_synthetic_data.py              # Synthetic data generator
│   ├── validate_synthetic_data.py              # Data validation (14 checks)
│   ├── run_clean_pipeline.py                   # Fold-sealed clean ML pipeline
│   ├── run_leakage_scenarios.py                # All 8 leakage experiments
│   ├── run_unified_shap.py                     # Unified SHAP / ADI analysis
│   ├── run_cross_classifier_shap.py            # RF / LR / XGB SHAP comparison
│   ├── run_bootstrap_adi.py                    # Bootstrap CIs for ADI
│   ├── run_alternate_formula.py                # ATC additive-change benchmark
│   ├── run_ratio_formula.py                    # Ratio-formula benchmark
│   ├── run_scenario_sensitivity.py             # Scenario × seed sweep (S1)
│   ├── run_sample_size_sensitivity.py          # Sample-size sweep (S2)
│   ├── run_noise_sensitivity.py                # Noise sensitivity (S3)
│   ├── run_missingness_sensitivity.py          # Missingness sensitivity (S4)
│   ├── run_outlier_stress_test.py              # Outlier stress test (S5)
│   ├── run_domain_shift.py                     # Domain-shift stress test
│   ├── run_domain_shift_wbv.py                 # Domain-shift WBV comparison
│   ├── run_domain_shift_multiseed.py           # Domain-shift multiseed robustness
│   ├── build_cgmacros_cohort.py                # CGMacros meal-level cohort builder
│   ├── run_external_cgmacros.py                # CGMacros leakage audit
│   ├── run_cgmacros_shap.py                    # CGMacros SHAP attribution
│   ├── run_cgmacros_subject_uncertainty.py     # CGMacros LOSO + bootstrap
│   ├── icu_glucose_leakage_audit.py            # ICU glucose leakage audit
│   ├── run_external_eicu.py                    # eICU mortality leakage audit
│   ├── run_external_mimic.py                   # MIMIC-IV leakage audit
│   ├── run_external_calibration.py             # External calibration analysis
│   ├── run_external_shap.py                    # eICU SHAP attribution
│   ├── run_dca.py                              # Decision Curve Analysis
│   ├── plot_external_cohort.py                 # Combined external-cohort figures
│   ├── plotting.py                             # All manuscript figures
│   ├── calibration.py                          # Calibration metrics (library)
│   ├── metrics.py                              # AUC, PR-AUC, Brier score (library)
│   ├── shap_analysis.py                        # SHAP utilities (library)
│   └── utils.py                               # Preprocessing utilities (library)
├── supplement/
│   ├── benchmark_card.md                       # Benchmark card
│   ├── leakage_checklist.md                    # Leakage taxonomy checklist
│   ├── mathematical_appendix.md                # Mathematical derivations
│   ├── simulation_protocol.md                  # Simulation protocol
│   └── reviewer_response_bank.md              # Anticipated reviewer Q&A
├── tests/
│   ├── conftest.py
│   ├── test_generator_ranges.py
│   ├── test_label_threshold_fold_sealed.py
│   ├── test_no_tg4h_in_clean_predictors.py
│   ├── test_reproducibility_seed.py
│   ├── test_scaler_fold_sealed.py
│   └── test_wbv_formula.py
├── CITATION.cff
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── environment.yml
├── requirements.txt
└── run_all.sh
```

---

## Installation

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate tcr-benchmark
```

### Option B: pip

```bash
pip install -r requirements.txt
```

### Option C: Docker

```bash
docker build -t tcr-leakage .
docker run --rm -v $(pwd)/results:/workspace/results tcr-leakage bash run_all.sh
```

**Requirements**: Python ≥ 3.10, scikit-learn ≥ 1.3, imbalanced-learn ≥ 0.11, xgboost ≥ 2.0, shap ≥ 0.44, matplotlib ≥ 3.7, pandas ≥ 2.0, numpy ≥ 1.24, PyYAML ≥ 6.0.

---

## Pipeline Steps

The full pipeline is orchestrated by `run_all.sh` (29 steps). Key steps are documented below.

### Step 1: Generate Synthetic Data

```bash
python src/generate_synthetic_data.py \
    --config config/generator_null.yaml \
    --seed 2026 --n 1500 \
    --out data/paired_tcr_null_v1_seed2026.csv

python src/generate_synthetic_data.py \
    --config config/generator_weak_signal.yaml \
    --seed 2026 --n 1500 \
    --out data/paired_tcr_weak_signal_v1_seed2026.csv

python src/generate_synthetic_data.py \
    --config config/generator_moderate_signal.yaml \
    --seed 2026 --n 1500 \
    --out data/paired_tcr_moderate_signal_v1_seed2026.csv

python src/generate_synthetic_data.py \
    --config config/generator_wbv_positive.yaml \
    --seed 2026 --n 1500 \
    --out data/paired_tcr_wbv_positive_v1_seed2026.csv
```

### Step 2: Validate Generated Data

```bash
python src/validate_synthetic_data.py \
    --data data/paired_tcr_null_v1_seed2026.csv \
    --scenario null \
    --out results/tables/validation_null.json \
    --figdir results/figures
```

### Step 3: Run Clean (Leak-Free) Pipeline

```bash
python src/run_clean_pipeline.py \
    --data data/paired_tcr_null_v1_seed2026.csv \
    --config config/model_config.yaml \
    --out results/tables/clean_results.csv \
    --seed 2026
```

Expected output (null scenario): **AUROC ≈ 0.48–0.52** for all models.

### Step 4: Run Leakage Experiments

```bash
python src/run_leakage_scenarios.py \
    --data data/paired_tcr_null_v1_seed2026.csv \
    --config config/model_config.yaml \
    --out results/tables/leakage_benchmark.csv \
    --seed 2026
```

Leakage types tested (8 total):
- **L1 — Definitional leakage** (TG4h as predictor) — ΔAUC ≈ +0.4900–+0.5150
- **L2 — Target leakage** (TCR as predictor) — AUROC ≈ 1.0000
- **L3 — Global-scaler leakage** (StandardScaler fitted on full dataset)
- **L4 — Global-winsorisation leakage** (winsorisation bounds from full dataset)
- **L5 — Global-label-threshold leakage** (Q1 threshold from full dataset)
- **L6 — SMOTE-before-CV leakage** (synthetic oversampling before CV split)
- **L7 — Feature-selection leakage** (F-statistic selection on full dataset)
- **L8 — Combined leakage** (L1 + L3 + L4 + L6 simultaneously)

### Step 5: Scenario Sensitivity

```bash
python src/run_scenario_sensitivity.py \
    --configdir config/ \
    --model_config config/model_config.yaml \
    --out results/tables/scenario_sensitivity.csv \
    --seeds 100 \
    --n 1500
```

### Step 6: Domain Shift Stress Test

```bash
python src/run_domain_shift.py \
    --train data/paired_tcr_null_v1_seed2026.csv \
    --shift_config config/generator_null.yaml \
    --shift_seed 2027 \
    --config config/model_config.yaml \
    --out results/tables/domain_shift.csv \
    --figdir results/figures
```

### Step 7: SHAP Analysis

```bash
python src/run_unified_shap.py \
    --data data/paired_tcr_null_v1_seed2026.csv \
    --config config/model_config.yaml \
    --out_dir results/tables \
    --figdir results/figures \
    --seed 2026 \
    --sample_size 500
```

### Step 8: Generate All Manuscript Figures

```bash
python src/plotting.py \
    --datadir data/ \
    --resultsdir results/tables/ \
    --figdir results/figures/
```

---

## One-Command Reproduction

```bash
# Full pipeline (~90–180 min)
bash run_all.sh

# Quick smoke test (~5–10 min)
bash run_all.sh --quick
```

This runs all 29 steps in order and produces all tables and figures in `results/`.

---

## Reproducibility

### Random Seeds

| Seed | Purpose |
|------|---------|
| 42   | Development / unit tests |
| 2026 | **Main analysis** (primary paper results) |
| 2027 | Domain-shift external validation set |
| 1–100 | Robustness sweep across random seeds |

**Critical**: The label column `low_TCR` is **NOT** pre-computed in the CSV files. It is derived inside each training fold from Q1 of that fold's TCR distribution. This prevents label leakage across folds.

### Fold-Sealed Preprocessing Order

All preprocessing parameters are derived exclusively from the training fold:

1. **Winsorize** (1st–99th percentile) — fit on train, apply to test
2. **StandardScaler** — fit on train, apply to test
3. **Label threshold** (Q1 TCR) — computed from training fold TCR only
4. **Binarise** y_train using fold-derived threshold
5. **SMOTE** — applied to training fold only, after binarisation

### Cross-Validation

Nested 5×5 stratified k-fold cross-validation. Outer loop: performance estimation. Hyperparameters are fixed (no inner tuning loop) to isolate leakage effects from hyperparameter selection effects.

### Expected Results (Null Scenario, seed 2026)

| Model              | Clean AUROC | Definitional-Leaky AUROC | ΔAUC      |
|--------------------|-------------|--------------------------|-----------|
| LogisticRegression | 0.48–0.52   | ≈ 0.9995                 | ≈ +0.4910 |
| RandomForest       | 0.48–0.52   | ≈ 0.9870                 | ≈ +0.4940 |
| SVM                | 0.48–0.52   | ≈ 0.9890                 | ≈ +0.4900 |
| XGBoost            | 0.48–0.52   | ≈ 0.9970                 | ≈ +0.5150 |

---

## Simulation Scenarios

| Scenario | Signal | Expected Clean AUROC |
|----------|--------|----------------------|
| `null` | TCR independent of all predictors | ≈ 0.48–0.52 |
| `weak_signal` | Age, HDL, BMI small effects on TCR | ≈ 0.55–0.60 |
| `moderate_signal` | TG0h, BMI moderate effects on TCR | ≈ 0.65–0.75 |
| `wbv_positive` | WBV strong effect (β = 7.0) on TCR | ≈ 0.80+ |

---

## Variables

| Column | Description | Units |
|--------|-------------|-------|
| `record_id` | Synthetic record identifier | — |
| `age` | Patient age | years |
| `sex` | Sex (1 = male, 0 = female) | binary |
| `bmi` | Body Mass Index | kg/m² |
| `hct` | Haematocrit | % |
| `tp` | Total plasma protein | g/dL |
| `hdl` | HDL cholesterol | mg/dL |
| `ldl` | LDL cholesterol | mg/dL |
| `wbv` | Whole Blood Viscosity (de Simone formula) | mPa·s |
| `tg0h` | Baseline triglycerides | mg/dL |
| `tcr` | Triglyceride Clearance Rate (primary outcome) | % |
| `tg4h` | Post-challenge triglycerides (**leakage variable**) | mg/dL |

**WBV formula (de Simone 1990)**:
```
WBV = 0.12 × Hct + 0.17 × (TP − 2.07)
```

**Generation order** (critical — do not reverse):
1. Generate TCR as primary latent response
2. Derive TG4h = TG0h × (1 − TCR/100)

---

## Tests

```bash
python -m pytest tests/ -v --tb=short
```

Test coverage includes:
- Variable range validation (15 tests)
- WBV formula correctness (7 tests)
- Seed reproducibility (5 tests)
- Scaler fold-sealing (8 tests)
- Label threshold fold-sealing (4 tests)
- Clean predictor list integrity — no TG4h, no TCR (6 tests)

**All 45 tests pass** on Python 3.10+.

---

## Citing This Work

```bibtex
@article{tcr-leakage-benchmark-2026,
  title   = {Algebraic Definitional Leakage in Paired Biomedical Prediction Models
             and a Reproducible Audit Framework},
  year    = {2026}
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The synthetic data and all generated outputs are released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication).

---

## Contributing

Bug reports and pull requests are welcome. Please open an issue first to discuss proposed changes. All contributions must pass `python -m pytest tests/` and `make lint`.

---

## Contact

For questions about methodology, open an issue or contact the corresponding author via the journal submission system.
