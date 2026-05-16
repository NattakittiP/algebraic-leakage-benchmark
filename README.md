# Definitional Leakage in ML Prediction of Paired Triglyceride-Response Phenotypes
## A Reproducible Synthetic-Data Benchmark
---

## Overview

This repository provides a **fully reproducible synthetic-data benchmark** for evaluating how *definitional leakage* inflates machine-learning performance when predicting paired biomedical measurement outcomes.

The motivating use case is **triglyceride clearance rate (TCR)** — a phenotype derived from baseline (TG0h) and post-challenge (TG4h) triglyceride measurements:

```
TCR = (TG0h − TG4h) / TG0h × 100
```

Because TG4h is algebraically embedded in TCR, including TG4h as a predictor constitutes **definitional leakage**: the model exploits the outcome's own formula rather than learning clinically meaningful signal. This benchmark quantifies that inflation and tests seven leakage types under controlled simulation.

**Paper**: *"Definitional leakage in machine-learning prediction of paired triglyceride-response phenotypes: a reproducible synthetic-data benchmark"* — submitted to BMC Bioinformatics.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/your-org/tcr-leakage-benchmark.git
cd tcr-leakage-benchmark
conda env create -f environment.yml
conda activate tcr-leakage

# 2. Run the complete pipeline (one command)
make all
# or
bash run_all.sh
```

Expected runtime: **~15–25 minutes** on a modern laptop (4 CPU cores, 8 GB RAM).

---

## Repository Structure

```
tcr-leakage-benchmark/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI
├── config/
│   ├── generator_null.yaml           # Null scenario parameters
│   ├── generator_weak_signal.yaml    # Weak signal parameters
│   ├── generator_moderate_signal.yaml # Moderate signal parameters
│   ├── generator_wbv_positive.yaml   # WBV positive-control parameters
│   └── model_config.yaml             # ML hyperparameters
├── data/                             # Generated synthetic datasets (git-ignored except .gitkeep)
│   ├── paired_tcr_null_v1_seed2026.csv
│   ├── paired_tcr_null_v1_seed2026.json
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
│   ├── tables/                       # CSV result tables (auto-generated)
│   └── figures/                      # Publication figures (auto-generated)
├── src/
│   ├── generate_synthetic_data.py    # Synthetic data generator
│   ├── run_clean_pipeline.py         # Fold-sealed clean ML pipeline
│   ├── run_leakage_scenarios.py      # Leakage experiments
│   ├── run_scenario_sensitivity.py   # Scenario comparisons
│   ├── run_domain_shift.py           # Domain shift stress test
│   ├── calibration.py                # Calibration metrics
│   ├── metrics.py                    # AUC, PR-AUC, Brier score
│   ├── shap_analysis.py              # SHAP & permutation importance
│   ├── plotting.py                   # Figure generation
│   ├── utils.py                      # Preprocessing utilities
│   └── validate_synthetic_data.py    # Data validation checks
├── supplement/
│   ├── benchmark_card.md             # Benchmark card (item 42)
│   ├── leakage_checklist.md          # Leakage taxonomy checklist
│   ├── mathematical_appendix.md      # Mathematical derivations
│   ├── simulation_protocol.md        # Simulation protocol
│   └── reviewer_response_bank.md     # Anticipated reviewer Q&A
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
conda activate tcr-leakage
```

### Option B: pip

```bash
pip install -r requirements.txt
```

### Option C: Docker

```bash
docker build -t tcr-leakage .
docker run --rm -v $(pwd)/results:/app/results tcr-leakage make all
```

**Requirements**: Python ≥ 3.10, scikit-learn ≥ 1.3, imbalanced-learn ≥ 0.11, xgboost ≥ 2.0, shap ≥ 0.44, matplotlib ≥ 3.7, pandas ≥ 2.0, numpy ≥ 1.24, PyYAML ≥ 6.0.

---

## Pipeline Steps

### Step 1: Generate Synthetic Data

```bash
# Main analysis dataset (seed 2026)
python src/generate_synthetic_data.py \
    --config config/generator_null.yaml \
    --seed 2026 --n 1500 \
    --out data/paired_tcr_null_v1_seed2026.csv

# Additional scenario datasets
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
    --data data/paired_tcr_null_v1_seed2026.csv
```

### Step 3: Run Clean (Leak-Free) Pipeline

```bash
python src/run_clean_pipeline.py \
    --data data/paired_tcr_null_v1_seed2026.csv \
    --config config/model_config.yaml \
    --out results/tables/clean_results.csv
```

Expected output (null scenario): **AUROC ≈ 0.48–0.52** for all models.

### Step 4: Run Leakage Experiments

```bash
python src/run_leakage_scenarios.py \
    --data data/paired_tcr_null_v1_seed2026.csv \
    --config config/model_config.yaml \
    --out results/tables/leakage_results.csv
```

Leakage types tested:
- **Definitional leakage** (TG4h as predictor) — expected AUROC inflation ≈ +0.30–0.45
- **Outcome leakage** (TCR as predictor) — expected AUROC ≈ 1.00
- **Preprocessing leakage** (scaler fitted on full dataset)
- **Resampling leakage** (SMOTE before CV split)
- **Label leakage** (threshold from full dataset)
- **Feature-selection leakage** (feature selection before CV split)
- **Calibration leakage** (calibration on training data)

### Step 5: Scenario Sensitivity

```bash
python src/run_scenario_sensitivity.py \
    --config config/model_config.yaml \
    --out results/tables/scenario_sensitivity.csv
```

### Step 6: Domain Shift Stress Test

```bash
python src/run_domain_shift.py \
    --config config/model_config.yaml \
    --out results/tables/domain_shift.csv
```

### Step 7: SHAP Analysis

```bash
python src/shap_analysis.py \
    --data data/paired_tcr_null_v1_seed2026.csv \
    --config config/model_config.yaml \
    --out results/figures/shap_summary.png
```

### Step 8: Generate Figures

```bash
python src/plotting.py \
    --tables results/tables/ \
    --out results/figures/
```

---

## One-Command Reproduction

```bash
make all
```

This runs all steps in order and produces all tables and figures in `results/`.

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

| Model              | Clean AUROC | Definitional-Leaky AUROC | Inflation |
|--------------------|-------------|--------------------------|-----------|
| LogisticRegression | 0.48–0.52   | 0.80–0.95                | +0.30–0.45|
| RandomForest       | 0.48–0.52   | 0.85–0.98                | +0.35–0.48|
| SVM                | 0.48–0.52   | 0.78–0.92                | +0.28–0.42|
| XGBoost            | 0.48–0.52   | 0.85–0.97                | +0.35–0.47|

---

## Simulation Scenarios

| Scenario | Signal | Expected Clean AUROC |
|----------|--------|----------------------|
| `null` | TCR independent of all predictors | ≈ 0.48–0.52 |
| `weak_signal` | Age, HDL, BMI small effects on TCR | ≈ 0.55–0.60 |
| `moderate_signal` | TG0h, BMI moderate effects on TCR | ≈ 0.65–0.75 |
| `wbv_positive` | WBV strong effect (β = 3.5) on TCR | ≈ 0.80+ |

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
pytest tests/ -v
```

Test coverage includes:
- Variable range validation
- WBV formula correctness
- Seed reproducibility
- Scaler fold-sealing
- Label threshold fold-sealing
- Clean predictor list integrity (no TG4h, no TCR)

---

## Contributing

Bug reports and pull requests are welcome. Please open an issue first to discuss proposed changes. All contributions must pass `pytest tests/` and `make lint`.

---

## Contact

For questions about methodology, open an issue or contact the corresponding author via the journal submission system.
