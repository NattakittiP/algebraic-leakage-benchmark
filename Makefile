# Makefile — convenience targets for the TCR Leakage Benchmark

.PHONY: all minimal full test clean install noise missingness outlier sample-size sensitivity_all external-eicu external-mimic external-all

## Default: full pipeline
all: full

## Install dependencies
install:
	pip install -r requirements.txt

## Minimal run: null scenario only, dev seed 42, n=300
minimal:
	bash run_all.sh --quick

## Full run: all scenarios, main seed 2026, n=1500, 100 seeds robustness sweep
full:
	bash run_all.sh

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Generate data only (no ML)
data:
	python src/generate_synthetic_data.py --config config/generator_null.yaml \
	    --seed 2026 --n 1500 --out data/paired_tcr_null_v1_seed2026.csv
	python src/generate_synthetic_data.py --config config/generator_weak_signal.yaml \
	    --seed 2026 --n 1500 --out data/paired_tcr_weak_signal_v1_seed2026.csv
	python src/generate_synthetic_data.py --config config/generator_moderate_signal.yaml \
	    --seed 2026 --n 1500 --out data/paired_tcr_moderate_signal_v1_seed2026.csv
	python src/generate_synthetic_data.py --config config/generator_wbv_positive.yaml \
	    --seed 2026 --n 1500 --out data/paired_tcr_wbv_positive_v1_seed2026.csv

## Validate generated data
validate:
	python src/validate_synthetic_data.py \
	    --data data/paired_tcr_null_v1_seed2026.csv \
	    --scenario null --out results/tables/validation_null.json

## Clean generated outputs (keep source code)
clean:
	rm -f data/*.csv data/*.json
	rm -f results/tables/*.csv results/tables/*.json
	rm -f results/figures/*.png results/figures/*.pdf

## Noise sensitivity (Plan item 37)
noise:
	python src/run_noise_sensitivity.py \
	    --config config/generator_null.yaml \
	    --model_config config/model_config.yaml \
	    --out results/tables/noise_sensitivity.csv \
	    --figdir results/figures \
	    --n 1500 \
	    --seeds 20

## Noise sensitivity — quick smoke test (3 seeds)
noise-quick:
	python src/run_noise_sensitivity.py \
	    --config config/generator_null.yaml \
	    --model_config config/model_config.yaml \
	    --out results/tables/noise_sensitivity.csv \
	    --figdir results/figures \
	    --n 1500 \
	    --quick

## Missingness sensitivity (Plan item 38)
missingness:
	python src/run_missingness_sensitivity.py \
	    --config config/generator_null.yaml \
	    --model_config config/model_config.yaml \
	    --out results/tables/missingness_sensitivity.csv \
	    --figdir results/figures \
	    --n 1500 \
	    --seeds 20

## Missingness sensitivity — quick smoke test (3 seeds)
missingness-quick:
	python src/run_missingness_sensitivity.py \
	    --config config/generator_null.yaml \
	    --model_config config/model_config.yaml \
	    --out results/tables/missingness_sensitivity.csv \
	    --figdir results/figures \
	    --n 1500 \
	    --quick

## Outlier / plausibility stress test (Plan item 39)
outlier:
	python src/run_outlier_stress_test.py \
	    --config config/generator_null.yaml \
	    --model_config config/model_config.yaml \
	    --out results/tables/outlier_stress_test.csv \
	    --figdir results/figures \
	    --n 1500 \
	    --seeds 15

## Outlier stress test — quick smoke test (3 seeds)
outlier-quick:
	python src/run_outlier_stress_test.py \
	    --config config/generator_null.yaml \
	    --model_config config/model_config.yaml \
	    --out results/tables/outlier_stress_test.csv \
	    --figdir results/figures \
	    --n 1500 \
	    --quick

## Sample size sensitivity (Plan item 36) — 7 sizes × 4 scenarios × 30 seeds
sample-size:
	python src/run_sample_size_sensitivity.py \
	    --configdir config/ \
	    --model_config config/model_config.yaml \
	    --out results/tables/sample_size_sensitivity.csv \
	    --figdir results/figures \
	    --seeds 30 \
	    --n_jobs -1

## Sample size sensitivity — quick smoke test (5 seeds)
sample-size-quick:
	python src/run_sample_size_sensitivity.py \
	    --configdir config/ \
	    --model_config config/model_config.yaml \
	    --out results/tables/sample_size_sensitivity.csv \
	    --figdir results/figures \
	    --quick

## Run all four sensitivity analyses (full)
sensitivity_all: noise missingness outlier sample-size

## ─────────────────────────────────────────────────────────────────
## External Real-World Leakage Audits (§6 of paper)
## ─────────────────────────────────────────────────────────────────

## eICU external cohort (24h + 48h labels) — full 30-seed run
## Expected runtime: ~60–120 min on a standard laptop (193k rows, 4 models)
external-eicu:
	python src/run_external_eicu.py \
	    --data24 "External Cohort/Dataset/eicu_label24h.csv" \
	    --data48 "External Cohort/Dataset/eicu_label48h.csv" \
	    --outdir  results/tables \
	    --figdir  results/figures \
	    --n_seeds 30 \
	    --n_folds 5

## eICU quick smoke test (5 seeds, 24h only)
external-eicu-quick:
	python src/run_external_eicu.py \
	    --data24 "External Cohort/Dataset/eicu_label24h.csv" \
	    --data48 "External Cohort/Dataset/eicu_label48h.csv" \
	    --outdir  results/tables \
	    --figdir  results/figures \
	    --n_seeds 5 \
	    --n_folds 5 \
	    --skip48

## MIMIC all-admissions mortality — full 30-seed run
## Expected runtime: ~20–40 min (14k rows, 4 models)
external-mimic:
	python src/run_external_mimic.py \
	    --data   "External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv" \
	    --outdir  results/tables \
	    --figdir  results/figures \
	    --n_seeds 30 \
	    --n_folds 5

## MIMIC quick smoke test (5 seeds)
external-mimic-quick:
	python src/run_external_mimic.py \
	    --data   "External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv" \
	    --outdir  results/tables \
	    --figdir  results/figures \
	    --n_seeds 5 \
	    --n_folds 5

## Run both external cohorts sequentially (full)
external-all: external-eicu external-mimic

## Quick smoke test for both (use this first to verify environment)
external-quick: external-eicu-quick external-mimic-quick

## ─────────────────────────────────────────────────────────────────
## Calibration + SHAP supplementary analyses (§6)
## ─────────────────────────────────────────────────────────────────

## Calibration check on eICU 24h + MIMIC — reliability diagrams + ECE/slope table
external-calibration:
	python src/run_external_calibration.py \
	    --eicu24 "External Cohort/Dataset/eicu_label24h.csv" \
	    --mimic  "External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv" \
	    --outdir  results/tables \
	    --figdir  results/figures

## SHAP attribution analysis on eICU 24h — clean vs leaky_definitional
## Requires: pip install shap
## Expected runtime: ~5–15 min (SHAP on RF, 500 test samples)
external-shap:
	python src/run_external_shap.py \
	    --eicu24 "External Cohort/Dataset/eicu_label24h.csv" \
	    --outdir  results/tables \
	    --figdir  results/figures

## Run both supplementary analyses
external-supplement: external-calibration external-shap

## Docker: build image
docker-build:
	docker build -t tcr-benchmark:latest .

## Docker: run full pipeline
docker-run:
	docker run --rm -v $(PWD)/results:/workspace/results tcr-benchmark:latest
