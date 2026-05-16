#!/usr/bin/env bash
# run_all.sh — One-command reproduction of all results.
#
# Usage:
#   bash run_all.sh           # full pipeline
#   bash run_all.sh --quick   # quick smoke test (5 seeds, n=300)
#
# Expected total runtime: ~30–60 minutes (full) | ~3–5 minutes (quick)
# Main analysis seed: 2026
# Development seed:   42

set -euo pipefail

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
    echo "=== QUICK MODE (n=300, 5 seeds) ==="
fi

N_MAIN=1500
N_SEEDS=100
if $QUICK; then
    N_MAIN=300
    N_SEEDS=5
fi

SEED_MAIN=2026
SEED_DEV=42

echo "============================================================"
echo "  TCR Leakage Benchmark — Full Reproduction"
echo "  Date: $(date)"
echo "  Seed: ${SEED_MAIN} (main), ${SEED_DEV} (dev)"
echo "  n = ${N_MAIN}, seeds = ${N_SEEDS}"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 1: Generate synthetic datasets
# ---------------------------------------------------------------------------
echo -e "\n[1/12] Generating synthetic datasets..."

python src/generate_synthetic_data.py \
    --config config/generator_null.yaml \
    --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out data/paired_tcr_null_v1_seed${SEED_MAIN}.csv

python src/generate_synthetic_data.py \
    --config config/generator_weak_signal.yaml \
    --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out data/paired_tcr_weak_signal_v1_seed${SEED_MAIN}.csv

python src/generate_synthetic_data.py \
    --config config/generator_moderate_signal.yaml \
    --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out data/paired_tcr_moderate_signal_v1_seed${SEED_MAIN}.csv

python src/generate_synthetic_data.py \
    --config config/generator_wbv_positive.yaml \
    --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out data/paired_tcr_wbv_positive_v1_seed${SEED_MAIN}.csv

# Domain shift dataset (different seed)
python src/generate_synthetic_data.py \
    --config config/generator_null.yaml \
    --seed 2027 --n ${N_MAIN} \
    --out data/paired_tcr_domain_shift_v1_seed2027.csv

echo "Datasets generated."

# ---------------------------------------------------------------------------
# STEP 2: Validate synthetic data
# ---------------------------------------------------------------------------
echo -e "\n[2/12] Validating synthetic data..."

python src/validate_synthetic_data.py \
    --data data/paired_tcr_null_v1_seed${SEED_MAIN}.csv \
    --scenario null \
    --out results/tables/validation_null.json \
    --figdir results/figures

echo "Validation complete."

# ---------------------------------------------------------------------------
# STEP 3: Run clean pipeline
# ---------------------------------------------------------------------------
echo -e "\n[3/12] Running clean (fold-sealed) pipeline..."

python src/run_clean_pipeline.py \
    --data data/paired_tcr_null_v1_seed${SEED_MAIN}.csv \
    --config config/model_config.yaml \
    --out results/tables/clean_results.csv \
    --seed ${SEED_MAIN}

echo "Clean pipeline complete. Expected AUROC ≈ 0.48–0.52."

# ---------------------------------------------------------------------------
# STEP 4: Run leakage scenarios
# ---------------------------------------------------------------------------
echo -e "\n[4/12] Running all 8 leakage scenarios..."

python src/run_leakage_scenarios.py \
    --data data/paired_tcr_null_v1_seed${SEED_MAIN}.csv \
    --config config/model_config.yaml \
    --out results/tables/leakage_benchmark.csv \
    --seed ${SEED_MAIN}

echo "Leakage benchmark complete. TG4h leakage expected AUROC > 0.90."

# ---------------------------------------------------------------------------
# STEP 5: SHAP analysis
# ---------------------------------------------------------------------------
echo -e "\n[5/12] Running SHAP analysis..."

python src/shap_analysis.py \
    --data data/paired_tcr_null_v1_seed${SEED_MAIN}.csv \
    --config config/model_config.yaml \
    --out results/tables/shap_comparison.csv \
    --figdir results/figures \
    --seed ${SEED_MAIN}

echo "SHAP analysis complete."

# ---------------------------------------------------------------------------
# STEP 6: Scenario sensitivity sweep
# ---------------------------------------------------------------------------
echo -e "\n[6/12] Running scenario × seed sensitivity sweep..."

SAMPLE_SIZE_FLAG=""
if ! $QUICK; then
    SAMPLE_SIZE_FLAG="--sample_size_sweep"
fi

python src/run_scenario_sensitivity.py \
    --configdir config/ \
    --model_config config/model_config.yaml \
    --out results/tables/scenario_sensitivity.csv \
    --seeds ${N_SEEDS} \
    --n ${N_MAIN} \
    ${SAMPLE_SIZE_FLAG}

echo "Sensitivity sweep complete."

# ---------------------------------------------------------------------------
# STEP 7: Domain shift analysis
# ---------------------------------------------------------------------------
echo -e "\n[7/12] Running domain shift stress test..."

python src/run_domain_shift.py \
    --train data/paired_tcr_null_v1_seed${SEED_MAIN}.csv \
    --shift_config config/generator_null.yaml \
    --shift_seed 2027 \
    --config config/model_config.yaml \
    --out results/tables/domain_shift.csv \
    --figdir results/figures

echo "Domain shift analysis complete."

# ---------------------------------------------------------------------------
# STEP 8: Generate all figures
# ---------------------------------------------------------------------------
echo -e "\n[8/12] Generating all manuscript figures..."

python src/plotting.py \
    --datadir data/ \
    --resultsdir results/tables/ \
    --figdir results/figures/

echo "All figures generated."

# ---------------------------------------------------------------------------
# STEP 9: Noise sensitivity (Plan item 37)
# ---------------------------------------------------------------------------
echo -e "\n[9/12] Running noise sensitivity analysis..."

NOISE_SEEDS=20
if $QUICK; then NOISE_SEEDS=3; fi

python src/run_noise_sensitivity.py \
    --config config/generator_null.yaml \
    --model_config config/model_config.yaml \
    --out results/tables/noise_sensitivity.csv \
    --figdir results/figures \
    --n ${N_MAIN} \
    --seeds ${NOISE_SEEDS}

echo "Noise sensitivity complete."

# ---------------------------------------------------------------------------
# STEP 10: Missingness sensitivity (Plan item 38)
# ---------------------------------------------------------------------------
echo -e "\n[10/12] Running missingness sensitivity analysis..."

MISS_SEEDS=20
if $QUICK; then MISS_SEEDS=3; fi

python src/run_missingness_sensitivity.py \
    --config config/generator_null.yaml \
    --model_config config/model_config.yaml \
    --out results/tables/missingness_sensitivity.csv \
    --figdir results/figures \
    --n ${N_MAIN} \
    --seeds ${MISS_SEEDS}

echo "Missingness sensitivity complete."

# ---------------------------------------------------------------------------
# STEP 11: Outlier / plausibility stress test (Plan item 39)
# ---------------------------------------------------------------------------
echo -e "\n[11/12] Running outlier / plausibility stress test..."

OUTLIER_SEEDS=15
if $QUICK; then OUTLIER_SEEDS=3; fi

python src/run_outlier_stress_test.py \
    --config config/generator_null.yaml \
    --model_config config/model_config.yaml \
    --out results/tables/outlier_stress_test.csv \
    --figdir results/figures \
    --n ${N_MAIN} \
    --seeds ${OUTLIER_SEEDS}

echo "Outlier stress test complete."

# ---------------------------------------------------------------------------
# STEP 12: Sample size sensitivity (Plan item 36)
# ---------------------------------------------------------------------------
echo -e "\n[12/12] Running sample size sensitivity analysis..."

SS_SEEDS=30
if $QUICK; then SS_SEEDS=5; fi

python src/run_sample_size_sensitivity.py \
    --configdir config/ \
    --model_config config/model_config.yaml \
    --out results/tables/sample_size_sensitivity.csv \
    --figdir results/figures \
    --seeds ${SS_SEEDS} \
    --n_jobs -1

echo "Sample size sensitivity complete."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Check results/tables/ for all benchmark tables"
echo "  Check results/figures/ for all manuscript figures"
echo ""
echo "  Key files:"
echo "    results/tables/clean_results.csv"
echo "    results/tables/leakage_benchmark.csv"
echo "    results/tables/scenario_sensitivity.csv"
echo "    results/tables/noise_sensitivity.csv"
echo "    results/tables/missingness_sensitivity.csv"
echo "    results/tables/outlier_stress_test.csv"
echo "    results/tables/sample_size_sensitivity.csv"
echo "    results/figures/fig3_leakage_benchmark.png"
echo "    results/figures/noise_sensitivity.png"
echo "    results/figures/missingness_sensitivity.png"
echo "    results/figures/outlier_stress_heatmap.png"
echo "    results/figures/sample_size_auroc_convergence.png"
echo "    results/figures/sample_size_precision_vs_n.png"
echo "    results/figures/sample_size_null_quality_vs_n.png"
echo "    results/figures/sample_size_leakage_vs_n.png"
echo "============================================================"
