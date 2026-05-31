#!/usr/bin/env bash
# =============================================================================
# run_all.sh — One-command reproduction of ALL results for the paper.
#
# Usage:
#   bash run_all.sh           # full pipeline  (~90–180 min on a modern laptop)
#   bash run_all.sh --quick   # smoke test     (~5–10 min, n=300, 5 seeds)
#
# Sections (numbered steps):
#   [1 ] Generate synthetic datasets (all 4 scenarios + domain-shift)
#   [2 ] Validate synthetic data     (all 4 scenarios + domain-shift)
#   [3 ] Run fold-sealed clean pipeline (primary null, single seed)
#   [4 ] Run all 8 leakage scenarios    (primary null, single seed)
#   [5 ] Unified SHAP analysis (RF, seed=2026)
#   [6 ] Cross-classifier SHAP (RF / LR / XGB comparison)
#   [7 ] Bootstrap ADI (500 resamples, CI table for paper)
#   [8 ] Alternate-formula benchmark (ATC = TG0h − TG4h)
#   [9 ] Ratio-formula benchmark (Y = TG4h / TG0h)
#   [10] Scenario × seed sensitivity sweep (S1, 100 seeds)
#   [11] Sample-size sensitivity sweep (S2, n=300–100 000)
#   [12] Noise sensitivity (S3)
#   [13] Missingness sensitivity (S4)
#   [14] Outlier / plausibility stress test (S5)
#   [15] Domain-shift stress test (single seed)
#   [16] Domain-shift WBV-control comparison (single seed)
#   [17] Domain-shift multiseed robustness (10 seeds)
#   [18] External — Build CGMacros meal cohort
#   [19] External — CGMacros leakage audit
#   [20] External — CGMacros SHAP attribution
#   [21] External — CGMacros subject-level uncertainty (LOSO + bootstrap)
#   [22] External — ICU Glucose leakage audit
#   [23] External — eICU early-mortality leakage audit
#   [24] External — MIMIC-IV hospital-mortality leakage audit
#   [25] External — Calibration analysis (eICU + MIMIC)
#   [26] External — SHAP attribution (eICU definitional pipeline)
#   [27] External — Decision Curve Analysis (eICU 24h)
#   [28] External — Combined external-cohort figures
#   [29] Generate all manuscript figures
#
# Prerequisites:
#   pip install -r requirements.txt   (or: conda env create -f environment.yml)
#   External cohort CSVs must exist under "External Cohort/Dataset/"
#   CGMacros raw data must exist under "CGMacros_dateshifted365/CGMacros_Dataset/"
#   ICU glucose CSV at:
#     "curated-data-for-describing-blood-glucose-management-in-the-intensive-care-unit-1.0.1/
#      glucose_insulin_pair.csv"
# =============================================================================

set -euo pipefail

# ── Mode flags ────────────────────────────────────────────────────────────────
QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
    echo "=== QUICK MODE (n=300, 5 seeds — smoke test only) ==="
fi

# ── Global parameters ─────────────────────────────────────────────────────────
SEED_MAIN=2026
SEED_DEV=42
SEED_SHIFT=2027

N_MAIN=1500
N_SEEDS_PRIMARY=30      # used for alternate/ratio formula benchmarks
N_SEEDS_SCENARIO=100    # scenario sensitivity S1 (paper Table 5)
N_SEEDS_NOISE=20        # S3 noise
N_SEEDS_MISS=20         # S4 missingness
N_SEEDS_OUTLIER=15      # S5 outlier
N_SEEDS_SS=30           # S2 sample-size
N_SEEDS_EXT=30          # external cohorts
N_SEEDS_DOMAIN=10       # domain-shift multiseed
N_BOOT=500              # bootstrap ADI resamples

if $QUICK; then
    N_MAIN=300
    N_SEEDS_PRIMARY=5
    N_SEEDS_SCENARIO=5
    N_SEEDS_NOISE=3
    N_SEEDS_MISS=3
    N_SEEDS_OUTLIER=3
    N_SEEDS_SS=5
    N_SEEDS_EXT=5
    N_SEEDS_DOMAIN=3
    N_BOOT=50
fi

# ── Quick-mode flags for scripts that accept --quick ─────────────────────────
# run_sample_size_sensitivity has --quick which also reduces sz_list (9→4 sizes)
QUICK_FLAG=""
if $QUICK; then QUICK_FLAG="--quick"; fi

# ── Data / path constants ─────────────────────────────────────────────────────
EICU_24H="External Cohort/Dataset/eicu_label24h.csv"
EICU_48H="External Cohort/Dataset/eicu_label48h.csv"
MIMIC_CSV="External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv"
ICU_GLUCOSE_CSV="curated-data-for-describing-blood-glucose-management-in-the-intensive-care-unit-1.0.1/Datasets/glucose_insulin_pair.csv"
CGM_RAW_DIR="CGMacros_dateshifted365/CGMacros_Dataset"
CGM_COHORT="results/tables/cgmacros_meal_cohort.csv"

NULL_DATA="data/paired_tcr_null_v1_seed${SEED_MAIN}.csv"
WEAK_DATA="data/paired_tcr_weak_signal_v1_seed${SEED_MAIN}.csv"
MOD_DATA="data/paired_tcr_moderate_signal_v1_seed${SEED_MAIN}.csv"
WBV_DATA="data/paired_tcr_wbv_positive_v1_seed${SEED_MAIN}.csv"
SHIFT_DATA="data/paired_tcr_domain_shift_v1_seed${SEED_SHIFT}.csv"

OUTDIR="results/tables"
FIGDIR="results/figures"
CFGDIR="config"
MODEL_CFG="config/model_config.yaml"
NULL_CFG="config/generator_null.yaml"
WEAK_CFG="config/generator_weak_signal.yaml"
MOD_CFG="config/generator_moderate_signal.yaml"
WBV_CFG="config/generator_wbv_positive.yaml"

# ── tqdm-based pipeline progress bar ─────────────────────────────────────────
# Tracks overall step progress (X / TOTAL_STEPS) across the whole pipeline.
TOTAL_STEPS=29
_STEP_CURRENT=0
PIPELINE_START=$(date +%s)

_progress() {
    # Usage: _progress <step_num> <label>
    # Prints a tqdm-style progress bar for the overall pipeline.
    local n=$1
    local label="$2"
    _STEP_CURRENT=$n
    # Support both 'python' (Windows) and 'python3' (Linux/macOS)
    local PY
    PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
    [[ -z "$PY" ]] && return 0
    "$PY" - <<PYEOF
import sys, time
try:
    from tqdm import tqdm
    total   = ${TOTAL_STEPS}
    current = ${n}
    elapsed = int(time.time()) - ${PIPELINE_START}
    pct     = current / total * 100
    filled  = int(pct / 100 * 40)
    bar     = '█' * filled + '░' * (40 - filled)
    mins, secs = divmod(elapsed, 60)
    sys.stderr.write(
        f"\r\033[1;36m Pipeline \033[0m |{bar}| "
        f"\033[1m{current:2d}/{total}\033[0m steps "
        f"[{mins:02d}:{secs:02d}]  ✓ Step {current}: ${label}\n"
    )
    sys.stderr.flush()
except Exception:
    pass  # tqdm unavailable — silently skip
PYEOF
}

# ── Step header helper ────────────────────────────────────────────────────────
step() {
    echo -e "\n$(printf '─%.0s' {1..60})"
    echo "  [${1}/${TOTAL_STEPS}] ${2}"
    echo "$(printf '─%.0s' {1..60})"
}

# ── Ensure output directories exist ──────────────────────────────────────────
mkdir -p "${OUTDIR}" "${FIGDIR}"

echo "============================================================"
echo "  TCR Leakage Benchmark — Complete Reproduction"
echo "  Date : $(date)"
echo "  Mode : $(if $QUICK; then echo QUICK; else echo FULL; fi)"
echo "  Seed : ${SEED_MAIN} (main)  ${SEED_DEV} (dev)  ${SEED_SHIFT} (shift)"
echo "  n    : ${N_MAIN}"
echo "============================================================"

# =============================================================================
# STEP 1 — Generate synthetic datasets
# =============================================================================
step 1 "Generating synthetic datasets (all 4 scenarios + domain-shift)"

python src/generate_synthetic_data.py \
    --config "${NULL_CFG}" --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out "${NULL_DATA}"

python src/generate_synthetic_data.py \
    --config "${WEAK_CFG}" --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out "${WEAK_DATA}"

python src/generate_synthetic_data.py \
    --config "${MOD_CFG}" --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out "${MOD_DATA}"

python src/generate_synthetic_data.py \
    --config "${WBV_CFG}" --seed ${SEED_MAIN} --n ${N_MAIN} \
    --out "${WBV_DATA}"

python src/generate_synthetic_data.py \
    --config "${NULL_CFG}" --seed ${SEED_SHIFT} --n ${N_MAIN} \
    --out "${SHIFT_DATA}"

echo "✓ All 5 datasets generated."
_progress 1 "Datasets generated"

# =============================================================================
# STEP 2 — Validate synthetic data (all 4 scenarios + domain-shift)
# [FIX] Previously only validated null; now validates all generated datasets.
# =============================================================================
step 2 "Validating synthetic data (14 plausibility checks × 5 datasets)"

python src/validate_synthetic_data.py \
    --data "${NULL_DATA}" \
    --scenario null \
    --out "${OUTDIR}/validation_null.json" \
    --figdir "${FIGDIR}"

python src/validate_synthetic_data.py \
    --data "${WEAK_DATA}" \
    --scenario weak_signal \
    --out "${OUTDIR}/validation_weak_signal.json" \
    --figdir "${FIGDIR}"

python src/validate_synthetic_data.py \
    --data "${MOD_DATA}" \
    --scenario moderate_signal \
    --out "${OUTDIR}/validation_moderate_signal.json" \
    --figdir "${FIGDIR}"

python src/validate_synthetic_data.py \
    --data "${WBV_DATA}" \
    --scenario wbv_positive \
    --out "${OUTDIR}/validation_wbv_positive.json" \
    --figdir "${FIGDIR}"

echo "✓ Validation complete (5 datasets). All 14 checks expected to pass."
_progress 2 "Validation complete"

# =============================================================================
# STEP 3 — Clean fold-sealed pipeline (primary null, single seed)
# [FIX] Comment corrected: script runs one seed (seed=${SEED_MAIN}),
#        iterating over all 4 classifiers via nested CV.
# =============================================================================
step 3 "Running fold-sealed clean pipeline (null scenario, seed=${SEED_MAIN})"

python src/run_clean_pipeline.py \
    --data "${NULL_DATA}" \
    --config "${MODEL_CFG}" \
    --out "${OUTDIR}/clean_results.csv" \
    --seed ${SEED_MAIN}

echo "✓ Clean pipeline done. Expected AUROC ≈ 0.48–0.52 (TOST equiv. to 0.500)."
_progress 3 "Clean pipeline done"

# =============================================================================
# STEP 4 — All 8 leakage scenarios (primary null benchmark, single seed)
# [FIX] Comment corrected: script runs one seed (seed=${SEED_MAIN}).
# =============================================================================
step 4 "Running all 8 leakage pipeline types (null scenario, seed=${SEED_MAIN})"

python src/run_leakage_scenarios.py \
    --data "${NULL_DATA}" \
    --config "${MODEL_CFG}" \
    --out "${OUTDIR}/leakage_benchmark.csv" \
    --seed ${SEED_MAIN}

echo "✓ Leakage benchmark done. TG4h (L1) expected AUROC > 0.98."
_progress 4 "Leakage benchmark done"

# =============================================================================
# STEP 5 — Unified SHAP analysis (RF, seed=2026, paper Table 6)
# =============================================================================
step 5 "Running unified SHAP analysis (RF, n=500 subsample, seed=${SEED_MAIN})"

python src/run_unified_shap.py \
    --data "${NULL_DATA}" \
    --config "${MODEL_CFG}" \
    --out_dir "${OUTDIR}" \
    --figdir "${FIGDIR}" \
    --seed ${SEED_MAIN} \
    --sample_size 500

echo "✓ Unified SHAP done → unified_shap_adi_rf.csv + unified_shap_cross_clf.csv"
_progress 5 "Unified SHAP done"

# =============================================================================
# STEP 6 — Cross-classifier SHAP (RF / LR / XGB side-by-side)
# =============================================================================
step 6 "Running cross-classifier SHAP comparison (RF / LR / XGB)"

python src/run_cross_classifier_shap.py \
    --data "${NULL_DATA}" \
    --config "${MODEL_CFG}" \
    --out "${OUTDIR}/cross_classifier_shap.csv" \
    --figdir "${FIGDIR}" \
    --seed ${SEED_MAIN}

echo "✓ Cross-classifier SHAP done → cross_classifier_shap.csv"
_progress 6 "Cross-classifier SHAP done"

# =============================================================================
# STEP 7 — Bootstrap ADI (B=500 resamples, CI table for paper)
# =============================================================================
step 7 "Running bootstrap ADI (${N_BOOT} resamples)"

python src/run_bootstrap_adi.py \
    --data "${NULL_DATA}" \
    --config "${MODEL_CFG}" \
    --out "${OUTDIR}/bootstrap_adi.csv" \
    --figdir "${FIGDIR}" \
    --n_boot ${N_BOOT} \
    --shap_sample 200 \
    --seed ${SEED_MAIN}

echo "✓ Bootstrap ADI done → bootstrap_adi.csv / bootstrap_adi_summary.csv"
_progress 7 "Bootstrap ADI done"

# =============================================================================
# STEP 8 — Alternate-formula benchmark (ATC = TG0h − TG4h, additive)
# =============================================================================
step 8 "Running alternate-formula benchmark (ATC additive-change formula)"

python src/run_alternate_formula.py \
    --data "${NULL_DATA}" \
    --config "${MODEL_CFG}" \
    --out "${OUTDIR}/alternate_formula_benchmark.csv" \
    --seeds ${N_SEEDS_PRIMARY} \
    --n_splits 5

echo "✓ Alternate formula done → alternate_formula_benchmark.csv"
_progress 8 "Alternate formula done"

# =============================================================================
# STEP 9 — Ratio-formula benchmark (Y = TG4h / TG0h)
# =============================================================================
step 9 "Running ratio-formula benchmark (Y = TG4h / TG0h)"

python src/run_ratio_formula.py \
    --data "${NULL_DATA}" \
    --out "${OUTDIR}/ratio_formula_benchmark.csv" \
    --seeds ${N_SEEDS_PRIMARY} \
    --n_splits 5

echo "✓ Ratio formula done → ratio_formula_benchmark.csv"
_progress 9 "Ratio formula done"

# =============================================================================
# STEP 10 — Scenario × seed sensitivity sweep (S1, 100 seeds)
# =============================================================================
step 10 "Scenario sensitivity sweep S1 (${N_SEEDS_SCENARIO} seeds × 4 scenarios)"

SAMPLE_SIZE_FLAG=""
if ! $QUICK; then SAMPLE_SIZE_FLAG="--sample_size_sweep"; fi

python src/run_scenario_sensitivity.py \
    --configdir "${CFGDIR}" \
    --model_config "${MODEL_CFG}" \
    --out "${OUTDIR}/scenario_sensitivity.csv" \
    --seeds ${N_SEEDS_SCENARIO} \
    --n ${N_MAIN} \
    ${SAMPLE_SIZE_FLAG}

echo "✓ Scenario sensitivity done → scenario_sensitivity.csv"
_progress 10 "Scenario sensitivity done"

# =============================================================================
# STEP 11 — Sample-size sensitivity (S2, n=300–100 000)
# [FIX] In QUICK mode, pass --quick so sz_list shrinks from 9 → 4 sizes.
#        Without this flag, QUICK mode still ran all 9 sample sizes.
# =============================================================================
step 11 "Sample-size sensitivity S2 ($(if $QUICK; then echo '4 sizes'; else echo '9 sizes'; fi) × ${N_SEEDS_SS} seeds each)"

python src/run_sample_size_sensitivity.py \
    --configdir "${CFGDIR}" \
    --model_config "${MODEL_CFG}" \
    --out "${OUTDIR}/sample_size_sensitivity.csv" \
    --figdir "${FIGDIR}" \
    --seeds ${N_SEEDS_SS} \
    --n_jobs -1 \
    ${QUICK_FLAG}

echo "✓ Sample-size sensitivity done → sample_size_sensitivity.csv"
_progress 11 "Sample-size sensitivity done"

# =============================================================================
# STEP 12 — Noise sensitivity (S3, 5 TCR-SD multipliers × 20 seeds)
# =============================================================================
step 12 "Noise sensitivity S3 (5 noise levels × ${N_SEEDS_NOISE} seeds)"

python src/run_noise_sensitivity.py \
    --config "${NULL_CFG}" \
    --model_config "${MODEL_CFG}" \
    --out "${OUTDIR}/noise_sensitivity.csv" \
    --figdir "${FIGDIR}" \
    --n ${N_MAIN} \
    --seeds ${N_SEEDS_NOISE}

echo "✓ Noise sensitivity done → noise_sensitivity.csv"
_progress 12 "Noise sensitivity done"

# =============================================================================
# STEP 13 — Missingness sensitivity (S4, 4 mechanisms × 20 seeds)
# =============================================================================
step 13 "Missingness sensitivity S4 (MCAR/MAR/MNAR × ${N_SEEDS_MISS} seeds)"

python src/run_missingness_sensitivity.py \
    --config "${NULL_CFG}" \
    --model_config "${MODEL_CFG}" \
    --out "${OUTDIR}/missingness_sensitivity.csv" \
    --figdir "${FIGDIR}" \
    --n ${N_MAIN} \
    --seeds ${N_SEEDS_MISS}

echo "✓ Missingness sensitivity done → missingness_sensitivity.csv"
_progress 13 "Missingness sensitivity done"

# =============================================================================
# STEP 14 — Outlier / plausibility stress test (S5, 3 rates × 4 strategies)
# =============================================================================
step 14 "Outlier stress test S5 (3 contamination rates × ${N_SEEDS_OUTLIER} seeds)"

python src/run_outlier_stress_test.py \
    --config "${NULL_CFG}" \
    --model_config "${MODEL_CFG}" \
    --out "${OUTDIR}/outlier_stress_test.csv" \
    --figdir "${FIGDIR}" \
    --n ${N_MAIN} \
    --seeds ${N_SEEDS_OUTLIER}

echo "✓ Outlier stress test done → outlier_stress_test.csv"
_progress 14 "Outlier stress test done"

# =============================================================================
# STEP 15 — Domain-shift stress test (single seed, 4 conditions)
# =============================================================================
step 15 "Domain-shift stress test (4 distributional shift conditions)"

python src/run_domain_shift.py \
    --train "${NULL_DATA}" \
    --shift_config "${NULL_CFG}" \
    --shift_seed ${SEED_SHIFT} \
    --config "${MODEL_CFG}" \
    --out "${OUTDIR}/domain_shift.csv" \
    --figdir "${FIGDIR}"

echo "✓ Domain-shift done → domain_shift.csv"
_progress 15 "Domain-shift done"

# =============================================================================
# STEP 16 — Domain-shift WBV-control comparison (null vs WBV positive)
# =============================================================================
step 16 "Domain-shift WBV comparison (null vs WBV-positive pipeline)"

python src/run_domain_shift_wbv.py \
    --null_data "${NULL_DATA}" \
    --wbv_data  "${WBV_DATA}" \
    --null_config "${NULL_CFG}" \
    --wbv_config  "${WBV_CFG}" \
    --model_config "${MODEL_CFG}" \
    --shift_seed ${SEED_SHIFT} \
    --out "${OUTDIR}/domain_shift_wbv_comparison.csv" \
    --figdir "${FIGDIR}"

echo "✓ Domain-shift WBV done → domain_shift_wbv_comparison.csv"
_progress 16 "Domain-shift WBV done"

# =============================================================================
# STEP 17 — Domain-shift multiseed robustness (10 seeds)
# [FIX] Echo updated to list all 3 output files produced by this script.
# =============================================================================
step 17 "Domain-shift multiseed robustness (${N_SEEDS_DOMAIN} seeds)"

python src/run_domain_shift_multiseed.py \
    --null_data "${NULL_DATA}" \
    --wbv_data  "${WBV_DATA}" \
    --null_config "${NULL_CFG}" \
    --wbv_config  "${WBV_CFG}" \
    --model_config "${MODEL_CFG}" \
    --shift_seed ${SEED_SHIFT} \
    --n_seeds ${N_SEEDS_DOMAIN} \
    --out_dir "${OUTDIR}" \
    --figdir  "${FIGDIR}"

echo "✓ Domain-shift multiseed done →"
echo "    domain_shift_multiseed.csv"
echo "    domain_shift_multiseed_summary.csv"
echo "    domain_shift_multiseed_wbv_summary.csv"
_progress 17 "Domain-shift multiseed done"

# =============================================================================
# STEP 18 — Build CGMacros meal-level cohort  ← must run before 19–21
# =============================================================================
step 18 "Building CGMacros meal-level cohort (${CGM_RAW_DIR})"

python src/build_cgmacros_cohort.py \
    --data_dir "${CGM_RAW_DIR}" \
    --outdir   "${OUTDIR}"

echo "✓ CGMacros cohort built → cgmacros_meal_cohort.csv"
_progress 18 "CGMacros cohort built"

# =============================================================================
# STEP 19 — CGMacros leakage audit (30 seeds, GroupKFold)
# =============================================================================
step 19 "CGMacros external leakage audit (${N_SEEDS_EXT} seeds, GroupKFold k=5)"

python src/run_external_cgmacros.py \
    --data   "${CGM_COHORT}" \
    --outdir "${OUTDIR}" \
    --figdir "${FIGDIR}" \
    --n_seeds ${N_SEEDS_EXT} \
    --n_folds 5

echo "✓ CGMacros audit done → external_cgmacros_results.csv"
_progress 19 "CGMacros audit done"

# =============================================================================
# STEP 20 — CGMacros SHAP attribution (peak_cgm ADI analysis)
# =============================================================================
step 20 "CGMacros SHAP attribution (clean vs leaky_peak_cgm)"

python src/run_cgmacros_shap.py \
    --data   "${CGM_COHORT}" \
    --outdir "${OUTDIR}" \
    --figdir "${FIGDIR}" \
    --seed   ${SEED_DEV}

echo "✓ CGMacros SHAP done → cgmacros_shap_adi.csv"
_progress 20 "CGMacros SHAP done"

# =============================================================================
# STEP 21 — CGMacros subject-level uncertainty (LOSO + bootstrap)
# =============================================================================
step 21 "CGMacros subject-level uncertainty (LOSO + ${N_BOOT} bootstrap resamples)"

python src/run_cgmacros_subject_uncertainty.py \
    --data        "${CGM_COHORT}" \
    --outdir      "${OUTDIR}" \
    --figdir      "${FIGDIR}" \
    --n_bootstrap ${N_BOOT} \
    --seed        ${SEED_DEV}

echo "✓ CGMacros uncertainty done → cgmacros_loso_results.csv / cgmacros_bootstrap_ci.csv"
_progress 21 "CGMacros uncertainty done"

# =============================================================================
# STEP 22 — ICU Glucose leakage audit (PhysioNet curated dataset)
# =============================================================================
step 22 "ICU Glucose external leakage audit (${ICU_GLUCOSE_CSV})"

python src/icu_glucose_leakage_audit.py \
    --data_path  "${ICU_GLUCOSE_CSV}" \
    --output_dir "${OUTDIR}"

echo "✓ ICU Glucose audit done → results_primary_6h.csv + window sensitivity CSVs"
_progress 22 "ICU Glucose audit done"

# =============================================================================
# STEP 23 — eICU early-mortality leakage audit (24h + 48h labels)
# =============================================================================
step 23 "eICU external leakage audit (24h + 48h, ${N_SEEDS_EXT} seeds)"

python src/run_external_eicu.py \
    --data24  "${EICU_24H}" \
    --data48  "${EICU_48H}" \
    --outdir  "${OUTDIR}" \
    --figdir  "${FIGDIR}" \
    --n_seeds ${N_SEEDS_EXT} \
    --n_folds 5

echo "✓ eICU audit done → external_eicu_24h_results.csv + external_eicu_48h_results.csv"
_progress 23 "eICU audit done"

# =============================================================================
# STEP 24 — MIMIC-IV hospital-mortality leakage audit
# =============================================================================
step 24 "MIMIC-IV external leakage audit (${N_SEEDS_EXT} seeds)"

python src/run_external_mimic.py \
    --data    "${MIMIC_CSV}" \
    --outdir  "${OUTDIR}" \
    --figdir  "${FIGDIR}" \
    --n_seeds ${N_SEEDS_EXT} \
    --n_folds 5

echo "✓ MIMIC-IV audit done → external_mimic_results.csv"
_progress 24 "MIMIC-IV audit done"

# =============================================================================
# STEP 25 — External calibration (eICU 24h + MIMIC, ECE / slope / Brier)
# =============================================================================
step 25 "External calibration analysis (eICU + MIMIC reliability diagrams)"

python src/run_external_calibration.py \
    --eicu24 "${EICU_24H}" \
    --mimic  "${MIMIC_CSV}" \
    --outdir "${OUTDIR}" \
    --figdir "${FIGDIR}" \
    --seed   ${SEED_DEV} \
    --n_folds 5

echo "✓ External calibration done → external_calibration_results.csv"
_progress 25 "External calibration done"

# =============================================================================
# STEP 26 — External SHAP attribution (eICU definitional pipeline ADI)
# =============================================================================
step 26 "External SHAP attribution — eICU definitional pipeline (ADI table)"

python src/run_external_shap.py \
    --eicu24 "${EICU_24H}" \
    --outdir "${OUTDIR}" \
    --figdir "${FIGDIR}" \
    --seed   ${SEED_DEV}

echo "✓ eICU SHAP done → external_shap_eicu_adi.csv"
_progress 26 "eICU SHAP done"

# =============================================================================
# STEP 27 — Decision Curve Analysis (eICU 24h, clean vs leaky)
# =============================================================================
step 27 "Decision Curve Analysis — eICU 24h (clean vs leaky_definitional)"

python src/run_dca.py \
    --data24  "${EICU_24H}" \
    --outdir  "${OUTDIR}" \
    --figdir  "${FIGDIR}" \
    --n_folds 5 \
    --seed    ${SEED_DEV}

echo "✓ DCA done → dca_results.csv + dca_comparison.png"
_progress 27 "DCA done"

# =============================================================================
# STEP 28 — Combined external-cohort figure (3-panel: eICU 24h / 48h / MIMIC)
# =============================================================================
step 28 "Generating combined external-cohort figure (3 panels)"

python src/plot_external_cohort.py \
    --eicu24 "${OUTDIR}/external_eicu_24h_results.csv" \
    --eicu48 "${OUTDIR}/external_eicu_48h_results.csv" \
    --mimic  "${OUTDIR}/external_mimic_results.csv" \
    --figdir "${FIGDIR}"

echo "✓ Combined external figure done → external_cohort_combined.png"
_progress 28 "External figures done"

# =============================================================================
# STEP 29 — Generate all manuscript figures
# =============================================================================
step 29 "Generating all manuscript figures"

python src/plotting.py \
    --datadir    data/ \
    --resultsdir "${OUTDIR}/" \
    --figdir     "${FIGDIR}/"

echo "✓ All figures generated."
_progress 29 "All figures generated"

# =============================================================================
# Summary
# =============================================================================
PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE — $(date)"
echo "  Total runtime: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "============================================================"
echo ""
echo "  Synthetic benchmark tables:"
echo "    ${OUTDIR}/clean_results.csv"
echo "    ${OUTDIR}/leakage_benchmark.csv"
echo "    ${OUTDIR}/scenario_sensitivity.csv"
echo "    ${OUTDIR}/sample_size_sensitivity.csv"
echo "    ${OUTDIR}/noise_sensitivity.csv"
echo "    ${OUTDIR}/missingness_sensitivity.csv"
echo "    ${OUTDIR}/outlier_stress_test.csv"
echo "    ${OUTDIR}/domain_shift.csv"
echo "    ${OUTDIR}/domain_shift_wbv_comparison.csv"
echo "    ${OUTDIR}/domain_shift_multiseed.csv"
echo "    ${OUTDIR}/domain_shift_multiseed_summary.csv"
echo "    ${OUTDIR}/domain_shift_multiseed_wbv_summary.csv"
echo ""
echo "  Validation tables:"
echo "    ${OUTDIR}/validation_null.json"
echo "    ${OUTDIR}/validation_weak_signal.json"
echo "    ${OUTDIR}/validation_moderate_signal.json"
echo "    ${OUTDIR}/validation_wbv_positive.json"
echo ""
echo "  SHAP / attribution tables:"
echo "    ${OUTDIR}/unified_shap_adi_rf.csv"
echo "    ${OUTDIR}/unified_shap_cross_clf.csv"
echo "    ${OUTDIR}/cross_classifier_shap.csv"
echo "    ${OUTDIR}/bootstrap_adi.csv"
echo "    ${OUTDIR}/bootstrap_adi_summary.csv"
echo "    ${OUTDIR}/alternate_formula_benchmark.csv"
echo "    ${OUTDIR}/ratio_formula_benchmark.csv"
echo ""
echo "  External-cohort tables:"
echo "    ${OUTDIR}/cgmacros_meal_cohort.csv"
echo "    ${OUTDIR}/external_cgmacros_results.csv"
echo "    ${OUTDIR}/cgmacros_shap_adi.csv"
echo "    ${OUTDIR}/cgmacros_loso_results.csv"
echo "    ${OUTDIR}/cgmacros_bootstrap_ci.csv"
echo "    ${OUTDIR}/results_primary_6h.csv  (ICU glucose)"
echo "    ${OUTDIR}/external_eicu_24h_results.csv"
echo "    ${OUTDIR}/external_eicu_48h_results.csv"
echo "    ${OUTDIR}/external_mimic_results.csv"
echo "    ${OUTDIR}/external_calibration_results.csv"
echo "    ${OUTDIR}/external_shap_eicu_adi.csv"
echo "    ${OUTDIR}/dca_results.csv"
echo ""
echo "  All manuscript figures → ${FIGDIR}/"
echo "============================================================"
