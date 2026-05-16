"""
run_scenario_sensitivity.py — Robustness sweep across 4 scenarios × seeds 1–100.

Also runs sample-size sensitivity: n = 300, 750, 1500, 3000.

Usage:
  python src/run_scenario_sensitivity.py \\
      --configdir config/ \\
      --out results/tables/scenario_sensitivity.csv \\
      --seeds 100 \\
      --n_jobs -1
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import nested_cv, build_models
from src.metrics import aggregate_seed_results

warnings.filterwarnings("ignore")


SCENARIO_CONFIGS = {
    "null":            "config/generator_null.yaml",
    "weak_signal":     "config/generator_weak_signal.yaml",
    "moderate_signal": "config/generator_moderate_signal.yaml",
    "wbv_positive":    "config/generator_wbv_positive.yaml",
}

SAMPLE_SIZES = [300, 750, 1500, 3000]


# ---------------------------------------------------------------------------
# Single seed runner
# ---------------------------------------------------------------------------

def run_one_seed(
    scenario: str,
    cfg_gen: dict,
    cfg_model: dict,
    seed: int,
    n: int,
    model_name: str = "LogisticRegression",
) -> dict:
    """Generate data for one seed and run the clean pipeline.

    Returns dict with scenario, seed, n, AUROC, PR_AUC, Brier.
    """
    try:
        df = generate(cfg_gen, seed=seed, n=n)
        models = build_models(cfg_model.get("models", {}))
        model = models[model_name]

        result = nested_cv(df, model_name, model, cfg_model, outer_folds=5, seed=seed)

        return {
            "scenario": scenario,
            "seed": seed,
            "n": n,
            "model": model_name,
            "AUROC": result.get("AUROC_mean"),
            "PR_AUC": result.get("PR_AUC_mean"),
            "Brier": result.get("Brier_mean"),
        }
    except Exception as e:
        return {
            "scenario": scenario,
            "seed": seed,
            "n": n,
            "model": model_name,
            "AUROC": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# WBV False Attribution Rate (across seeds, null scenario only)
# ---------------------------------------------------------------------------

def compute_far_across_seeds(
    records: list[dict],
    top_k: int = 3,
    target_feature: str = "wbv",
) -> float:
    """Proportion of seeds where WBV ranked in top-k SHAP (null scenario).

    NOTE: SHAP per-seed is expensive — this is a placeholder that uses
    feature importance from RF as a proxy when full SHAP is not computed.
    """
    shap_ranks = [r.get("wbv_shap_rank") for r in records if "wbv_shap_rank" in r]
    if not shap_ranks:
        return float("nan")
    return float(sum(1 for r in shap_ranks if r is not None and r <= top_k) / len(shap_ranks))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_seed_sweep(
    scenarios: dict,
    cfg_model: dict,
    n_seeds: int,
    n: int,
    model_name: str,
    n_jobs: int,
) -> list[dict]:
    """Run all seeds for all scenarios at fixed sample size n."""
    tasks = [
        (scenario, cfg, seed, n)
        for scenario, cfg in scenarios.items()
        for seed in range(1, n_seeds + 1)
    ]

    tqdm.write(f"Seed sweep: {len(tasks)} tasks ({len(scenarios)} scenarios × {n_seeds} seeds)")
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_one_seed)(scenario, cfg, cfg_model, seed, n, model_name)
        for scenario, cfg, seed, n in tqdm(
            tasks,
            desc="Seed sweep",
            ncols=90,
            colour="yellow",
        )
    )
    return results


def run_sample_size_sensitivity(
    scenarios: dict,
    cfg_model: dict,
    sample_sizes: list[int],
    n_seeds: int,
    model_name: str,
    n_jobs: int,
) -> list[dict]:
    """Run seeds 1–n_seeds for each sample size and scenario."""
    tasks = [
        (scenario, cfg, seed, n_sz)
        for scenario, cfg in scenarios.items()
        for n_sz in sample_sizes
        for seed in range(1, n_seeds + 1)
    ]

    tqdm.write(f"Sample-size sweep: {len(tasks)} tasks ({len(scenarios)} scenarios × {len(sample_sizes)} sizes × {n_seeds} seeds)")
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_one_seed)(scenario, cfg, cfg_model, seed, n_sz, model_name)
        for scenario, cfg, seed, n_sz in tqdm(
            tasks,
            desc="Sample-size sweep",
            ncols=90,
            colour="cyan",
        )
    )
    return results


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary_table(records: list[dict]) -> pd.DataFrame:
    """Aggregate results by scenario × n."""
    df = pd.DataFrame(records)
    df = df[df["AUROC"].notna()]

    rows = []
    for (scenario, n), grp in df.groupby(["scenario", "n"]):
        auc_vals = grp["AUROC"].values
        rows.append({
            "scenario": scenario,
            "n": n,
            "n_seeds": len(auc_vals),
            "AUROC_mean": round(float(auc_vals.mean()), 4),
            "AUROC_sd": round(float(auc_vals.std(ddof=1)), 4),
            "AUROC_p2_5": round(float(np.percentile(auc_vals, 2.5)), 4),
            "AUROC_p97_5": round(float(np.percentile(auc_vals, 97.5)), 4),
            "pct_null_range": round(float(np.mean((auc_vals >= 0.45) & (auc_vals <= 0.55)) * 100), 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run scenario × seed sensitivity sweep.")
    p.add_argument("--configdir", default="config/", help="Directory containing config YAMLs")
    p.add_argument("--model_config", default="config/model_config.yaml")
    p.add_argument("--out", default="results/tables/scenario_sensitivity.csv")
    p.add_argument("--seeds", type=int, default=100, help="Number of seeds (1..N)")
    p.add_argument("--n", type=int, default=1500, help="Fixed sample size for main sweep")
    p.add_argument("--sample_size_sweep", action="store_true",
                   help="Also run sample size sensitivity (300, 750, 1500, 3000)")
    p.add_argument("--model", default="LogisticRegression")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 5 seeds only (for testing)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model config
    with open(args.model_config) as f:
        cfg_model = yaml.safe_load(f)

    # Load all scenario generator configs
    config_dir = Path(args.configdir)
    scenarios = {}
    for scenario_name, cfg_path in SCENARIO_CONFIGS.items():
        cfg_file = Path(cfg_path)
        if not cfg_file.exists():
            cfg_file = config_dir / cfg_file.name
        if cfg_file.exists():
            with open(cfg_file) as f:
                scenarios[scenario_name] = yaml.safe_load(f)
        else:
            print(f"WARNING: config not found: {cfg_file} — skipping {scenario_name}")

    n_seeds = 5 if args.quick else args.seeds
    print(f"Running {n_seeds} seeds × {len(scenarios)} scenarios at n={args.n}")

    # Main seed sweep
    records = run_seed_sweep(
        scenarios, cfg_model, n_seeds=n_seeds,
        n=args.n, model_name=args.model, n_jobs=args.n_jobs,
    )

    # Sample size sweep (optional)
    if args.sample_size_sweep:
        print("\nRunning sample size sensitivity sweep ...")
        ss_records = run_sample_size_sensitivity(
            scenarios, cfg_model, sample_sizes=SAMPLE_SIZES,
            n_seeds=min(n_seeds, 20),   # 20 seeds per size is sufficient
            model_name=args.model, n_jobs=args.n_jobs,
        )
        records.extend(ss_records)

    # Build and save summary
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary_table(records)
    summary.to_csv(out_path, index=False)
    print(f"\nSaved sensitivity summary -> {out_path}")

    # Save raw records
    raw_path = out_path.with_name(out_path.stem + "_raw.csv")
    pd.DataFrame(records).to_csv(raw_path, index=False)
    print(f"Saved raw records -> {raw_path}")

    print("\n=== Scenario Sensitivity Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
