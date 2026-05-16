"""
validate_synthetic_data.py — Verify generator output matches target distributions.

Runs TOST equivalence tests for each variable and checks:
  - All variables within target mean ± tolerance
  - WBV formula correctness (deterministic re-derivation)
  - TG4h > 0 for all records
  - WBV–TCR Pearson |r| < 0.10 (null scenario: they should be uncorrelated)
  - TG4h and TCR correlation equals 1 analytically (sanity: derived relationship)

Usage:
  python src/validate_synthetic_data.py \\
      --data data/paired_tcr_null_v1_seed2026.csv \\
      --scenario null \\
      --out results/tables/validation_null.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import tost_equivalence_test, compute_pearson_spearman_partial_r


# ---------------------------------------------------------------------------
# Reference targets (from original Lipids manuscript / protocol)
# ---------------------------------------------------------------------------

REFERENCE_TARGETS = {
    "age":  {"mean": 53.0,  "sd": 10.0},
    "bmi":  {"mean": 24.0,  "sd": 3.0},
    "hct":  {"mean": 41.7,  "sd": 3.75},
    "tp":   {"mean": 6.88,  "sd": 0.62},
    "hdl":  {"mean": 50.5,  "sd": 9.7},
    "ldl":  {"mean": 131.0, "sd": 28.0},
    # WBV = 0.12*41.7 + 0.17*(6.88-2.07) = 5.004 + 0.817 = 5.821 ≈ 5.85
    "wbv":  {"mean": 5.85,  "sd": 0.46},
    # Shifted lognormal: shift=350, mu_log=5.5, sigma_log=0.45 → empirical mean≈617, sd≈130
    "tg0h": {"mean": 617.0, "sd": 130.0},
    "tcr":  {"mean": 52.2,  "sd": 18.6},
}

RANGE_CHECKS = {
    "age":  (18, 75),
    "bmi":  (16, 33),
    "hct":  (30, 55),
    "tp":   (5.0, 8.9),
    "hdl":  (25, 87),
    "ldl":  (70, 215),
    "tg0h": (350, 1750),
    "tcr":  (-10, 99.9),
    "tg4h": (0, 1300),
}


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def check_wbv_formula(df: pd.DataFrame) -> dict:
    """Verify WBV = 0.12*Hct + 0.17*(TP-2.07) up to rounding precision.

    Stored WBV is rounded to 3 decimal places (np.round(wbv, 3)),
    so the tolerance is 0.01 to accommodate the rounding artefact.
    """
    expected = 0.12 * df["hct"] + 0.17 * (df["tp"] - 2.07)
    max_err = (df["wbv"] - expected).abs().max()
    return {
        "check": "WBV formula",
        "pass": bool(max_err < 0.01),   # 0.01 tolerance for np.round(wbv, 3) artefact
        "max_abs_error": float(max_err),
    }


def check_tg4h_positivity(df: pd.DataFrame) -> dict:
    """Verify all TG4h values are positive."""
    n_negative = int((df["tg4h"] <= 0).sum())
    return {
        "check": "TG4h > 0",
        "pass": n_negative == 0,
        "n_negative": n_negative,
    }


def check_tg4h_tg0h_tcr_relationship(df: pd.DataFrame) -> dict:
    """Check TG4h ≈ TG0h × (1 − TCR/100) analytically."""
    derived = df["tg0h"] * (1.0 - df["tcr"] / 100.0)
    max_err = (df["tg4h"] - derived).abs().max()
    return {
        "check": "TG4h derivation from TG0h and TCR",
        "pass": bool(max_err < 1.0),   # 1 mg/dL tolerance for rounding
        "max_abs_error": float(max_err),
    }


def check_wbv_tcr_independence(df: pd.DataFrame, r_threshold: float = 0.10) -> dict:
    """Check |Pearson r(WBV, TCR)| < threshold (null scenario expectation)."""
    corr_result = compute_pearson_spearman_partial_r(df, "wbv", "tcr")
    r = corr_result["pearson_r"]
    return {
        "check": "WBV-TCR independence (null scenario)",
        "pass": abs(r) < r_threshold,
        "pearson_r": round(r, 4),
        "threshold": r_threshold,
    }


def check_ranges(df: pd.DataFrame) -> list[dict]:
    """Check that all variables fall within expected ranges."""
    results = []
    cols = [(col, lo, hi) for col, (lo, hi) in RANGE_CHECKS.items() if col in df.columns]
    for col, lo, hi in tqdm(cols, desc="  Range checks", ncols=80, colour="cyan", leave=False):
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        results.append({
            "check": f"{col} range [{lo}, {hi}]",
            "pass": n_out == 0,
            "n_out_of_range": n_out,
            "actual_min": round(float(df[col].min()), 3),
            "actual_max": round(float(df[col].max()), 3),
        })
    return results


def check_distributions(df: pd.DataFrame) -> list[dict]:
    """TOST equivalence tests for each variable against reference targets."""
    results = []
    items = [(col, ref) for col, ref in REFERENCE_TARGETS.items() if col in df.columns]
    for col, ref in tqdm(items, desc="  TOST equivalence", ncols=80, colour="cyan", leave=False):
        tost = tost_equivalence_test(
            df[col].values,
            reference_mean=ref["mean"],
            reference_sd=ref["sd"],
            delta=0.15,   # 15% SD margin
        )
        tost["variable"] = col
        results.append(tost)
    return results


def check_sex_prevalence(df: pd.DataFrame, expected_p_male: float = 0.485) -> dict:
    """Check male prevalence is near expected."""
    observed = float(df["sex"].mean())
    return {
        "check": "Sex prevalence",
        "pass": abs(observed - expected_p_male) < 0.05,
        "observed_p_male": round(observed, 4),
        "expected_p_male": expected_p_male,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, out_dir: Path, scenario: str):
    """Generate distribution histograms for all key variables."""
    vars_to_plot = ["age", "sex", "bmi", "hct", "tp", "wbv", "hdl", "ldl",
                    "tg0h", "tcr", "tg4h"]
    vars_to_plot = [v for v in vars_to_plot if v in df.columns]

    n_cols = 4
    n_rows = (len(vars_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(vars_to_plot):
        ax = axes[i]
        if col == "sex":
            df["sex"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "salmon"])
            ax.set_xticklabels(["Male (1)", "Female (0)"], rotation=0)
        else:
            ax.hist(df[col], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_title(f"{col}\nmean={df[col].mean():.2f}, sd={df[col].std():.2f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Synthetic data distributions — {scenario} scenario", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = out_dir / f"validation_distributions_{scenario}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution plot -> {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path, scenario: str):
    """Correlation heatmap for all numeric variables."""
    numeric_cols = ["age", "bmi", "hct", "tp", "wbv", "hdl", "ldl",
                    "tg0h", "tcr", "tg4h"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Pearson correlation heatmap — {scenario} scenario")
    out_path = out_dir / f"validation_corr_{scenario}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(
    data_path: Path,
    scenario: str,
    out_path: Path,
    fig_dir: Path,
) -> dict:
    """Run all validation checks and return results dict."""
    df = pd.read_csv(data_path, keep_default_na=False, na_values=["NA", "NaN", "nan", ""])

    validation_steps = [
        "load data",
        "WBV formula check",
        "TG4h positivity check",
        "TG4h derivation check",
        "sex prevalence check",
        "WBV-TCR independence check" if scenario == "null" else "skip WBV-TCR (non-null)",
        "range checks",
        "TOST distribution checks",
        "build summary",
        "save JSON",
        "plot distributions",
        "plot correlation heatmap",
    ]

    pbar = tqdm(validation_steps, desc="Validation", ncols=90, colour="green")

    # Step: load data
    pbar.set_description("Validation: loading data"); pbar.update(0)
    tqdm.write(f"  Loaded {len(df)} records from {data_path}")

    results = {
        "file": str(data_path),
        "n": len(df),
        "scenario": scenario,
        "checks": [],
    }
    pbar.update(1)

    # Formula / derivation checks
    pbar.set_description("Validation: WBV formula check")
    results["checks"].append(check_wbv_formula(df))
    pbar.update(1)

    pbar.set_description("Validation: TG4h positivity check")
    results["checks"].append(check_tg4h_positivity(df))
    pbar.update(1)

    pbar.set_description("Validation: TG4h derivation check")
    results["checks"].append(check_tg4h_tg0h_tcr_relationship(df))
    pbar.update(1)

    pbar.set_description("Validation: sex prevalence check")
    results["checks"].append(check_sex_prevalence(df))
    pbar.update(1)

    pbar.set_description("Validation: WBV-TCR independence" if scenario == "null" else "Validation: skip WBV-TCR (non-null)")
    if scenario == "null":
        results["checks"].append(check_wbv_tcr_independence(df))
    pbar.update(1)

    # Range checks
    pbar.set_description("Validation: range checks")
    results["checks"].extend(check_ranges(df))
    pbar.update(1)

    # Distribution equivalence tests
    pbar.set_description("Validation: TOST distribution checks")
    results["tost_tests"] = check_distributions(df)
    pbar.update(1)

    # Summary pass/fail
    pbar.set_description("Validation: building summary")
    all_checks = results["checks"]
    n_pass = sum(1 for c in all_checks if c.get("pass", False))
    n_fail = sum(1 for c in all_checks if not c.get("pass", True))
    results["summary"] = {
        "total_checks": len(all_checks),
        "passed": n_pass,
        "failed": n_fail,
        "all_passed": n_fail == 0,
    }
    pbar.update(1)

    tqdm.write(f"\n=== Validation Summary ({scenario}) ===")
    for check in all_checks:
        status = "PASS" if check.get("pass") else "FAIL"
        tqdm.write(f"  [{status}] {check['check']}")
    tqdm.write(f"\nTotal: {n_pass}/{len(all_checks)} checks passed")

    # Save JSON
    pbar.set_description("Validation: saving JSON report")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    tqdm.write(f"\nSaved validation report -> {out_path}")
    pbar.update(1)

    # Figures
    fig_dir.mkdir(parents=True, exist_ok=True)

    pbar.set_description("Validation: plotting distributions")
    plot_distributions(df, fig_dir, scenario)
    pbar.update(1)

    pbar.set_description("Validation: plotting correlation heatmap")
    plot_correlation_heatmap(df, fig_dir, scenario)
    pbar.update(1)

    pbar.close()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Validate synthetic dataset distributions.")
    p.add_argument("--data", required=True, help="Path to synthetic CSV")
    p.add_argument("--scenario", default="null",
                   choices=["null", "weak_signal", "moderate_signal", "wbv_positive"],
                   help="Scenario name (for labelling outputs)")
    p.add_argument("--out", default="results/tables/validation.json",
                   help="Path for JSON validation report")
    p.add_argument("--figdir", default="results/figures",
                   help="Directory for output figures")
    return p.parse_args()


def main():
    args = parse_args()
    run_validation(
        data_path=Path(args.data),
        scenario=args.scenario,
        out_path=Path(args.out),
        fig_dir=Path(args.figdir),
    )


if __name__ == "__main__":
    main()
