"""
run_domain_shift.py — Synthetic domain shift stress test.

Tests how performance and calibration degrade when training and test data
come from slightly different distributions (simulating real-world deployment).

Shifts tested:
  1. TG0h distribution shift (different lognormal parameters)
  2. BMI mean shift (+2 SD)
  3. Noise shift (increased residual variance in TCR)
  4. Class prevalence shift (different Q threshold)

Usage:
  python src/run_domain_shift.py \\
      --train data/paired_tcr_null_v1_seed2026.csv \\
      --shift_config config/generator_null.yaml \\
      --shift_seed 2027 \\
      --config config/model_config.yaml \\
      --out results/tables/domain_shift.csv \\
      --figdir results/figures
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
import yaml
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import CLEAN_FEATURES
from src.calibration import (
    calibration_summary,
    get_calibration_curve_data,
    platt_scaling,
    isotonic_regression_calibration,
)
from src.utils import set_seed

import copy


# ---------------------------------------------------------------------------
# Domain shift generators
# ---------------------------------------------------------------------------

def make_shift_config(base_cfg: dict, shift_type: str) -> dict:
    """Return a modified config dict for a particular shift type."""
    cfg = copy.deepcopy(base_cfg)

    if shift_type == "tg0h_shift":
        # Shift TG0h to higher baseline values
        cfg["tg0h_mu_log"] = base_cfg.get("tg0h_mu_log", 5.5) + 0.3
        cfg["tg0h_shift"] = base_cfg.get("tg0h_shift", 350.0) + 100.0

    elif shift_type == "bmi_shift":
        # +2 SD shift in BMI mean
        bmi_sd = base_cfg.get("bmi_sd", 3.0)
        cfg["bmi_mu"] = base_cfg.get("bmi_mu", 24.0) + 2 * bmi_sd

    elif shift_type == "noise_shift":
        # Increase TCR residual variance
        cfg["tcr_sd"] = base_cfg.get("tcr_sd", 18.6) * 1.5

    elif shift_type == "prevalence_shift":
        # Different class balance — harder threshold
        cfg["_prevalence_q"] = 35.0   # Q35 instead of Q25

    return cfg


SHIFT_TYPES = ["tg0h_shift", "bmi_shift", "noise_shift", "prevalence_shift"]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_on_shifted(
    model,
    scaler: StandardScaler,
    df_test: pd.DataFrame,
    threshold: float,
    label_q: float = 25.0,
) -> dict:
    """Evaluate a model (trained on source domain) on shifted test data."""
    X_test = df_test[CLEAN_FEATURES].values
    X_test_s = scaler.transform(X_test)
    tcr_test = df_test["tcr"].values
    y_test = (tcr_test <= threshold).astype(int)

    if len(np.unique(y_test)) < 2:
        return {"error": "single class in shifted test set"}

    y_prob = model.predict_proba(X_test_s)[:, 1]

    auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    cal = calibration_summary(y_test, y_prob, label="shifted")

    return {
        "AUROC": round(auc, 4),
        "Brier": round(brier, 4),
        "ECE": cal["ece"],
        "cal_slope": cal["cal_slope"],
        "cal_intercept": cal["cal_intercept"],
        "prevalence": round(float(y_test.mean()), 4),
    }


# ---------------------------------------------------------------------------
# Recalibration comparison
# ---------------------------------------------------------------------------

def recalibration_comparison(
    model,
    scaler: StandardScaler,
    df_shift: pd.DataFrame,
    threshold: float,
) -> dict:
    """Compare raw vs Platt vs isotonic recalibration on shifted data."""
    X = df_shift[CLEAN_FEATURES].values
    X_s = scaler.transform(X)
    tcr = df_shift["tcr"].values
    y = (tcr <= threshold).astype(int)

    if len(np.unique(y)) < 2:
        return {}

    # Split shifted data into recalibration set and test set
    n_cal = min(200, len(df_shift) // 2)
    idx_cal = np.arange(n_cal)
    idx_test = np.arange(n_cal, len(df_shift))

    y_prob_all = model.predict_proba(X_s)[:, 1]
    y_prob_cal = y_prob_all[idx_cal]
    y_prob_test_raw = y_prob_all[idx_test]
    y_test = y[idx_test]

    if len(np.unique(y_test)) < 2:
        return {}

    # Platt
    y_prob_platt = platt_scaling(y[idx_cal], y_prob_cal, y_prob_test_raw)
    # Isotonic
    y_prob_iso = isotonic_regression_calibration(y[idx_cal], y_prob_cal, y_prob_test_raw)

    return {
        "raw":      calibration_summary(y_test, y_prob_test_raw, "raw"),
        "platt":    calibration_summary(y_test, y_prob_platt, "platt"),
        "isotonic": calibration_summary(y_test, y_prob_iso, "isotonic"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration_shift(results: dict, out_path: Path):
    """Plot calibration degradation across shift types."""
    shift_types = list(results.keys())
    eces = [results[s].get("shifted", {}).get("ECE", np.nan) for s in shift_types]
    slopes = [results[s].get("shifted", {}).get("cal_slope", np.nan) for s in shift_types]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(shift_types, eces, color="coral")
    axes[0].axhline(y=0.05, color="red", linestyle="--", label="ECE=0.05 threshold")
    axes[0].set_ylabel("ECE")
    axes[0].set_title("Expected Calibration Error by Shift Type")
    axes[0].legend()

    axes[1].bar(shift_types, slopes, color="steelblue")
    axes[1].axhline(y=1.0, color="black", linestyle="--", label="Perfect slope=1.0")
    axes[1].set_ylabel("Calibration Slope")
    axes[1].set_title("Calibration Slope by Shift Type")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration shift plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Domain shift stress test.")
    p.add_argument("--train", required=True, help="Training dataset CSV")
    p.add_argument("--shift_config", required=True, help="Generator YAML for shifted data")
    p.add_argument("--shift_seed", type=int, default=2027)
    p.add_argument("--config", required=True, help="model_config.yaml")
    p.add_argument("--out", default="results/tables/domain_shift.csv")
    p.add_argument("--figdir", default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()

    df_train = pd.read_csv(args.train, keep_default_na=False, na_values=["NA", "NaN", "nan", ""])
    with open(args.shift_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)

    # Train on source domain
    X_all = df_train[CLEAN_FEATURES].values
    tcr_all = df_train["tcr"].values
    label_q = model_cfg.get("label_threshold_percentile", 25.0)
    threshold = float(np.percentile(tcr_all, label_q))
    y_all = (tcr_all <= threshold).astype(int)

    # Split 80/20 so source_AUROC is out-of-sample (not in-sample RF overfit)
    X_tr, X_src_test, y_tr, y_src_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr)
    X_src_test_s = scaler.transform(X_src_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_tr)

    # Evaluate on held-out source test set (out-of-sample)
    source_auc = float(roc_auc_score(y_src_test, model.predict_proba(X_src_test_s)[:, 1]))
    print(f"Source domain AUROC (held-out 20%): {source_auc:.4f}")

    # Test on each shift type
    all_results = {}
    rows = []

    shift_pbar = tqdm(
        SHIFT_TYPES,
        desc="Domain shift types",
        ncols=90,
        colour="yellow",
    )
    for shift_type in shift_pbar:
        shift_pbar.set_description(f"Shift: {shift_type}")
        shift_cfg = make_shift_config(base_cfg, shift_type)
        n_shift = len(df_train)

        shift_pbar.set_postfix(status="generating shifted data")
        df_shift = generate(shift_cfg, seed=args.shift_seed, n=n_shift)

        shift_pbar.set_postfix(status="evaluating")
        shifted_metrics = evaluate_on_shifted(model, scaler, df_shift, threshold, label_q)

        shift_pbar.set_postfix(status="recalibrating")
        recal = recalibration_comparison(model, scaler, df_shift, threshold)

        all_results[shift_type] = {
            "shifted": shifted_metrics,
            "recalibration": recal,
        }

        shift_pbar.set_postfix(
            AUROC=f"{shifted_metrics.get('AUROC', 'N/A')}",
            ECE=f"{shifted_metrics.get('ECE', 'N/A')}",
            refresh=True,
        )
        tqdm.write(f"  ✓ {shift_type}: AUROC={shifted_metrics.get('AUROC','N/A')}  ECE={shifted_metrics.get('ECE','N/A')}")

        rows.append({
            "shift_type": shift_type,
            "source_AUROC": round(source_auc, 4),
            **{f"shifted_{k}": v for k, v in shifted_metrics.items() if not isinstance(v, dict)},
        })

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved domain shift results -> {out_path}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed JSON -> {json_path}")

    # Figure
    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration_shift(all_results, fig_dir / "domain_shift_calibration.png")


if __name__ == "__main__":
    main()
