"""
run_external_calibration.py — Calibration analysis for external real-world leakage audits.

Paper section: §6 External Real-World Leakage Audits (supplementary calibration evidence)

Purpose
-------
Demonstrates that the leaky pipelines are MISCALIBRATED despite achieving near-perfect AUROC.
A classifier trained on definitional antecedents / proxy variables learns a degenerate mapping
(predicted probabilities ≈ 0 or 1), giving a collapsed reliability diagram and calibration
slope << 1.  This mirrors the synthetic domain-shift results (slope 0.109–0.330) and
strengthens the argument that leakage detection cannot rely on AUROC alone.

Cohorts
-------
  eICU 24h  — label = Expired AND offset <= 1440 min  (clean vs leaky_definitional)
  MIMIC-IV  — label = in-hospital mortality            (clean vs leaky_discharge_location)

Design
------
  • Single seed (42), 5-fold stratified CV
  • Out-of-fold predictions concatenated → global calibration curve
  • Models evaluated: RandomForest (primary) and LogisticRegression
  • Metrics: ECE (10 bins, uniform), Brier score, calibration slope, calibration intercept
  • Figures: reliability diagram (2-pipeline overlay) per cohort × model
  • Output CSV: one row per cohort × pipeline × model

Usage
-----
    python src/run_external_calibration.py \\
        --eicu24  "External Cohort/Dataset/eicu_label24h.csv" \\
        --mimic   "External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv" \\
        --outdir  results/tables \\
        --figdir  results/figures
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration import (
    calibration_summary,
    get_calibration_curve_data,
)

# Feature lists and preprocessors re-imported from the leakage audit scripts
from src.run_external_eicu import (
    FoldSealedPreprocessorICU,
    NUMERIC_CLEAN        as EICU_NUMERIC_CLEAN,
    CATEGORICAL_CLEAN    as EICU_CATEGORICAL_CLEAN,
    LEAKY_DEFN_NUMERIC   as EICU_LEAKY_NUM,
    LEAKY_DEFN_CATEGORICAL as EICU_LEAKY_CAT,
)
from src.run_external_mimic import (
    FoldSealedPreprocessorMIMIC,
    NUMERIC_CLEAN        as MIMIC_NUMERIC_CLEAN,
    CATEGORICAL_CLEAN    as MIMIC_CATEGORICAL_CLEAN,
    LEAKY_NUMERIC        as MIMIC_LEAKY_NUM,
    LEAKY_CATEGORICAL    as MIMIC_LEAKY_CAT,
)

# ── Visual style (matches main paper figures) ────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

COLOUR_CLEAN = "#2C7BB6"
COLOUR_LEAKY = "#D7191C"
N_BINS       = 10
SEED         = 42
N_FOLDS      = 5


# ─────────────────────────────────────────────────────────────────────────────
# Classifier factory  (two models only — RF and LR)
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=SEED,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core: collect out-of-fold predictions for a single pipeline
# ─────────────────────────────────────────────────────────────────────────────

def collect_oof_predictions(
    df: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    Preprocessor,           # FoldSealedPreprocessorICU or MIMIC
    model,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run stratified k-fold CV, return (y_true_oof, y_prob_oof).
    Preprocessing is fold-sealed: fitted on training fold only.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_true  = np.empty(len(y), dtype=int)
    oof_probs = np.empty(len(y), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)
        y_train  = y[train_idx]
        y_test   = y[test_idx]

        # Fold-sealed preprocessing
        prep = Preprocessor()
        X_train = prep.fit_transform(df_train, num_cols, cat_cols)
        X_test  = prep.transform(df_test)

        # Handle all-zero or all-one train fold (degenerate edge case)
        if len(np.unique(y_train)) < 2:
            oof_true[test_idx]  = y_test
            oof_probs[test_idx] = y_train.mean()
            continue

        clf = deepcopy(model)
        clf.fit(X_train, y_train)
        oof_true[test_idx]  = y_test
        oof_probs[test_idx] = clf.predict_proba(X_test)[:, 1]

    return oof_true, oof_probs


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    results: list[dict],   # each entry: {pipeline, model, y_true, y_prob}
    cohort_label: str,
    out_path: Path,
    n_bins: int = N_BINS,
) -> None:
    """
    Reliability diagram for all pipeline × model combinations in `results`.
    Each unique model gets its own subplot column; pipelines share a colour.
    """
    models  = list(dict.fromkeys(r["model"]    for r in results))
    n_cols  = len(models)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), squeeze=False)
    axes = axes[0]  # 1-D array of Axes

    for ax, model_name in zip(axes, models):
        # Perfect calibration reference
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.45, label="Perfect calibration")

        for entry in results:
            if entry["model"] != model_name:
                continue
            pipe   = entry["pipeline"]
            colour = COLOUR_LEAKY if "leaky" in pipe else COLOUR_CLEAN
            label  = "Leaky" if "leaky" in pipe else "Clean"
            ls     = "-" if "clean" in pipe else "-."

            try:
                frac_pos, mean_pred = get_calibration_curve_data(
                    entry["y_true"], entry["y_prob"], n_bins=n_bins
                )
                ax.plot(mean_pred, frac_pos, color=colour, lw=2.0, ls=ls,
                        marker="o", markersize=4, label=label)
            except Exception as exc:
                print(f"  [warn] calibration curve failed for {pipe}/{model_name}: {exc}")

        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{model_name}", fontsize=9.5)
        ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        f"Reliability Diagram — {cohort_label}\n"
        "(Leaky pipeline: predictions collapse to 0/1 → overconfident, miscalibrated)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_path).replace(".png", ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-cohort analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_cohort(
    df: pd.DataFrame,
    label_col: str,
    cohort_tag: str,
    cohort_label: str,
    pipelines: dict,        # {pipeline_name: (num_cols, cat_cols)}
    Preprocessor,           # class (not instance)
    figdir: Path,
) -> list[dict]:
    """
    Run calibration analysis for all pipelines × models on one cohort.

    Parameters
    ----------
    pipelines : dict mapping pipeline name → (num_cols, cat_cols)
    Preprocessor : FoldSealedPreprocessorICU or FoldSealedPreprocessorMIMIC

    Returns
    -------
    List of calibration summary dicts (one per pipeline × model)
    """
    y = df[label_col].values.astype(int)
    print(f"\n{'─'*60}")
    print(f"  Cohort: {cohort_label}  |  N={len(df):,}  |  positives={y.sum():,}  "
          f"({100*y.mean():.1f}%)")
    print(f"{'─'*60}")

    models  = build_models()
    records = []
    plot_inputs: list[dict] = []

    for pipe_name, (num_cols, cat_cols) in pipelines.items():
        for model_name, model in models.items():
            print(f"  {pipe_name} × {model_name} … ", end="", flush=True)
            try:
                y_true, y_prob = collect_oof_predictions(
                    df, y, num_cols, cat_cols, Preprocessor, model
                )
                summary = calibration_summary(y_true, y_prob,
                                              label=f"{pipe_name}/{model_name}")
                record = {
                    "cohort":        cohort_tag,
                    "pipeline":      pipe_name,
                    "model":         model_name,
                    "n_total":       len(y),
                    "n_positive":    int(y.sum()),
                    "prevalence":    round(float(y.mean()), 5),
                    "brier":         summary["brier"],
                    "ece":           summary["ece"],
                    "cal_slope":     summary["cal_slope"],
                    "cal_intercept": summary["cal_intercept"],
                }
                records.append(record)
                print(f"ECE={summary['ece']:.4f}  slope={summary['cal_slope']:.3f}  "
                      f"Brier={summary['brier']:.4f}")

                plot_inputs.append({
                    "pipeline": pipe_name,
                    "model":    model_name,
                    "y_true":   y_true,
                    "y_prob":   y_prob,
                })
            except Exception as exc:
                print(f"ERROR — {exc}")
                import traceback; traceback.print_exc()

    # Plot reliability diagram
    out_fig = figdir / f"calibration_reliability_{cohort_tag}.png"
    plot_reliability_diagram(plot_inputs, cohort_label, out_fig)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Calibration comparison bar-chart (ECE + slope) — combined figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_summary_figure(df_results: pd.DataFrame, figdir: Path) -> None:
    """
    Two-row figure:
      Row 1: ECE by cohort × pipeline × model
      Row 2: Calibration slope by cohort × pipeline × model

    Perfect calibration:  ECE = 0,  slope = 1.
    Leaky pipeline expected:  ECE >> 0,  slope << 1.
    """
    cohorts = df_results["cohort"].unique().tolist()
    models  = df_results["model"].unique().tolist()
    n_cols  = len(cohorts)

    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8), squeeze=False)

    for col, cohort in enumerate(cohorts):
        sub = df_results[df_results["cohort"] == cohort]
        pipes   = sub["pipeline"].unique().tolist()
        n_pipes = len(pipes)
        n_models = len(models)
        bar_w  = 0.7 / n_models
        x_pos  = np.arange(n_pipes)

        colours = {"clean": COLOUR_CLEAN}   # leaky pipelines get red
        for p in pipes:
            if p not in colours:
                colours[p] = COLOUR_LEAKY

        hatches = ["", "//", "xx", ".."]

        for row_idx, metric in enumerate(["ece", "cal_slope"]):
            ax = axes[row_idx, col]
            for mi, model in enumerate(models):
                vals = []
                for pipe in pipes:
                    row = sub[(sub["pipeline"] == pipe) & (sub["model"] == model)]
                    vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
                offsets = (mi - (n_models - 1) / 2) * bar_w
                ax.bar(
                    x_pos + offsets, vals, bar_w * 0.88,
                    label=model,
                    color=[colours[p] for p in pipes],
                    hatch=hatches[mi % len(hatches)],
                    alpha=0.85,
                )
            # Reference lines
            if metric == "cal_slope":
                ax.axhline(1.0, color="gray", lw=1.2, ls="--", alpha=0.6,
                           label="Perfect (slope=1)")
                ax.set_ylabel("Calibration slope")
                ax.set_ylim(0, 1.5)
            else:
                ax.axhline(0.0, color="gray", lw=1.2, ls="--", alpha=0.6)
                ax.set_ylabel("ECE (↓ better)")
                ax.set_ylim(0, None)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(pipes, fontsize=8.5, rotation=10, ha="right")
            ax.set_title(f"{cohort}  —  {'ECE' if metric == 'ece' else 'Cal. slope'}",
                         fontsize=9.5)
            ax.legend(fontsize=7.5, loc="best")

    fig.suptitle(
        "Calibration Metrics: Clean vs Leaky Pipelines\n"
        "(Leaky: ECE ↑, slope << 1  →  overconfident, miscalibrated predictions)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        out = figdir / f"calibration_summary{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibration analysis for external real-world leakage audits."
    )
    p.add_argument(
        "--eicu24",
        default="External Cohort/Dataset/eicu_label24h.csv",
        help="eICU 24h label dataset CSV",
    )
    p.add_argument(
        "--mimic",
        default="External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv",
        help="MIMIC-IV all-admissions mortality CSV",
    )
    p.add_argument(
        "--outdir", default="results/tables",
        help="Directory for output CSV files",
    )
    p.add_argument(
        "--figdir", default="results/figures",
        help="Directory for output figures",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--n_folds", type=int, default=N_FOLDS,
        help="Number of CV folds (default: 5)",
    )
    return p.parse_args()


def main():
    args  = parse_args()
    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    # ── eICU 24h ─────────────────────────────────────────────────────────────
    eicu_path = Path(args.eicu24)
    if eicu_path.exists():
        print(f"\nLoading eICU 24h data from {eicu_path} …")
        df_eicu = pd.read_csv(
            eicu_path,
            keep_default_na=False,
            na_values=["NA", "NaN", "nan", ""],
        )
        print(f"  Loaded {len(df_eicu):,} rows, {df_eicu.shape[1]} columns.")

        # eICU pipelines: clean and leaky_definitional
        eicu_pipelines = {
            "clean": (
                EICU_NUMERIC_CLEAN,
                EICU_CATEGORICAL_CLEAN,
            ),
            "leaky_definitional": (
                EICU_NUMERIC_CLEAN + EICU_LEAKY_NUM,
                EICU_CATEGORICAL_CLEAN + EICU_LEAKY_CAT,
            ),
        }

        recs = analyse_cohort(
            df            = df_eicu,
            label_col     = "label24h",
            cohort_tag    = "eicu_24h",
            cohort_label  = "eICU 24h label (Expired ∩ offset ≤ 1440 min)",
            pipelines     = eicu_pipelines,
            Preprocessor  = FoldSealedPreprocessorICU,
            figdir        = figdir,
        )
        all_records.extend(recs)
    else:
        print(f"[skip] eICU 24h file not found: {eicu_path}")

    # ── MIMIC-IV ─────────────────────────────────────────────────────────────
    mimic_path = Path(args.mimic)
    if mimic_path.exists():
        print(f"\nLoading MIMIC-IV data from {mimic_path} …")
        df_mimic = pd.read_csv(
            mimic_path,
            keep_default_na=False,
            na_values=["NA", "NaN", "nan", ""],
        )
        print(f"  Loaded {len(df_mimic):,} rows, {df_mimic.shape[1]} columns.")

        mimic_pipelines = {
            "clean": (
                MIMIC_NUMERIC_CLEAN,
                MIMIC_CATEGORICAL_CLEAN,
            ),
            "leaky_discharge_location": (
                MIMIC_NUMERIC_CLEAN + MIMIC_LEAKY_NUM,
                MIMIC_CATEGORICAL_CLEAN + MIMIC_LEAKY_CAT,
            ),
        }

        recs = analyse_cohort(
            df            = df_mimic,
            label_col     = "label_mortality",
            cohort_tag    = "mimic",
            cohort_label  = "MIMIC-IV in-hospital mortality",
            pipelines     = mimic_pipelines,
            Preprocessor  = FoldSealedPreprocessorMIMIC,
            figdir        = figdir,
        )
        all_records.extend(recs)
    else:
        print(f"[skip] MIMIC file not found: {mimic_path}")

    if not all_records:
        print("\nNo data processed. Exiting.")
        return

    # ── Save combined CSV ────────────────────────────────────────────────────
    df_results = pd.DataFrame(all_records)
    out_csv = outdir / "external_calibration_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Saved calibration CSV → {out_csv}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  CALIBRATION SUMMARY")
    print("═" * 80)
    cols_display = ["cohort", "pipeline", "model", "ece", "cal_slope",
                    "cal_intercept", "brier"]
    print(df_results[cols_display].to_string(index=False))

    # ── Combined summary figure ──────────────────────────────────────────────
    print("\nBuilding combined calibration summary figure …")
    plot_calibration_summary_figure(df_results, figdir)

    print("\n✓  Calibration analysis complete.")
    print(f"     Tables  → {outdir}")
    print(f"     Figures → {figdir}")


if __name__ == "__main__":
    main()
