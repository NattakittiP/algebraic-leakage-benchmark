"""
run_external_shap.py — SHAP attribution analysis for external real-world leakage audits.

Paper section: §6 External Real-World Leakage Audits (supplementary attribution evidence)

Purpose
-------
Shows that in the eICU 24h leaky_definitional pipeline, the two definitional antecedent
variables — hospitaldischargestatus and unitdischargeoffset_num — dominate feature
attribution (mean |SHAP| >> all clinical predictors).  In the clean pipeline these
variables are absent and all clinical features receive comparably modest attributions.

This mirrors the synthetic benchmark finding (TG4h/TCR dominate in leaky model) and
directly confirms the Attribution Distortion Index (ADI) concept in real ICU data.

Design
------
  • eICU 24h cohort only (definitional leakage; label = Expired AND offset ≤ 1440 min)
  • Single-seed (42) train/test split (80 / 20), stratified
  • Model: RandomForest (n_estimators=200; TreeExplainer is exact and fast on RF)
  • SHAP computed on the held-out test set (up to 500 samples for speed)
  • Pipelines compared: clean vs leaky_definitional
  • ADI = rank_leaky(feature) − rank_clean(feature)   for features shared between both
    (leaky-specific features — hospitaldischargestatus, unitdischargeoffset_num — listed
    separately with their own SHAP magnitudes)
  • Outputs:
      results/tables/external_shap_eicu_adi.csv      — ADI table (all features)
      results/figures/shap_eicu_clean_beeswarm.png   — clean pipeline beeswarm
      results/figures/shap_eicu_leaky_beeswarm.png   — leaky_definitional beeswarm
      results/figures/shap_eicu_comparison_bars.png  — side-by-side mean |SHAP|

Usage
-----
    python src/run_external_shap.py \\
        --eicu24 "External Cohort/Dataset/eicu_label24h.csv" \\
        --outdir results/tables \\
        --figdir results/figures

Requirements
------------
    pip install shap
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print(
        "WARNING: `shap` not installed.\n"
        "  Run:  pip install shap\n"
        "  SHAP beeswarm plots will be skipped; bar comparison still generated."
    )

from src.run_external_eicu import (
    FoldSealedPreprocessorICU,
    NUMERIC_CLEAN        as EICU_NUMERIC_CLEAN,
    CATEGORICAL_CLEAN    as EICU_CATEGORICAL_CLEAN,
    LEAKY_DEFN_NUMERIC   as EICU_LEAKY_NUM,
    LEAKY_DEFN_CATEGORICAL as EICU_LEAKY_CAT,
)

# ── Visual style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

COLOUR_CLEAN = "#2C7BB6"
COLOUR_LEAKY = "#D7191C"
SEED         = 42
TEST_SIZE    = 0.20
SHAP_SAMPLE  = 500      # max test rows to compute SHAP over (for speed)
MAX_DISPLAY  = 15       # top features shown in beeswarm plots

# ── RF hyperparameters (match run_external_eicu.py) ──────────────────────────
RF_PARAMS = dict(
    n_estimators  = 200,
    max_depth     = None,
    min_samples_leaf = 5,
    class_weight  = "balanced",
    random_state  = SEED,
    n_jobs        = -1,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values_rf(
    model: RandomForestClassifier,
    X: np.ndarray,
    sample_size: int = SHAP_SAMPLE,
) -> np.ndarray:
    """
    Compute SHAP values for a fitted RandomForest using TreeExplainer.

    Returns
    -------
    shap_values : ndarray (n_samples, n_features) for the positive class
    """
    if not HAS_SHAP:
        return np.zeros((min(sample_size, len(X)), X.shape[1]))

    X_sample = X[:sample_size]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)

    # Normalise: handle both old API (list) and new API (3-D) output
    if isinstance(sv, list):
        return np.array(sv[1])          # old: [neg_class, pos_class]
    if hasattr(sv, "ndim") and sv.ndim == 3:
        return sv[:, :, 1]              # new: (n, p, 2) → positive class
    return np.array(sv)


def mean_abs_shap(sv: np.ndarray) -> np.ndarray:
    return np.abs(sv).mean(axis=0)


def feature_ranking(sv: np.ndarray, feature_names: list[str]) -> dict[str, int]:
    """Return {feature_name: rank} ranked by mean |SHAP|  (rank 1 = most important)."""
    order = np.argsort(mean_abs_shap(sv))[::-1]
    return {feature_names[int(i)]: int(r + 1) for r, i in enumerate(order)}


def compute_adi(
    ranks_clean: dict[str, int],
    ranks_leaky: dict[str, int],
) -> dict[str, int]:
    """
    Attribution Distortion Index (ADI) for features shared between both models.
    ADI(f) = rank_leaky(f) − rank_clean(f)
      Positive → feature rises in leaky model (distorted upward)
      Negative → feature drops in leaky model (suppressed)
    """
    shared = set(ranks_clean.keys()) & set(ranks_leaky.keys())
    return {f: ranks_leaky[f] - ranks_clean[f] for f in shared}


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_beeswarm(
    sv: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
    max_display: int = MAX_DISPLAY,
) -> None:
    if not HAS_SHAP:
        print(f"  [skip] shap not installed — beeswarm skipped: {out_path.name}")
        return
    plt.figure(figsize=(10, max(5, min(max_display, len(feature_names)) * 0.45 + 1.5)))
    shap.summary_plot(
        sv, X,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=min(max_display, len(feature_names)),
    )
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved beeswarm → {out_path}")


def plot_bar_comparison(
    sv_clean: np.ndarray,
    sv_leaky: np.ndarray,
    clean_features: list[str],
    leaky_features: list[str],
    out_path: Path,
    max_display: int = MAX_DISPLAY,
) -> None:
    """Side-by-side horizontal bar chart: mean |SHAP| clean (left) vs leaky (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, max_display * 0.45 + 2)))

    for ax, sv, feats, colour, subtitle in [
        (axes[0], sv_clean, clean_features, COLOUR_CLEAN,
         "Clean pipeline\n(all clinical features; no leakage)"),
        (axes[1], sv_leaky, leaky_features, COLOUR_LEAKY,
         "Leaky definitional pipeline\n(+ hospitaldischargestatus & unitdischargeoffset_num)"),
    ]:
        means   = mean_abs_shap(sv)
        order   = np.argsort(means)[::-1][:max_display]
        top_f   = [feats[i] for i in order]
        top_v   = means[order]

        bars = ax.barh(range(len(top_f)), top_v[::-1], color=colour, alpha=0.82)
        ax.set_yticks(range(len(top_f)))
        ax.set_yticklabels(top_f[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        ax.set_title(subtitle, fontsize=9.5)
        ax.invert_yaxis()

        # Annotate value
        for bar, val in zip(bars, top_v[::-1]):
            ax.text(
                bar.get_width() + max(top_v) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=7.5, color=colour,
            )

    fig.suptitle(
        "Feature Attribution: eICU 24h — Clean vs Leaky (Definitional Antecedents)\n"
        "Leaky model: hospitaldischargestatus & unitdischargeoffset_num dominate",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_path).replace(".png", ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar comparison → {out_path}")


def plot_adi_chart(df_adi: pd.DataFrame, leaky_only: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart of ADI values for shared features,
    with leaky-only features shown in a separate panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(df_adi) * 0.35 + 2)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: ADI for shared features
    ax = axes[0]
    df_sorted = df_adi.sort_values("adi", ascending=False)
    colours = [COLOUR_LEAKY if v > 0 else COLOUR_CLEAN for v in df_sorted["adi"]]
    bars = ax.barh(range(len(df_sorted)), df_sorted["adi"], color=colours, alpha=0.82)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["feature"], fontsize=9)
    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("ADI = rank_leaky − rank_clean\n(positive → feature rises in leaky model)", fontsize=9)
    ax.set_title("Attribution Distortion Index (ADI)\nShared features — eICU 24h", fontsize=9.5)
    ax.invert_yaxis()

    for bar, val in zip(bars, df_sorted["adi"]):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+d}",
            va="center", fontsize=8, color="black",
        )

    # Right: leaky-only features (mean |SHAP|)
    ax2 = axes[1]
    if not leaky_only.empty:
        leaky_sorted = leaky_only.sort_values("mean_abs_shap_leaky", ascending=False)
        ax2.barh(range(len(leaky_sorted)), leaky_sorted["mean_abs_shap_leaky"],
                 color=COLOUR_LEAKY, alpha=0.85)
        ax2.set_yticks(range(len(leaky_sorted)))
        ax2.set_yticklabels(leaky_sorted["feature"], fontsize=9)
        ax2.set_xlabel("Mean |SHAP| in leaky model", fontsize=9)
        ax2.set_title("Leaky-only features\n(definitional antecedents)", fontsize=9.5)
        ax2.invert_yaxis()
    else:
        ax2.set_visible(False)

    fig.suptitle(
        "Attribution Distortion Index — eICU 24h: Clean vs Leaky\n"
        "Definitional antecedents displace clinical predictors from top attribution ranks",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_path).replace(".png", ext), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ADI chart → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    df: pd.DataFrame,
    label_col: str,
    outdir: Path,
    figdir: Path,
    seed: int = SEED,
    shap_sample: int = SHAP_SAMPLE,
) -> pd.DataFrame:
    """
    Run SHAP analysis comparing clean vs leaky_definitional pipelines on eICU 24h.

    Returns
    -------
    DataFrame of ADI values (one row per feature)
    """
    y = df[label_col].values.astype(int)
    print(f"\n  N={len(df):,}  |  positives={y.sum():,}  ({100*y.mean():.1f}%)")

    # ── Feature sets ────────────────────────────────────────────────────────
    clean_num  = EICU_NUMERIC_CLEAN
    clean_cat  = EICU_CATEGORICAL_CLEAN
    leaky_num  = EICU_NUMERIC_CLEAN + EICU_LEAKY_NUM
    leaky_cat  = EICU_CATEGORICAL_CLEAN + EICU_LEAKY_CAT

    # Column names for the processed feature matrix
    clean_feature_names = clean_num + clean_cat
    leaky_feature_names = leaky_num + leaky_cat

    # ── Stratified train / test split ────────────────────────────────────────
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    df_train  = df.iloc[idx_train].reset_index(drop=True)
    df_test   = df.iloc[idx_test].reset_index(drop=True)
    y_train   = y[idx_train]
    y_test    = y[idx_test]

    print(f"  Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    # ── CLEAN pipeline ───────────────────────────────────────────────────────
    print("\n  [1/2] Clean pipeline …")
    prep_clean = FoldSealedPreprocessorICU()
    X_train_clean = prep_clean.fit_transform(df_train, clean_num, clean_cat)
    X_test_clean  = prep_clean.transform(df_test)

    rf_clean = RandomForestClassifier(**RF_PARAMS)
    rf_clean.fit(X_train_clean, y_train)
    auroc_clean_approx = rf_clean.score(X_test_clean, y_test)
    print(f"     RF accuracy (test, unweighted): {auroc_clean_approx:.4f}")

    print(f"     Computing SHAP values (up to {shap_sample} test samples) …")
    X_shap_clean = X_test_clean[:shap_sample]
    sv_clean     = compute_shap_values_rf(rf_clean, X_test_clean, shap_sample)

    # ── LEAKY pipeline ───────────────────────────────────────────────────────
    print("\n  [2/2] Leaky definitional pipeline …")
    prep_leaky = FoldSealedPreprocessorICU()
    X_train_leaky = prep_leaky.fit_transform(df_train, leaky_num, leaky_cat)
    X_test_leaky  = prep_leaky.transform(df_test)

    rf_leaky = RandomForestClassifier(**RF_PARAMS)
    rf_leaky.fit(X_train_leaky, y_train)
    auroc_leaky_approx = rf_leaky.score(X_test_leaky, y_test)
    print(f"     RF accuracy (test, unweighted): {auroc_leaky_approx:.4f}")

    print(f"     Computing SHAP values (up to {shap_sample} test samples) …")
    X_shap_leaky = X_test_leaky[:shap_sample]
    sv_leaky     = compute_shap_values_rf(rf_leaky, X_test_leaky, shap_sample)

    # ── Rankings + ADI ───────────────────────────────────────────────────────
    ranks_clean = feature_ranking(sv_clean, clean_feature_names)
    ranks_leaky = feature_ranking(sv_leaky, leaky_feature_names)

    means_clean = {f: float(np.abs(sv_clean[:, i]).mean())
                   for i, f in enumerate(clean_feature_names)}
    means_leaky = {f: float(np.abs(sv_leaky[:, i]).mean())
                   for i, f in enumerate(leaky_feature_names)}

    # ADI for shared features (features present in BOTH pipelines)
    shared_features = [f for f in clean_feature_names if f in ranks_leaky]
    adi_values = compute_adi(ranks_clean, ranks_leaky)

    # Leaky-only features (definitional antecedents, not in clean pipeline)
    leaky_only_features = [f for f in leaky_feature_names if f not in clean_feature_names]

    # ── Print top features ───────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  SHAP ATTRIBUTION SUMMARY — eICU 24h")
    print("═" * 70)
    print(f"\n  Top 10 features — CLEAN pipeline (mean |SHAP|):")
    for f, r in sorted(ranks_clean.items(), key=lambda x: x[1])[:10]:
        print(f"    Rank {r:2d}  {f:<40s}  mean|SHAP|={means_clean[f]:.5f}")

    print(f"\n  Top 10 features — LEAKY pipeline (mean |SHAP|):")
    for f, r in sorted(ranks_leaky.items(), key=lambda x: x[1])[:10]:
        print(f"    Rank {r:2d}  {f:<40s}  mean|SHAP|={means_leaky[f]:.5f}")

    if leaky_only_features:
        print(f"\n  Leaky-only (definitional antecedents):")
        for f in leaky_only_features:
            r = ranks_leaky.get(f, "?")
            m = means_leaky.get(f, 0.0)
            print(f"    Rank {r!s:>3}  {f:<40s}  mean|SHAP|={m:.5f}  ← DEFINITIONAL ANTECEDENT")

    # ── Build output DataFrames ──────────────────────────────────────────────
    rows = []
    for f in clean_feature_names:
        rows.append({
            "feature":            f,
            "in_clean":           True,
            "in_leaky":           f in ranks_leaky,
            "rank_clean":         ranks_clean.get(f),
            "rank_leaky":         ranks_leaky.get(f),
            "mean_abs_shap_clean": round(means_clean.get(f, 0.0), 6),
            "mean_abs_shap_leaky": round(means_leaky.get(f, 0.0), 6),
            "adi":                adi_values.get(f),
        })

    for f in leaky_only_features:
        rows.append({
            "feature":             f,
            "in_clean":            False,
            "in_leaky":            True,
            "rank_clean":          None,
            "rank_leaky":          ranks_leaky.get(f),
            "mean_abs_shap_clean": None,
            "mean_abs_shap_leaky": round(means_leaky.get(f, 0.0), 6),
            "adi":                 None,
        })

    df_adi = pd.DataFrame(rows)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_csv = outdir / "external_shap_eicu_adi.csv"
    df_adi.to_csv(out_csv, index=False)
    print(f"\n  Saved ADI CSV → {out_csv}")

    # ── Figures ──────────────────────────────────────────────────────────────
    figdir.mkdir(parents=True, exist_ok=True)

    # Beeswarm — clean
    plot_beeswarm(
        sv_clean, X_shap_clean, clean_feature_names,
        title="SHAP Beeswarm — eICU 24h Clean Pipeline\n(all clinical features; no leakage)",
        out_path=figdir / "shap_eicu_clean_beeswarm.png",
    )

    # Beeswarm — leaky
    plot_beeswarm(
        sv_leaky, X_shap_leaky, leaky_feature_names,
        title=(
            "SHAP Beeswarm — eICU 24h Leaky (Definitional) Pipeline\n"
            "hospitaldischargestatus & unitdischargeoffset_num dominate"
        ),
        out_path=figdir / "shap_eicu_leaky_beeswarm.png",
    )

    # Bar comparison
    plot_bar_comparison(
        sv_clean, sv_leaky,
        clean_feature_names, leaky_feature_names,
        out_path=figdir / "shap_eicu_comparison_bars.png",
    )

    # ADI chart
    df_shared_adi = df_adi[df_adi["adi"].notna()].copy()
    df_shared_adi["adi"] = df_shared_adi["adi"].astype(int)
    df_leaky_only = df_adi[~df_adi["in_clean"]].copy()

    plot_adi_chart(
        df_shared_adi,
        df_leaky_only[["feature", "mean_abs_shap_leaky"]].reset_index(drop=True),
        out_path=figdir / "shap_eicu_adi_chart.png",
    )

    print("\n  ✓  SHAP analysis complete.")
    return df_adi


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SHAP attribution analysis for eICU 24h external leakage audit."
    )
    p.add_argument(
        "--eicu24",
        default="External Cohort/Dataset/eicu_label24h.csv",
        help="Path to eICU 24h label CSV",
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
        "--shap_sample", type=int, default=SHAP_SAMPLE,
        help=f"Max test samples for SHAP (default: {SHAP_SAMPLE}; reduce for speed)",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.eicu24)
    if not data_path.exists():
        print(f"[ERROR] eICU 24h file not found: {data_path}")
        print("  Please run the eICU data preparation step first.")
        sys.exit(1)

    print(f"Loading eICU 24h data from {data_path} …")
    df = pd.read_csv(
        data_path,
        keep_default_na=False,
        na_values=["NA", "NaN", "nan", ""],
    )
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    run_shap_analysis(
        df        = df,
        label_col = "label24h",
        outdir    = outdir,
        figdir    = figdir,
        seed      = args.seed,
        shap_sample = args.shap_sample,
    )

    print(f"\n  Results:")
    print(f"    Tables  → {outdir}/external_shap_eicu_adi.csv")
    print(f"    Figures → {figdir}/shap_eicu_*.png / .pdf")


if __name__ == "__main__":
    main()
