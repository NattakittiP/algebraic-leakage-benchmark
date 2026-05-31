"""SHAP attribution + ADI analysis for CGMacros postprandial glucose leakage audit."""

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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

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

from src.run_external_cgmacros import (
    FoldSealedPreprocessorCGMacros,
    NUMERIC_CLEAN_BASE,
    CATEGORICAL_CLEAN,
    LEAKY_PEAK_NUMERIC,
    LEAKY_PEAK_CATEGORICAL,
    LABEL_PERCENTILE,
)

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
TEST_SIZE    = 0.20        # 20% of subjects held out as test
SHAP_SAMPLE  = 500         # max test rows to compute SHAP over
MAX_DISPLAY  = 20          # top features shown in beeswarm / bar plots

RF_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = None,
    min_samples_leaf = 5,
    class_weight     = "balanced",
    random_state     = SEED,
    n_jobs           = -1,
)


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
        return np.array(sv[1])          # old shap: [neg_class, pos_class]
    if hasattr(sv, "ndim") and sv.ndim == 3:
        return sv[:, :, 1]              # new shap: (n, p, 2) → positive class
    return np.array(sv)


def mean_abs_shap(sv: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP value per feature."""
    return np.abs(sv).mean(axis=0)


def feature_ranking(sv: np.ndarray, feature_names: list[str]) -> dict[str, int]:
    """Return {feature_name: rank} by mean |SHAP| (rank 1 = most important)."""
    order = np.argsort(mean_abs_shap(sv))[::-1]
    return {feature_names[int(i)]: int(r + 1) for r, i in enumerate(order)}


def compute_adi(
    ranks_clean: dict[str, int],
    ranks_leaky: dict[str, int],
) -> dict[str, int]:
    """
    Attribution Distortion Index (ADI) for features shared between both models.
    ADI(f) = rank_leaky(f) − rank_clean(f)
      Positive → feature rises in leaky model (pushed upward by leakage)
      Negative → feature falls in leaky model (suppressed by leakage)
    """
    shared = set(ranks_clean.keys()) & set(ranks_leaky.keys())
    return {f: ranks_leaky[f] - ranks_clean[f] for f in shared}


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    """Save figure as both PNG and PDF."""
    for ext in (".png", ".pdf"):
        p = out_path.with_suffix(ext)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path.with_suffix('.pdf')} / .png")


def plot_beeswarm(
    sv: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
    max_display: int = MAX_DISPLAY,
) -> None:
    """SHAP beeswarm (dot) plot for a single pipeline."""
    if not HAS_SHAP:
        print(f"  [skip] shap not installed — beeswarm skipped: {out_path.name}")
        return
    n_show = min(max_display, len(feature_names))
    fig = plt.figure(figsize=(10, max(5, n_show * 0.45 + 1.5)))
    shap.summary_plot(
        sv, X,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=n_show,
    )
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_bar_comparison(
    sv_clean: np.ndarray,
    sv_leaky: np.ndarray,
    clean_features: list[str],
    leaky_features: list[str],
    out_path: Path,
    max_display: int = MAX_DISPLAY,
) -> None:
    """Side-by-side horizontal bar chart: mean |SHAP| clean vs leaky."""
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, max_display * 0.45 + 2)))

    for ax, sv, feats, colour, subtitle in [
        (
            axes[0], sv_clean, clean_features, COLOUR_CLEAN,
            "Clean pipeline\n(pre-meal covariates; no leakage)",
        ),
        (
            axes[1], sv_leaky, leaky_features, COLOUR_LEAKY,
            "Leaky pipeline\n(+ peak_cgm = B, the algebraic component)",
        ),
    ]:
        means = mean_abs_shap(sv)
        order = np.argsort(means)[::-1][:max_display]
        top_f = [feats[i] for i in order]
        top_v = means[order]

        bars = ax.barh(range(len(top_f)), top_v[::-1], color=colour, alpha=0.82)
        ax.set_yticks(range(len(top_f)))
        ax.set_yticklabels(top_f[::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        ax.set_title(subtitle, fontsize=9.5)
        ax.invert_yaxis()

        x_pad = max(top_v) * 0.01
        for bar, val in zip(bars, top_v[::-1]):
            ax.text(
                bar.get_width() + x_pad,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=7.5, color=colour,
            )

    fig.suptitle(
        "Feature Attribution Comparison: CGMacros — Clean vs Leaky (peak_cgm)\n"
        "Leaky model: peak_cgm dominates because Y = peak_cgm − pre_meal_cgm",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_adi_chart(
    df_adi: pd.DataFrame,
    df_leaky_only: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Horizontal bar chart of ADI values for shared features,
    with leaky-only features shown in a separate panel.

    df_adi         : rows with adi != None (shared features)
    df_leaky_only  : rows with in_clean=False (leaky-only features)
    """
    n_shared     = len(df_adi)
    n_leaky_only = len(df_leaky_only)
    has_leaky    = n_leaky_only > 0

    width_ratios = [4, 1] if has_leaky else [1]
    n_panels     = 2 if has_leaky else 1

    fig, axes_raw = plt.subplots(
        1, n_panels,
        figsize=(15 if has_leaky else 9, max(5, n_shared * 0.38 + 2)),
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = [axes_raw] if n_panels == 1 else list(axes_raw)

    ax = axes[0]
    df_sorted = df_adi.sort_values("adi", ascending=False).reset_index(drop=True)
    colours   = [COLOUR_LEAKY if v > 0 else COLOUR_CLEAN for v in df_sorted["adi"]]

    bars = ax.barh(range(len(df_sorted)), df_sorted["adi"], color=colours, alpha=0.82)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["feature"], fontsize=8)
    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel(
        "ADI = rank_leaky − rank_clean\n"
        "(positive → feature rank rises in leaky model; negative → suppressed)",
        fontsize=9,
    )
    ax.set_title(
        "Attribution Distortion Index (ADI)\n"
        "Shared features — CGMacros postprandial glucose",
        fontsize=9.5,
    )
    ax.invert_yaxis()

    for bar, val in zip(bars, df_sorted["adi"]):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{int(val):+d}",
            va="center", fontsize=8, color="black",
        )

    if has_leaky:
        ax2 = axes[1]
        lo_sorted = df_leaky_only.sort_values(
            "mean_abs_shap_leaky", ascending=False
        ).reset_index(drop=True)
        ax2.barh(
            range(len(lo_sorted)), lo_sorted["mean_abs_shap_leaky"],
            color=COLOUR_LEAKY, alpha=0.85,
        )
        ax2.set_yticks(range(len(lo_sorted)))
        ax2.set_yticklabels(lo_sorted["feature"], fontsize=9)
        ax2.set_xlabel("Mean |SHAP| in leaky model", fontsize=9)
        ax2.set_title(
            "Leaky-only feature\n(algebraic antecedent B)", fontsize=9.5
        )
        ax2.invert_yaxis()

        for idx, row in lo_sorted.iterrows():
            ax2.text(
                row["mean_abs_shap_leaky"] * 1.02,
                idx,
                f"{row['mean_abs_shap_leaky']:.4f}",
                va="center", fontsize=8, color=COLOUR_LEAKY,
            )

    fig.suptitle(
        "Attribution Distortion Index — CGMacros: Clean vs Leaky (peak_cgm)\n"
        "peak_cgm displaces physiological predictors from top attribution ranks",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def subject_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split meal-level rows into train/test while keeping each subject's
    meals entirely in one partition (prevents patient-level leakage).

    Uses GroupShuffleSplit to randomly assign ~80% of subjects to train,
    ~20% to test.

    Returns
    -------
    (train_idx, test_idx) — integer arrays of row indices
    """
    groups = df["subject_id"].values
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return train_idx, test_idx


def run_shap_analysis(
    df: pd.DataFrame,
    gut_cols: list[str],
    outdir: Path,
    figdir: Path,
    seed: int = SEED,
    shap_sample: int = SHAP_SAMPLE,
) -> pd.DataFrame:
    """
    Run SHAP analysis comparing clean vs leaky_peak_cgm pipelines on CGMacros.

    Parameters
    ----------
    df        : meal-level cohort (from cgmacros_meal_cohort.csv)
    gut_cols  : list of gut health column names (detected at runtime)
    outdir    : directory for output CSVs
    figdir    : directory for output figures
    seed      : random seed for GroupShuffleSplit
    shap_sample : max test observations for SHAP computation

    Returns
    -------
    DataFrame of ADI values (one row per feature)
    """
    print(f"\n  Total: {len(df):,} meals  |  {df['subject_id'].nunique()} subjects")

    clean_num  = [c for c in (NUMERIC_CLEAN_BASE + gut_cols) if c in df.columns]
    clean_cat  = [c for c in CATEGORICAL_CLEAN if c in df.columns]
    leaky_num  = [c for c in (NUMERIC_CLEAN_BASE + gut_cols + LEAKY_PEAK_NUMERIC) if c in df.columns]
    leaky_cat  = [c for c in (CATEGORICAL_CLEAN + LEAKY_PEAK_CATEGORICAL) if c in df.columns]

    clean_feature_names = clean_num + clean_cat
    leaky_feature_names = leaky_num + leaky_cat

    print(f"  Clean feature set:  {len(clean_feature_names)} features")
    print(f"  Leaky feature set:  {len(leaky_feature_names)} features "
          f"(+{len(leaky_feature_names) - len(clean_feature_names)} leaky)")

    print(f"\n  Splitting by subject (seed={seed}, test_size={TEST_SIZE:.0%}) …")
    train_idx, test_idx = subject_split(df, TEST_SIZE, seed)

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    n_subj_train = df_train["subject_id"].nunique()
    n_subj_test  = df_test["subject_id"].nunique()
    print(f"  Train: {len(df_train):,} meals ({n_subj_train} subjects)  |  "
          f"Test: {len(df_test):,} meals ({n_subj_test} subjects)")

    y_rise_train = df_train["glucose_rise"].values
    y_rise_test  = df_test["glucose_rise"].values
    threshold    = float(np.percentile(y_rise_train, LABEL_PERCENTILE))

    y_train = (y_rise_train >= threshold).astype(int)
    y_test  = (y_rise_test  >= threshold).astype(int)

    print(f"\n  Fold-sealed Q{LABEL_PERCENTILE:.0f} threshold = {threshold:.2f} mg/dL")
    print(f"  Train positives: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    print(f"  Test  positives: {y_test.sum()}  ({100*y_test.mean():.1f}%)")

    if len(np.unique(y_test)) < 2:
        print("  [WARNING] Test set has only one class — SHAP analysis may be degenerate.")

    print("\n  [1/2] Fitting CLEAN pipeline …")
    prep_clean    = FoldSealedPreprocessorCGMacros()
    X_train_clean = prep_clean.fit_transform(df_train, clean_num, clean_cat)
    X_test_clean  = prep_clean.transform(df_test)

    rf_clean = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
    rf_clean.fit(X_train_clean, y_train)
    print(f"     RF trained on {len(X_train_clean):,} training observations.")

    try:
        from sklearn.metrics import roc_auc_score
        y_prob_clean = rf_clean.predict_proba(X_test_clean)[:, 1]
        auroc_clean  = float(roc_auc_score(y_test, y_prob_clean))
        print(f"     RF AUROC (test): {auroc_clean:.4f}")
    except Exception:
        auroc_clean = float("nan")

    print(f"     Computing SHAP values (up to {shap_sample} test samples) …")
    sv_clean     = compute_shap_values_rf(rf_clean, X_test_clean, shap_sample)
    X_shap_clean = X_test_clean[:min(shap_sample, len(X_test_clean))]

    print("\n  [2/2] Fitting LEAKY pipeline (+ peak_cgm = B) …")
    prep_leaky    = FoldSealedPreprocessorCGMacros()
    X_train_leaky = prep_leaky.fit_transform(df_train, leaky_num, leaky_cat)
    X_test_leaky  = prep_leaky.transform(df_test)

    rf_leaky = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
    rf_leaky.fit(X_train_leaky, y_train)
    print(f"     RF trained on {len(X_train_leaky):,} training observations.")

    try:
        y_prob_leaky = rf_leaky.predict_proba(X_test_leaky)[:, 1]
        auroc_leaky  = float(roc_auc_score(y_test, y_prob_leaky))
        print(f"     RF AUROC (test): {auroc_leaky:.4f}")
    except Exception:
        auroc_leaky = float("nan")

    print(f"     Computing SHAP values (up to {shap_sample} test samples) …")
    sv_leaky     = compute_shap_values_rf(rf_leaky, X_test_leaky, shap_sample)
    X_shap_leaky = X_test_leaky[:min(shap_sample, len(X_test_leaky))]

    # Defensive guard against edge cases in SHAP output shape
    if sv_clean.shape[1] != len(clean_feature_names):
        n = min(sv_clean.shape[1], len(clean_feature_names))
        print(f"  [WARNING] sv_clean.shape[1]={sv_clean.shape[1]} != "
              f"len(clean_feature_names)={len(clean_feature_names)}; truncating to {n}")
        sv_clean          = sv_clean[:, :n]
        clean_feature_names = clean_feature_names[:n]
    if sv_leaky.shape[1] != len(leaky_feature_names):
        n = min(sv_leaky.shape[1], len(leaky_feature_names))
        print(f"  [WARNING] sv_leaky.shape[1]={sv_leaky.shape[1]} != "
              f"len(leaky_feature_names)={len(leaky_feature_names)}; truncating to {n}")
        sv_leaky          = sv_leaky[:, :n]
        leaky_feature_names = leaky_feature_names[:n]

    ranks_clean = feature_ranking(sv_clean, clean_feature_names)
    ranks_leaky = feature_ranking(sv_leaky, leaky_feature_names)

    means_clean = {
        f: float(np.abs(sv_clean[:, i]).mean())
        for i, f in enumerate(clean_feature_names)
    }
    means_leaky = {
        f: float(np.abs(sv_leaky[:, i]).mean())
        for i, f in enumerate(leaky_feature_names)
    }

    adi_values        = compute_adi(ranks_clean, ranks_leaky)
    leaky_only_feats  = [f for f in leaky_feature_names if f not in clean_feature_names]

    print(f"\n  [DEBUG] clean_feature_names ({len(clean_feature_names)}): "
          f"{clean_feature_names[:3]} ... {clean_feature_names[-3:]}")
    print(f"  [DEBUG] leaky_feature_names ({len(leaky_feature_names)}): "
          f"{leaky_feature_names[:3]} ... {leaky_feature_names[-3:]}")
    print(f"  [DEBUG] leaky_only_feats: {leaky_only_feats}")
    print(f"  [DEBUG] peak_cgm in clean_feature_names: {'peak_cgm' in clean_feature_names}")
    print(f"  [DEBUG] peak_cgm in leaky_feature_names: {'peak_cgm' in leaky_feature_names}")
    print(f"  [DEBUG] len(ranks_leaky)={len(ranks_leaky)}, "
          f"peak_cgm rank={ranks_leaky.get('peak_cgm', 'NOT IN RANKS')}")
    print(f"  [DEBUG] sv_clean.shape={sv_clean.shape}, sv_leaky.shape={sv_leaky.shape}")

    # If peak_cgm was in leaky_num, recover its SHAP values directly from the column index
    # (independent of any feature-name mapping issue)
    peak_cgm_direct: dict | None = None
    if "peak_cgm" in leaky_num:
        pk_col = leaky_num.index("peak_cgm")
        if pk_col < sv_leaky.shape[1]:
            pk_shap      = float(np.abs(sv_leaky[:, pk_col]).mean())
            all_means_lk = np.abs(sv_leaky).mean(axis=0)
            pk_rank      = int((all_means_lk > all_means_lk[pk_col]).sum()) + 1
            peak_cgm_direct = {
                "feature":             "peak_cgm",
                "in_clean":            False,
                "in_leaky":            True,
                "rank_clean":          None,
                "rank_leaky":          pk_rank,
                "mean_abs_shap_clean": None,
                "mean_abs_shap_leaky": round(pk_shap, 6),
                "adi":                 None,
                "is_leaky_antecedent": True,
            }
            print(f"  [DIRECT] peak_cgm found at leaky_num col {pk_col}: "
                  f"rank={pk_rank}, mean|SHAP|={pk_shap:.5f}")
        else:
            print(f"  [WARNING] peak_cgm leaky_num index {pk_col} out of "
                  f"sv_leaky column range ({sv_leaky.shape[1]})")

    print("\n" + "═" * 70)
    print("  SHAP ATTRIBUTION SUMMARY — CGMacros")
    print("═" * 70)

    print(f"\n  Top 10 features — CLEAN pipeline (mean |SHAP|):")
    for f, r in sorted(ranks_clean.items(), key=lambda x: x[1])[:10]:
        print(f"    Rank {r:2d}  {f:<45s}  mean|SHAP|={means_clean[f]:.5f}")

    print(f"\n  Top 10 features — LEAKY pipeline (mean |SHAP|):")
    for f, r in sorted(ranks_leaky.items(), key=lambda x: x[1])[:10]:
        m = means_leaky.get(f, 0.0)
        tag = "  ← ALGEBRAIC ANTECEDENT (B)" if f in leaky_only_feats else ""
        print(f"    Rank {r:2d}  {f:<45s}  mean|SHAP|={m:.5f}{tag}")

    if leaky_only_feats:
        print(f"\n  Leaky-only features (algebraic antecedents):")
        for f in leaky_only_feats:
            r = ranks_leaky.get(f, "?")
            m = means_leaky.get(f, 0.0)
            print(f"    Rank {r!s:>3}  {f:<45s}  mean|SHAP|={m:.5f}  ← B = peak_cgm")

    print(f"\n  Top 10 ADI distortions (shared features):")
    adi_sorted = sorted(adi_values.items(), key=lambda x: abs(x[1]), reverse=True)
    for f, adi in adi_sorted[:10]:
        direction = "↑ rises" if adi > 0 else "↓ falls"
        print(f"    ADI={adi:+4d}  {f:<45s}  ({direction} in leaky model)")

    rows = []

    for f in clean_feature_names:
        rows.append({
            "feature":             f,
            "in_clean":            True,
            "in_leaky":            f in ranks_leaky,
            "rank_clean":          ranks_clean.get(f),
            "rank_leaky":          ranks_leaky.get(f),
            "mean_abs_shap_clean": round(means_clean.get(f, 0.0), 6),
            "mean_abs_shap_leaky": round(means_leaky.get(f, 0.0), 6),
            "adi":                 adi_values.get(f),
            "is_leaky_antecedent": False,
        })

    for f in leaky_only_feats:
        if f == "peak_cgm" and peak_cgm_direct is not None:
            rows.append(peak_cgm_direct)        # use direct column-index recovery
        else:
            rows.append({
                "feature":             f,
                "in_clean":            False,
                "in_leaky":            True,
                "rank_clean":          None,
                "rank_leaky":          ranks_leaky.get(f),
                "mean_abs_shap_clean": None,
                "mean_abs_shap_leaky": round(means_leaky.get(f, 0.0), 6),
                "adi":                 None,
                "is_leaky_antecedent": True,
            })

    # Safety net 1: features in ranks_leaky not yet written
    already_written = {r["feature"] for r in rows}
    clean_feature_set = set(clean_feature_names)
    for f in sorted(ranks_leaky.keys(), key=lambda x: ranks_leaky[x]):
        if f not in already_written and f not in clean_feature_set:
            print(f"  [safety-net-1] Adding leaky-only feature: {f!r}")
            rows.append({
                "feature":             f,
                "in_clean":            False,
                "in_leaky":            True,
                "rank_clean":          None,
                "rank_leaky":          ranks_leaky.get(f),
                "mean_abs_shap_clean": None,
                "mean_abs_shap_leaky": round(means_leaky.get(f, 0.0), 6),
                "adi":                 None,
                "is_leaky_antecedent": True,
            })

    # Safety net 2: peak_cgm column-index direct recovery (last resort)
    already_written = {r["feature"] for r in rows}
    if "peak_cgm" not in already_written and peak_cgm_direct is not None:
        print("  [safety-net-2] peak_cgm not captured by any loop — using direct recovery.")
        rows.append(peak_cgm_direct)

    df_adi = pd.DataFrame(rows)

    df_adi["cohort"]       = "CGMacros"
    df_adi["n_train_meals"] = len(df_train)
    df_adi["n_test_meals"]  = len(df_test)
    df_adi["n_train_subj"]  = n_subj_train
    df_adi["n_test_subj"]   = n_subj_test
    df_adi["auroc_clean"]   = round(auroc_clean, 4) if not np.isnan(auroc_clean) else None
    df_adi["auroc_leaky"]   = round(auroc_leaky, 4) if not np.isnan(auroc_leaky) else None
    df_adi["label_threshold_mg_dl"] = round(threshold, 2)
    df_adi["label"] = f"high_responder_Q{LABEL_PERCENTILE:.0f}"

    out_csv = outdir / "cgmacros_shap_adi.csv"
    df_adi.to_csv(out_csv, index=False)
    print(f"\n  Saved ADI CSV → {out_csv}")

    figdir.mkdir(parents=True, exist_ok=True)

    plot_beeswarm(
        sv_clean, X_shap_clean, clean_feature_names,
        title=(
            "SHAP Beeswarm — CGMacros Clean Pipeline\n"
            "(pre-meal covariates: macros, demographics, gut scores; no leakage)"
        ),
        out_path=figdir / "shap_cgmacros_clean_beeswarm.png",
        max_display=MAX_DISPLAY,
    )

    plot_beeswarm(
        sv_leaky, X_shap_leaky, leaky_feature_names,
        title=(
            "SHAP Beeswarm — CGMacros Leaky Pipeline (+ peak_cgm = B)\n"
            "peak_cgm dominates because Y = peak_cgm − pre_meal_cgm"
        ),
        out_path=figdir / "shap_cgmacros_leaky_beeswarm.png",
        max_display=MAX_DISPLAY,
    )

    plot_bar_comparison(
        sv_clean, sv_leaky,
        clean_feature_names, leaky_feature_names,
        out_path=figdir / "shap_cgmacros_comparison_bars.png",
        max_display=MAX_DISPLAY,
    )

    df_shared_adi = df_adi[df_adi["adi"].notna()].copy()
    df_shared_adi = df_shared_adi.assign(adi=df_shared_adi["adi"].astype(int))
    df_leaky_only = df_adi[~df_adi["in_clean"]][
        ["feature", "mean_abs_shap_leaky"]
    ].reset_index(drop=True)

    plot_adi_chart(
        df_shared_adi,
        df_leaky_only,
        out_path=figdir / "shap_cgmacros_adi_chart.png",
    )

    print("\n  ✓  CGMacros SHAP analysis complete.")
    return df_adi


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "SHAP attribution + ADI analysis for CGMacros postprandial glucose "
            "leakage audit.\n"
            "Demonstrates algebraic feature attribution distortion when peak_cgm "
            "(the definitional antecedent B) is included."
        )
    )
    p.add_argument(
        "--data",
        default="results/tables/cgmacros_meal_cohort.csv",
        help="Path to cgmacros_meal_cohort.csv (output of build_cgmacros_cohort.py)",
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
        help=f"Random seed for GroupShuffleSplit (default: {SEED})",
    )
    p.add_argument(
        "--shap_sample", type=int, default=SHAP_SAMPLE,
        help=f"Max test samples for SHAP computation (default: {SHAP_SAMPLE}; reduce for speed)",
    )
    p.add_argument(
        "--max_display", type=int, default=MAX_DISPLAY,
        help=f"Max features to display in plots (default: {MAX_DISPLAY})",
    )
    return p.parse_args()


def main():
    global MAX_DISPLAY
    args   = parse_args()
    MAX_DISPLAY = args.max_display

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Cohort file not found: {data_path}")
        print("  Please run build_cgmacros_cohort.py first:")
        print("    python src/build_cgmacros_cohort.py "
              "--data_dir CGMacros_dateshifted365/CGMacros_Dataset --outdir results/tables")
        sys.exit(1)

    print("=" * 70)
    print("  CGMacros SHAP Attribution + ADI Analysis")
    print("  Algebraic antecedent: peak_cgm (B), where Y = B − pre_meal_cgm (A)")
    print("=" * 70)

    print(f"\nLoading cohort from: {data_path}")
    df = pd.read_csv(
        data_path,
        na_values=["", "nan", "NaN", "NA", "N/A"],
        keep_default_na=True,
    )
    print(f"  Loaded: {len(df):,} meals, {df['subject_id'].nunique()} subjects, "
          f"{df.shape[1]} columns")

    gut_cols = sorted([c for c in df.columns if c.startswith("gut_")])
    print(f"  Gut health columns: {len(gut_cols)}")

    df_adi = run_shap_analysis(
        df          = df,
        gut_cols    = gut_cols,
        outdir      = outdir,
        figdir      = figdir,
        seed        = args.seed,
        shap_sample = args.shap_sample,
    )

    print(f"\n{'─'*70}")
    print("  Outputs:")
    print(f"    Table   → {outdir}/cgmacros_shap_adi.csv")
    print(f"    Figures → {figdir}/shap_cgmacros_*.png / .pdf")
    print(f"{'─'*70}")
    print("\nDone.")


if __name__ == "__main__":
    main()
