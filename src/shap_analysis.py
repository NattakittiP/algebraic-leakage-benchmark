"""
shap_analysis.py — SHAP-based feature attribution analysis.

Key findings this module must demonstrate:
  - Clean model: WBV SHAP ≈ 0 (null scenario)
  - Leaky model (TG4h included): TG4h/TCR dominates SHAP
  - Attribution Distortion metric computed for each feature

Usage:
  python src/shap_analysis.py \\
      --data data/paired_tcr_null_v1_seed2026.csv \\
      --config config/model_config.yaml \\
      --out results/tables/shap_comparison.csv \\
      --figdir results/figures
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
import yaml
from tqdm import tqdm

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. SHAP analysis will be skipped.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import FoldSealedScaler, FoldSealedWinsorizer
from src.run_clean_pipeline import CLEAN_FEATURES
from src.metrics import compute_attribution_distortion

warnings.filterwarnings("ignore")

LEAKY_FEATURES = CLEAN_FEATURES + ["tg4h"]


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: np.ndarray,
    model_name: str = "",
    sample_size: int = 300,
) -> np.ndarray:
    """Compute SHAP values for a fitted model.

    Parameters
    ----------
    model      : fitted sklearn/xgboost model with predict_proba
    X          : feature matrix (already preprocessed)
    model_name : used to choose TreeExplainer vs LinearExplainer
    sample_size: max rows to compute SHAP on (for speed)

    Returns
    -------
    shap_values : array (n_samples, n_features) — values for the positive class
    """
    if not HAS_SHAP:
        return np.zeros((min(sample_size, len(X)), X.shape[1]))

    X_sample = X[:sample_size]

    if "Forest" in model_name or "XGB" in model_name:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
    else:
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X_sample, nsamples=50)

    # Normalise output to 2-D (n_samples, n_features) for positive class.
    # SHAP ≥ 0.40 may return:
    #   list  → [neg_class, pos_class]        (old API)
    #   3-D   → (n_samples, n_features, 2)    (new API, binary)
    #   2-D   → (n_samples, n_features)        (already correct)
    if isinstance(sv, list):
        return sv[1]              # old API: pick positive class
    if sv.ndim == 3:
        return sv[:, :, 1]       # new API: slice positive class
    return sv                     # already 2-D


def get_feature_ranking(shap_values: np.ndarray, feature_names: list[str]) -> dict[str, int]:
    """Return {feature_name: rank} based on mean |SHAP| (rank 1 = most important)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]   # descending
    return {feature_names[int(i)]: int(rank + 1) for rank, i in enumerate(order)}


def compare_shap_clean_vs_leaky(
    shap_values_clean: np.ndarray,
    shap_values_leaky: np.ndarray,
    clean_features: list[str],
    leaky_features: list[str],
) -> dict:
    """Compare SHAP attributions between clean and leaky models.

    Returns
    -------
    dict with ranks, distortions, and WBV SHAP statistics
    """
    ranks_clean = get_feature_ranking(shap_values_clean, clean_features)
    ranks_leaky = get_feature_ranking(shap_values_leaky, leaky_features)

    # Distortion only for features present in both
    shared = {f: ranks_clean[f] for f in clean_features if f in ranks_leaky}
    ranks_leaky_shared = {f: ranks_leaky[f] for f in shared}
    distortion = compute_attribution_distortion(shared, ranks_leaky_shared)

    # WBV SHAP summary
    wbv_idx_clean = clean_features.index("wbv") if "wbv" in clean_features else -1
    wbv_idx_leaky = leaky_features.index("wbv") if "wbv" in leaky_features else -1

    wbv_clean_mean_abs = float(np.abs(shap_values_clean[:, wbv_idx_clean]).mean()) if wbv_idx_clean >= 0 else None
    wbv_leaky_mean_abs = float(np.abs(shap_values_leaky[:, wbv_idx_leaky]).mean()) if wbv_idx_leaky >= 0 else None

    return {
        "ranks_clean": ranks_clean,
        "ranks_leaky": ranks_leaky,
        "attribution_distortion": distortion,
        "wbv_mean_abs_shap_clean": round(wbv_clean_mean_abs, 5) if wbv_clean_mean_abs is not None else None,
        "wbv_mean_abs_shap_leaky": round(wbv_leaky_mean_abs, 5) if wbv_leaky_mean_abs is not None else None,
        "wbv_rank_clean": ranks_clean.get("wbv"),
        "wbv_rank_leaky": ranks_leaky.get("wbv"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
):
    """Save a SHAP beeswarm (summary) plot."""
    if not HAS_SHAP:
        print(f"SHAP not installed — skipping beeswarm plot: {out_path}")
        return

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=min(len(feature_names), 12),
    )
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved beeswarm plot -> {out_path}")


def plot_mean_abs_shap_comparison(
    shap_values_clean: np.ndarray,
    shap_values_leaky: np.ndarray,
    clean_features: list[str],
    leaky_features: list[str],
    out_path: Path,
):
    """Side-by-side bar chart: mean |SHAP| clean vs leaky."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Clean
    clean_mean = np.abs(shap_values_clean).mean(axis=0)
    idx_c = np.argsort(clean_mean)[::-1]
    axes[0].barh(
        [clean_features[i] for i in idx_c],
        clean_mean[idx_c],
        color="steelblue",
    )
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("Clean model\n(WBV ≈ 0 expected)")
    axes[0].invert_yaxis()

    # Leaky
    leaky_mean = np.abs(shap_values_leaky).mean(axis=0)
    idx_l = np.argsort(leaky_mean)[::-1]
    axes[1].barh(
        [leaky_features[i] for i in idx_l],
        leaky_mean[idx_l],
        color="firebrick",
    )
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Leaky model (TG4h included)\n(TG4h dominates)")
    axes[1].invert_yaxis()

    plt.suptitle("Feature Attribution: Clean vs Leaky Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP comparison plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------

def run_shap_analysis(
    df: pd.DataFrame,
    cfg: dict,
    out_csv: Path,
    fig_dir: Path,
    seed: int = 42,
):
    """Train clean and leaky models, compute SHAP, compare attributions."""
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    analysis_steps = [
        "scale clean features",
        "train clean RF",
        "compute SHAP (clean)",
        "scale leaky features",
        "train leaky RF",
        "compute SHAP (leaky)",
        "compare attributions",
        "save CSV",
        "beeswarm (clean)",
        "beeswarm (leaky)",
        "bar comparison",
    ]
    pbar = tqdm(analysis_steps, desc="SHAP analysis", ncols=90, colour="green")

    # --- Clean pipeline ---
    pbar.set_description("SHAP: scale clean features"); pbar.update(0)
    X_clean = df[CLEAN_FEATURES].values
    scaler_c = StandardScaler()
    X_clean_s = scaler_c.fit_transform(X_clean)
    pbar.update(1)

    pbar.set_description("SHAP: train clean RandomForest")
    rf_clean = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf_clean.fit(X_clean_s, y)
    pbar.update(1)

    pbar.set_description("SHAP: computing SHAP values (clean)")
    shap_clean = compute_shap_values(rf_clean, X_clean_s, "RandomForest")
    pbar.update(1)

    # --- Leaky pipeline (TG4h included) ---
    pbar.set_description("SHAP: scale leaky features")
    X_leaky = df[LEAKY_FEATURES].values
    scaler_l = StandardScaler()
    X_leaky_s = scaler_l.fit_transform(X_leaky)
    pbar.update(1)

    pbar.set_description("SHAP: train leaky RandomForest")
    rf_leaky = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf_leaky.fit(X_leaky_s, y)
    pbar.update(1)

    pbar.set_description("SHAP: computing SHAP values (leaky)")
    shap_leaky = compute_shap_values(rf_leaky, X_leaky_s, "RandomForest")
    pbar.update(1)

    # --- Compare ---
    pbar.set_description("SHAP: comparing attributions")
    comparison = compare_shap_clean_vs_leaky(
        shap_clean, shap_leaky, CLEAN_FEATURES, LEAKY_FEATURES
    )
    pbar.update(1)

    tqdm.write("\n=== SHAP Comparison ===")
    tqdm.write(f"  WBV rank (clean model):  {comparison['wbv_rank_clean']}")
    tqdm.write(f"  WBV rank (leaky model):  {comparison['wbv_rank_leaky']}")
    tqdm.write(f"  WBV mean |SHAP| clean:   {comparison['wbv_mean_abs_shap_clean']}")
    tqdm.write(f"  WBV mean |SHAP| leaky:   {comparison['wbv_mean_abs_shap_leaky']}")
    tqdm.write(f"  TG4h rank (leaky model): {comparison['ranks_leaky'].get('tg4h', 'N/A')}")

    # --- Save CSV ---
    pbar.set_description("SHAP: saving CSV")
    rows = []
    for feat in set(list(comparison["ranks_clean"].keys()) + list(comparison["ranks_leaky"].keys())):
        rows.append({
            "feature": feat,
            "rank_clean": comparison["ranks_clean"].get(feat),
            "rank_leaky": comparison["ranks_leaky"].get(feat),
            "distortion": comparison["attribution_distortion"].get(feat),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("rank_clean").to_csv(out_csv, index=False)
    tqdm.write(f"  Saved SHAP CSV -> {out_csv}")
    pbar.update(1)

    # --- Figures ---
    fig_dir.mkdir(parents=True, exist_ok=True)

    pbar.set_description("SHAP: beeswarm plot (clean)")
    plot_shap_beeswarm(
        shap_clean, X_clean_s[:len(shap_clean)], CLEAN_FEATURES,
        "SHAP Beeswarm — Clean Model (no leakage)",
        fig_dir / "shap_beeswarm_clean.png",
    )
    pbar.update(1)

    pbar.set_description("SHAP: beeswarm plot (leaky)")
    plot_shap_beeswarm(
        shap_leaky, X_leaky_s[:len(shap_leaky)], LEAKY_FEATURES,
        "SHAP Beeswarm — Leaky Model (TG4h included)",
        fig_dir / "shap_beeswarm_leaky.png",
    )
    pbar.update(1)

    pbar.set_description("SHAP: bar comparison plot")
    plot_mean_abs_shap_comparison(
        shap_clean, shap_leaky, CLEAN_FEATURES, LEAKY_FEATURES,
        fig_dir / "shap_comparison_bars.png",
    )
    pbar.update(1)
    pbar.close()

    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SHAP analysis: clean vs leaky model comparison.")
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="results/tables/shap_comparison.csv")
    p.add_argument("--figdir", default="results/figures")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data, keep_default_na=False, na_values=["NA", "NaN", "nan", ""])
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_shap_analysis(df, cfg, Path(args.out), Path(args.figdir), args.seed)


if __name__ == "__main__":
    main()
