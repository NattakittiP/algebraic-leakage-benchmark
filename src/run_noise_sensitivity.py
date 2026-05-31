"""run_noise_sensitivity.py — Noise sensitivity analysis: vary TCR SD, measure clean/leaky AUROC."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import CLEAN_FEATURES, fold_sealed_preprocess
from src.calibration import calibration_summary
from src.utils import FoldSealedScaler, FoldSealedWinsorizer

warnings.filterwarnings("ignore")

NOISE_LEVELS = {
    "low":       0.5,
    "baseline":  1.0,
    "medium":    1.5,
    "high":      2.0,
    "very_high": 3.0,
}

LEAKY_FEATURES = CLEAN_FEATURES + ["tg4h"]


def _run_single_fold_clean(
    X_train: np.ndarray,
    X_test: np.ndarray,
    tcr_train: np.ndarray,
    tcr_test: np.ndarray,
    cfg: dict,
    seed: int,
) -> dict:
    """One outer fold of the clean pipeline."""
    X_tr_p, X_te_p, y_tr, threshold = fold_sealed_preprocess(
        X_train, X_test, None, tcr_train, cfg
    )
    y_te = (tcr_test <= threshold).astype(int)

    if len(np.unique(y_te)) < 2:
        return {}

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    model.fit(X_tr_p, y_tr)
    y_prob = model.predict_proba(X_te_p)[:, 1]

    return {
        "auc":   float(roc_auc_score(y_te, y_prob)),
        "brier": float(brier_score_loss(y_te, y_prob)),
        "prev":  float(y_te.mean()),
    }


def _run_single_fold_leaky(
    X_train_leaky: np.ndarray,
    X_test_leaky: np.ndarray,
    tcr_train: np.ndarray,
    tcr_test: np.ndarray,
    cfg: dict,
    seed: int,
) -> dict:
    """One outer fold of the leaky pipeline (TG4h included, global scaling)."""
    label_q = cfg.get("label_threshold_percentile", 25.0)
    threshold = float(np.percentile(tcr_train, label_q))
    y_tr = (tcr_train <= threshold).astype(int)
    y_te = (tcr_test <= threshold).astype(int)

    if len(np.unique(y_te)) < 2:
        return {}

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_leaky)
    X_te_s = scaler.transform(X_test_leaky)

    if y_tr.sum() >= 2 and (1 - y_tr).sum() >= 2:
        sm = SMOTE(k_neighbors=5, random_state=seed)
        X_tr_s, y_tr = sm.fit_resample(X_tr_s, y_tr)

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    model.fit(X_tr_s, y_tr)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    return {
        "auc":   float(roc_auc_score(y_te, y_prob)),
        "brier": float(brier_score_loss(y_te, y_prob)),
        "prev":  float(y_te.mean()),
    }


def run_one_noise_seed(
    base_cfg: dict,
    model_cfg: dict,
    noise_level: str,
    noise_multiplier: float,
    seed: int,
    n: int,
    outer_folds: int = 5,
    fold_pbar: tqdm | None = None,
) -> dict:
    """Generate data at a given noise level, run clean + leaky 5-fold CV."""
    cfg = deepcopy(base_cfg)
    base_tcr_sd = base_cfg.get("tcr_sd", 18.6)
    cfg["tcr_sd"] = base_tcr_sd * noise_multiplier

    try:
        df = generate(cfg, seed=seed, n=n)
    except Exception as e:
        return {"noise_level": noise_level, "seed": seed, "error": str(e)}

    X_clean = df[CLEAN_FEATURES].values
    X_leaky = df[LEAKY_FEATURES].values
    tcr = df["tcr"].values

    label_q = model_cfg.get("label_threshold_percentile", 25.0)
    y_global = (tcr <= np.percentile(tcr, label_q)).astype(int)

    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    clean_aucs, leaky_aucs = [], []
    clean_briers, leaky_briers = [], []

    for fold_i, (train_idx, test_idx) in enumerate(
        tqdm(
            skf.split(X_clean, y_global),
            total=outer_folds,
            desc=f"    Folds (seed={seed})",
            leave=False,
            ncols=88,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    ):
        fold_clean = _run_single_fold_clean(
            X_clean[train_idx], X_clean[test_idx],
            tcr[train_idx], tcr[test_idx],
            model_cfg, seed,
        )
        if fold_clean:
            clean_aucs.append(fold_clean["auc"])
            clean_briers.append(fold_clean["brier"])

        fold_leaky = _run_single_fold_leaky(
            X_leaky[train_idx], X_leaky[test_idx],
            tcr[train_idx], tcr[test_idx],
            model_cfg, seed,
        )
        if fold_leaky:
            leaky_aucs.append(fold_leaky["auc"])
            leaky_briers.append(fold_leaky["brier"])

        if fold_pbar is not None:
            fold_pbar.update(1)
            if clean_aucs and leaky_aucs:
                fold_pbar.set_postfix(
                    clean=f"{clean_aucs[-1]:.3f}",
                    leaky=f"{leaky_aucs[-1]:.3f}",
                )

    if not clean_aucs or not leaky_aucs:
        return {"noise_level": noise_level, "seed": seed, "error": "no valid folds"}

    clean_auc = float(np.mean(clean_aucs))
    leaky_auc = float(np.mean(leaky_aucs))

    return {
        "noise_level":   noise_level,
        "noise_mult":    noise_multiplier,
        "tcr_sd":        round(base_tcr_sd * noise_multiplier, 2),
        "seed":          seed,
        "clean_AUROC":   round(clean_auc, 4),
        "leaky_AUROC":   round(leaky_auc, 4),
        "auc_inflation": round(leaky_auc - clean_auc, 4),
        "clean_Brier":   round(float(np.mean(clean_briers)), 4),
        "leaky_Brier":   round(float(np.mean(leaky_briers)), 4),
    }


def aggregate_noise_results(records: list[dict]) -> pd.DataFrame:
    """Aggregate per-seed results into summary per noise level."""
    df = pd.DataFrame([r for r in records if "error" not in r])
    if df.empty:
        return pd.DataFrame()

    rows = []
    for noise_level, grp in df.groupby("noise_level"):
        tcr_sd = grp["tcr_sd"].iloc[0]
        noise_mult = grp["noise_mult"].iloc[0]

        def _stats(col):
            v = grp[col].values
            return {
                f"{col}_mean":  round(float(v.mean()), 4),
                f"{col}_sd":    round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
                f"{col}_p2_5":  round(float(np.percentile(v, 2.5)), 4),
                f"{col}_p97_5": round(float(np.percentile(v, 97.5)), 4),
            }

        row = {
            "noise_level":  noise_level,
            "noise_mult":   noise_mult,
            "tcr_sd":       tcr_sd,
            "n_seeds":      len(grp),
        }
        row.update(_stats("clean_AUROC"))
        row.update(_stats("leaky_AUROC"))
        row.update(_stats("auc_inflation"))
        row.update(_stats("clean_Brier"))
        row.update(_stats("leaky_Brier"))
        rows.append(row)

    result = pd.DataFrame(rows)
    sort_order = {k: v for k, v in NOISE_LEVELS.items()}
    result["_sort"] = result["noise_level"].map(sort_order)
    result = result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return result


def plot_noise_sensitivity(summary: pd.DataFrame, raw: pd.DataFrame, out_path: Path):
    """Three-panel noise sensitivity figure."""
    levels = summary["noise_level"].tolist()
    x = np.arange(len(levels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    clean_mean  = summary["clean_AUROC_mean"].values
    leaky_mean  = summary["leaky_AUROC_mean"].values
    clean_sd    = summary["clean_AUROC_sd"].values
    leaky_sd    = summary["leaky_AUROC_sd"].values

    ax.bar(x - width/2, clean_mean, width, label="Clean",
           color="steelblue", alpha=0.85,
           yerr=clean_sd, capsize=4, error_kw={"elinewidth": 1.2})
    ax.bar(x + width/2, leaky_mean, width, label="Leaky (TG4h)",
           color="firebrick", alpha=0.85,
           yerr=leaky_sd, capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean AUROC (±SD across seeds)")
    ax.set_title("Clean vs Leaky AUROC\nby Noise Level")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1]
    inflation_mean = summary["auc_inflation_mean"].values
    inflation_sd   = summary["auc_inflation_sd"].values

    colors = ["#d73027" if v > 0.3 else "#fc8d59" if v > 0.15 else "#fee090"
              for v in inflation_mean]
    ax.bar(x, inflation_mean, color=colors, alpha=0.9,
           yerr=inflation_sd, capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC Inflation (leaky − clean, ±SD)")
    ax.set_title("AUC Inflation by Noise Level\n(Leaky − Clean AUROC)")
    ax.set_ylim(bottom=0)
    for xi, (mean_val, sd_val) in enumerate(zip(inflation_mean, inflation_sd)):
        ax.text(xi, mean_val + sd_val + 0.01, f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax = axes[2]
    clean_brier = summary["clean_Brier_mean"].values
    leaky_brier = summary["leaky_Brier_mean"].values
    ax.plot(x, clean_brier, "o-", color="steelblue", label="Clean", linewidth=2, markersize=6)
    ax.plot(x, leaky_brier, "s-", color="firebrick", label="Leaky (TG4h)", linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Brier Score (lower = better)")
    ax.set_title("Brier Score by Noise Level\n(Calibration quality)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    plt.suptitle(
        "Noise Sensitivity Analysis: Effect of TCR Residual Noise\n"
        "on Clean vs Leaky Pipeline Performance",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_noise_scatter(raw: pd.DataFrame, out_path: Path):
    """Scatter plot: per-seed AUC inflation vs TCR SD."""
    fig, ax = plt.subplots(figsize=(9, 5))

    df_valid = raw[raw["auc_inflation"].notna()].copy()
    noise_order = {k: v for k, v in NOISE_LEVELS.items()}
    df_valid["_sort"] = df_valid["noise_level"].map(noise_order)
    df_valid = df_valid.sort_values("_sort")

    cmap = plt.cm.RdYlBu_r
    for i, (noise_level, grp) in enumerate(df_valid.groupby("noise_level", sort=False)):
        color = cmap(i / max(len(NOISE_LEVELS) - 1, 1))
        ax.scatter(grp["tcr_sd"], grp["auc_inflation"],
                   alpha=0.55, s=40, color=color, label=noise_level)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("TCR Residual SD (noise level)")
    ax.set_ylabel("AUC Inflation per seed (leaky − clean AUROC)")
    ax.set_title("AUC Inflation vs Noise Level (per-seed scatter)")
    ax.legend(title="Noise level", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Noise sensitivity: vary TCR SD, measure clean/leaky AUROC."
    )
    p.add_argument("--config",       default="config/generator_null.yaml")
    p.add_argument("--model_config", default="config/model_config.yaml")
    p.add_argument("--out",          default="results/tables/noise_sensitivity.csv")
    p.add_argument("--figdir",       default="results/figures")
    p.add_argument("--n",            type=int, default=1500)
    p.add_argument("--seeds",        type=int, default=20)
    p.add_argument("--outer_folds",  type=int, default=5)
    p.add_argument("--quick",        action="store_true",
                   help="Quick mode: 3 seeds only")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    n_seeds = 3 if args.quick else args.seeds
    base_tcr_sd = base_cfg.get("tcr_sd", 18.6)

    n_levels = len(NOISE_LEVELS)
    total_runs = n_levels * n_seeds
    total_folds = total_runs * args.outer_folds

    tqdm.write("=" * 70)
    tqdm.write("  Noise Sensitivity Analysis (Plan item 37)")
    tqdm.write(f"  {n_levels} noise levels × {n_seeds} seeds × n={args.n}")
    tqdm.write(f"  Base TCR σ = {base_tcr_sd:.1f}  →  "
               f"testing σ ∈ {[round(base_tcr_sd*m,1) for m in NOISE_LEVELS.values()]}")
    tqdm.write(f"  Total runs: {total_runs}  |  Total CV folds: {total_folds}")
    tqdm.write("=" * 70)

    records: list[dict] = []

    overall_pbar = tqdm(
        total=total_runs,
        desc="Overall",
        ncols=90,
        colour="blue",
        position=0,
        bar_format="  [{elapsed}<{remaining}, {rate_fmt}]  {l_bar}{bar}| {n_fmt}/{total_fmt}",
    )

    noise_pbar = tqdm(
        NOISE_LEVELS.items(),
        desc="Noise level",
        ncols=90,
        colour="magenta",
        position=1,
        leave=True,
    )

    for noise_level, noise_mult in noise_pbar:
        tcr_sd = round(base_tcr_sd * noise_mult, 1)
        noise_pbar.set_description(
            f"Noise [{noise_level:9s}] σ={tcr_sd:5.1f}"
        )

        level_aucs_clean, level_aucs_leaky, level_inflations = [], [], []

        seed_pbar = tqdm(
            range(1, n_seeds + 1),
            desc=f"  Seeds",
            ncols=90,
            colour="cyan",
            position=2,
            leave=False,
            bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} seeds  [{elapsed}<{remaining}]  {postfix}",
        )

        for seed in seed_pbar:
            seed_pbar.set_description(f"  Seed {seed:3d}/{n_seeds}")

            result = run_one_noise_seed(
                base_cfg, model_cfg,
                noise_level, noise_mult,
                seed=seed, n=args.n,
                outer_folds=args.outer_folds,
            )
            records.append(result)

            if "error" not in result:
                level_aucs_clean.append(result["clean_AUROC"])
                level_aucs_leaky.append(result["leaky_AUROC"])
                level_inflations.append(result["auc_inflation"])

                seed_pbar.set_postfix(
                    clean=f"{result['clean_AUROC']:.3f}",
                    leaky=f"{result['leaky_AUROC']:.3f}",
                    infl=f"+{result['auc_inflation']:.3f}",
                    refresh=True,
                )
            else:
                seed_pbar.set_postfix(status="ERROR", refresh=True)

            overall_pbar.update(1)
            overall_pbar.set_postfix(
                level=noise_level,
                done=len(records),
                ok=len(level_aucs_clean),
            )

        seed_pbar.close()

        if level_aucs_clean:
            c_mu = np.mean(level_aucs_clean)
            l_mu = np.mean(level_aucs_leaky)
            i_mu = np.mean(level_inflations)
            i_sd = np.std(level_inflations, ddof=1) if len(level_inflations) > 1 else 0.0
            stable = "✓ stable" if l_mu > 0.70 else "⚠ low"
            tqdm.write(
                f"  ✓ {noise_level:9s} σ={tcr_sd:5.1f} | "
                f"clean={c_mu:.4f}  leaky={l_mu:.4f}  "
                f"inflation=+{i_mu:.4f}±{i_sd:.4f}  leaky={stable}"
            )
        else:
            tqdm.write(f"  ✗ {noise_level}: all seeds errored")

    noise_pbar.close()
    overall_pbar.close()

    elapsed = time.time() - t0
    tqdm.write(
        f"\n  Finished {total_runs} runs in {elapsed:.1f}s "
        f"({elapsed / max(total_runs, 1):.1f}s/run avg)\n"
    )

    steps = [
        "Aggregate results",
        "Save raw CSV",
        "Save summary CSV",
        "Save JSON",
        "Print summary table",
        "Plot: noise_sensitivity.png",
        "Plot: noise_scatter.png",
        "Done",
    ]
    steps_pbar = tqdm(
        steps,
        desc="Post-processing",
        ncols=90,
        colour="yellow",
        bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}  {desc}",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    steps_pbar.set_description("Aggregate results")
    summary = aggregate_noise_results(records)
    steps_pbar.update(1)

    steps_pbar.set_description("Save raw CSV")
    raw_df = pd.DataFrame(records)
    raw_path = out_path.with_name(out_path.stem + "_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    tqdm.write(f"  Saved raw records ({len(records)} rows) -> {raw_path}")
    steps_pbar.update(1)

    steps_pbar.set_description("Save summary CSV")
    summary.to_csv(out_path, index=False)
    tqdm.write(f"  Saved summary ({len(summary)} rows) -> {out_path}")
    steps_pbar.update(1)

    steps_pbar.set_description("Save JSON")
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as jf:
        json.dump(
            {
                "summary": summary.to_dict(orient="records"),
                "noise_levels": NOISE_LEVELS,
                "base_tcr_sd": base_tcr_sd,
                "n_seeds": n_seeds,
            },
            jf, indent=2,
        )
    tqdm.write(f"  Saved JSON -> {json_path}")
    steps_pbar.update(1)

    steps_pbar.set_description("Print summary table")
    tqdm.write("\n=== Noise Sensitivity Summary ===")
    if not summary.empty:
        cols = ["noise_level", "tcr_sd",
                "clean_AUROC_mean", "leaky_AUROC_mean",
                "auc_inflation_mean", "auc_inflation_sd"]
        tqdm.write(summary[cols].round(4).to_string(index=False))
        tqdm.write("\n--- Stability Check ---")
        for _, row in summary.iterrows():
            flag = "✓ STABLE" if row["leaky_AUROC_mean"] > 0.70 else "⚠ LOW"
            tqdm.write(
                f"  {row['noise_level']:9s}: inflation={row['auc_inflation_mean']:+.4f} "
                f"± {row['auc_inflation_sd']:.4f}  {flag}"
            )
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: noise_sensitivity.png")
    plot_noise_sensitivity(summary, raw_df, fig_dir / "noise_sensitivity.png")
    tqdm.write(f"  Saved -> {fig_dir / 'noise_sensitivity.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: noise_scatter.png")
    raw_valid = (raw_df[raw_df["auc_inflation"].notna()].copy()
                 if "auc_inflation" in raw_df.columns else raw_df)
    plot_noise_scatter(raw_valid, fig_dir / "noise_scatter.png")
    tqdm.write(f"  Saved -> {fig_dir / 'noise_scatter.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Done ✓")
    steps_pbar.update(1)
    steps_pbar.close()

    total_elapsed = time.time() - t0
    tqdm.write(f"\n{'='*70}")
    tqdm.write(f"  Noise sensitivity COMPLETE  |  total time: {total_elapsed:.1f}s")
    tqdm.write(f"{'='*70}\n")


if __name__ == "__main__":
    main()
