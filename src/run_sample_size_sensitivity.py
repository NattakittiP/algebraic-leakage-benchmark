"""Sample size sensitivity analysis: AUROC convergence, precision, and leakage invariance."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import CLEAN_FEATURES, fold_sealed_preprocess

warnings.filterwarnings("ignore")


SAMPLE_SIZES = [300, 500, 750, 1500, 3000, 6000, 10000, 50000, 100000]

SCENARIO_CONFIGS = {
    "null":            "config/generator_null.yaml",
    "weak_signal":     "config/generator_weak_signal.yaml",
    "moderate_signal": "config/generator_moderate_signal.yaml",
    "wbv_positive":    "config/generator_wbv_positive.yaml",
}

LEAKY_FEATURES = CLEAN_FEATURES + ["tg4h"]

NULL_RANGE = (0.45, 0.55)      # AUROC range considered "null"
TOST_MARGIN = 0.05             # equivalence margin for TOST-like check


def _eval_fold_clean(
    X_train: np.ndarray,
    X_test: np.ndarray,
    tcr_train: np.ndarray,
    tcr_test: np.ndarray,
    cfg_model: dict,
    seed: int,
) -> dict:
    """One outer fold — fold-sealed clean pipeline (LR)."""
    X_tr_p, X_te_p, y_tr, threshold = fold_sealed_preprocess(
        X_train, X_test, None, tcr_train, cfg_model
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


def _eval_fold_leaky(
    X_train_leaky: np.ndarray,
    X_test_leaky: np.ndarray,
    tcr_train: np.ndarray,
    tcr_test: np.ndarray,
    cfg_model: dict,
    seed: int,
) -> dict:
    """One outer fold — leaky pipeline (TG4h included, global scaling)."""
    label_q = cfg_model.get("label_threshold_percentile", 25.0)
    threshold = float(np.percentile(tcr_train, label_q))
    y_tr = (tcr_train <= threshold).astype(int)
    y_te = (tcr_test <= threshold).astype(int)

    if len(np.unique(y_te)) < 2:
        return {}

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_leaky)
    X_te_s = scaler.transform(X_test_leaky)

    if y_tr.sum() >= 2 and (1 - y_tr).sum() >= 2:
        sm = SMOTE(k_neighbors=min(5, y_tr.sum() - 1), random_state=seed)
        try:
            X_tr_s, y_tr = sm.fit_resample(X_tr_s, y_tr)
        except Exception:
            pass

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    model.fit(X_tr_s, y_tr)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    return {
        "auc":   float(roc_auc_score(y_te, y_prob)),
        "brier": float(brier_score_loss(y_te, y_prob)),
    }


def run_one_seed(
    scenario_name: str,
    cfg_gen: dict,
    cfg_model: dict,
    n: int,
    seed: int,
    outer_folds: int = 5,
    run_leaky: bool = False,
) -> dict:
    """Generate data for one (scenario, n, seed) and run clean (+ optionally leaky) pipeline.

    Parameters
    ----------
    run_leaky : bool
        If True, also run leaky (TG4h) pipeline alongside clean.
        Only recommended for null scenario to avoid confusion.
    """
    try:
        df = generate(cfg_gen, seed=seed, n=n)
    except Exception as e:
        return {
            "scenario": scenario_name, "n": n, "seed": seed,
            "error": str(e),
        }

    X_clean = df[CLEAN_FEATURES].values
    tcr     = df["tcr"].values

    label_q  = cfg_model.get("label_threshold_percentile", 25.0)
    y_global = (tcr <= np.percentile(tcr, label_q)).astype(int)

    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    clean_aucs, clean_briers = [], []
    leaky_aucs, leaky_briers = [], []

    if run_leaky:
        X_leaky = df[LEAKY_FEATURES].values

    for train_idx, test_idx in skf.split(X_clean, y_global):
        fold_c = _eval_fold_clean(
            X_clean[train_idx], X_clean[test_idx],
            tcr[train_idx], tcr[test_idx],
            cfg_model, seed,
        )
        if fold_c:
            clean_aucs.append(fold_c["auc"])
            clean_briers.append(fold_c["brier"])

        if run_leaky:
            fold_l = _eval_fold_leaky(
                X_leaky[train_idx], X_leaky[test_idx],
                tcr[train_idx], tcr[test_idx],
                cfg_model, seed,
            )
            if fold_l:
                leaky_aucs.append(fold_l["auc"])
                leaky_briers.append(fold_l["brier"])

    if not clean_aucs:
        return {
            "scenario": scenario_name, "n": n, "seed": seed,
            "error": "no valid clean folds",
        }

    clean_auc   = float(np.mean(clean_aucs))
    clean_brier = float(np.mean(clean_briers))
    in_null     = int(NULL_RANGE[0] <= clean_auc <= NULL_RANGE[1])

    record: dict = {
        "scenario":    scenario_name,
        "n":           n,
        "seed":        seed,
        "clean_AUROC": round(clean_auc, 5),
        "clean_Brier": round(clean_brier, 5),
        "in_null_range": in_null,
    }

    if run_leaky and leaky_aucs:
        leaky_auc   = float(np.mean(leaky_aucs))
        leaky_brier = float(np.mean(leaky_briers))
        record["leaky_AUROC"]     = round(leaky_auc, 5)
        record["leaky_Brier"]     = round(leaky_brier, 5)
        record["auc_inflation"]   = round(leaky_auc - clean_auc, 5)

    return record


def aggregate_results(records: list[dict]) -> pd.DataFrame:
    """Aggregate per-seed records into (scenario × n) summary rows."""
    df = pd.DataFrame(records)
    if "error" in df.columns:
        df_ok = df[df["error"].isna()].copy()
    else:
        df_ok = df.copy()
    df_ok = df_ok[df_ok["clean_AUROC"].notna()].copy()

    rows = []
    for (scenario, n_val), grp in df_ok.groupby(["scenario", "n"]):
        aucs  = grp["clean_AUROC"].values

        def _stats(col_vals):
            return {
                "mean":  round(float(col_vals.mean()), 4),
                "sd":    round(float(col_vals.std(ddof=1)), 4) if len(col_vals) > 1 else 0.0,
                "p2_5":  round(float(np.percentile(col_vals, 2.5)), 4),
                "p97_5": round(float(np.percentile(col_vals, 97.5)), 4),
            }

        row: dict = {
            "scenario":        scenario,
            "n":               int(n_val),
            "n_seeds":         len(aucs),
        }
        s = _stats(aucs)
        row["clean_AUROC_mean"]  = s["mean"]
        row["clean_AUROC_sd"]    = s["sd"]
        row["clean_AUROC_p2_5"]  = s["p2_5"]
        row["clean_AUROC_p97_5"] = s["p97_5"]
        row["ci_width_95"]       = round(s["p97_5"] - s["p2_5"], 4)
        row["pct_null_range"]    = round(float(grp["in_null_range"].mean() * 100), 1)

        row["tost_equivalent"] = int(
            s["p2_5"] >= NULL_RANGE[0] - TOST_MARGIN
            and s["p97_5"] <= NULL_RANGE[1] + TOST_MARGIN
        )

        if "clean_Brier" in grp.columns:
            b = _stats(grp["clean_Brier"].values)
            row["clean_Brier_mean"] = b["mean"]
            row["clean_Brier_sd"]   = b["sd"]

        if "leaky_AUROC" in grp.columns and grp["leaky_AUROC"].notna().any():
            laucs = grp["leaky_AUROC"].dropna().values
            ls    = _stats(laucs)
            row["leaky_AUROC_mean"]  = ls["mean"]
            row["leaky_AUROC_sd"]    = ls["sd"]
            row["leaky_AUROC_p2_5"]  = ls["p2_5"]
            row["leaky_AUROC_p97_5"] = ls["p97_5"]
            infl = grp["auc_inflation"].dropna().values
            row["auc_inflation_mean"] = round(float(infl.mean()), 4)
            row["auc_inflation_sd"]   = round(float(infl.std(ddof=1)), 4) if len(infl) > 1 else 0.0

        rows.append(row)

    result = pd.DataFrame(rows)
    sc_order = {"null": 0, "weak_signal": 1, "moderate_signal": 2, "wbv_positive": 3}
    result["_sc_sort"] = result["scenario"].map(sc_order).fillna(99)
    result = result.sort_values(["_sc_sort", "n"]).drop(columns="_sc_sort").reset_index(drop=True)
    return result


def plot_auroc_convergence(summary: pd.DataFrame, out_path: Path):
    """4-panel figure: AUROC mean (±SD) vs n, one panel per scenario."""
    scenarios = ["null", "weak_signal", "moderate_signal", "wbv_positive"]
    titles    = ["Null (WBV negative-control)",
                 "Weak signal", "Moderate signal", "WBV positive-control"]
    colors    = ["#2166ac", "#4dac26", "#d6604d", "#762a83"]
    ref_auroc = [0.50, 0.58, 0.67, 0.83]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

    for ax, sc, title, col, ref in zip(axes, scenarios, titles, colors, ref_auroc):
        sub = summary[summary["scenario"] == sc].sort_values("n")
        if sub.empty:
            ax.set_title(title + "\n(no data)")
            continue

        xs  = sub["n"].values
        mu  = sub["clean_AUROC_mean"].values
        lo  = sub["clean_AUROC_p2_5"].values
        hi  = sub["clean_AUROC_p97_5"].values

        ax.plot(xs, mu, "o-", color=col, linewidth=2.2, markersize=6, label="Clean AUROC")
        ax.fill_between(xs, lo, hi, alpha=0.15, color=col, label="95% CI (across seeds)")

        ax.axhline(ref, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"Expected ≈ {ref:.2f}")

        if sc == "null":
            ax.axhline(0.50, color="black", linestyle=":", linewidth=0.9)
            ax.fill_between(xs, 0.45, 0.55, alpha=0.06, color="gray", label="Null range [0.45,0.55]")

            if "leaky_AUROC_mean" in sub.columns and sub["leaky_AUROC_mean"].notna().any():
                l_mu = sub["leaky_AUROC_mean"].values
                ax.plot(xs, l_mu, "s--", color="firebrick", linewidth=1.8,
                        markersize=5, label="Leaky AUROC (TG4h)")

        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Sample size (n)", fontsize=9)
        ax.set_ylabel("AUROC", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.set_ylim(0.35, 1.05)

    plt.suptitle(
        "Sample Size Sensitivity: AUROC Convergence by Scenario\n"
        "(log-scale x-axis; shaded = 95% CI across seeds)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_vs_n(summary: pd.DataFrame, out_path: Path):
    """AUROC SD and 95-CI width vs n — precision improves with n."""
    scenarios = ["null", "weak_signal", "moderate_signal", "wbv_positive"]
    colors    = ["#2166ac", "#4dac26", "#d6604d", "#762a83"]
    labels    = ["Null", "Weak signal", "Moderate", "WBV-positive"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for sc, col, lab in zip(scenarios, colors, labels):
        sub = summary[summary["scenario"] == sc].sort_values("n")
        if sub.empty:
            continue
        xs = sub["n"].values

        axes[0].plot(xs, sub["clean_AUROC_sd"].values, "o-", color=col,
                     linewidth=2, markersize=5, label=lab)

        if "ci_width_95" in sub.columns:
            axes[1].plot(xs, sub["ci_width_95"].values, "s--", color=col,
                         linewidth=2, markersize=5, label=lab)

    for ax, ylabel, title in zip(
        axes,
        ["AUROC SD (across seeds)", "95% CI width (p97.5 − p2.5)"],
        ["AUROC Standard Deviation vs n\n(lower = more precise)",
         "95% CI Width vs n\n(narrower = more conclusive)"],
    ):
        ax.set_xscale("log")
        ns_for_ticks = sorted(summary["n"].unique())
        ax.set_xticks(ns_for_ticks)
        ax.set_xticklabels([str(x) for x in ns_for_ticks], rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Sample size (n)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.suptitle("AUROC Precision vs Sample Size\n"
                 "(SD and 95-CI width decrease with n as expected)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_null_quality_vs_n(summary: pd.DataFrame, out_path: Path):
    """Null-scenario only: % in null range + TOST equivalence vs n."""
    sub = summary[summary["scenario"] == "null"].sort_values("n")
    if sub.empty:
        return

    xs = sub["n"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(range(len(xs)), sub["pct_null_range"].values,
           color=["#2166ac" if v >= 90 else "#fc8d59" for v in sub["pct_null_range"].values],
           alpha=0.85, edgecolor="white")
    ax.axhline(90, color="gray", linestyle="--", linewidth=1.2, label="90% threshold")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"n={x}" for x in xs], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% seeds with AUROC ∈ [0.45, 0.55]", fontsize=9)
    ax.set_title("% Seeds in Null Range vs n\n(null scenario, clean pipeline)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    for i, v in enumerate(sub["pct_null_range"].values):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax = axes[1]
    mu = sub["clean_AUROC_mean"].values
    sd = sub["clean_AUROC_sd"].values
    ax.errorbar(range(len(xs)), mu, yerr=sd, fmt="o-", color="#2166ac",
                linewidth=2, markersize=6, capsize=5, label="AUROC mean ± SD")
    ax.fill_between(range(len(xs)), 0.45, 0.55, alpha=0.08, color="gray",
                    label="Null range [0.45, 0.55]")
    ax.axhline(0.50, color="black", linestyle=":", linewidth=0.8)

    if "tost_equivalent" in sub.columns:
        for i, (tost, y) in enumerate(zip(sub["tost_equivalent"].values, mu)):
            marker = "✓" if tost else "✗"
            color  = "green" if tost else "red"
            ax.text(i, y - sd[i] - 0.012, marker, ha="center", fontsize=12, color=color)

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"n={x}" for x in xs], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Clean AUROC", fontsize=9)
    ax.set_title("Null AUROC Convergence ± SD\n(✓/✗ = TOST-like equivalence)",
                 fontsize=10)
    ax.set_ylim(0.35, 0.65)
    ax.legend(fontsize=8)

    plt.suptitle("Null Scenario Quality Metrics vs Sample Size\n"
                 "(null scenario should yield AUROC ≈ 0.50 across all n)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_leakage_vs_n(summary: pd.DataFrame, out_path: Path):
    """Leaky AUROC and AUC inflation vs n — leakage should be n-invariant."""
    sub = summary[
        (summary["scenario"] == "null") &
        summary.get("leaky_AUROC_mean", pd.Series(dtype=float)).notna()
    ].sort_values("n") if "leaky_AUROC_mean" in summary.columns else pd.DataFrame()

    if sub.empty or sub["leaky_AUROC_mean"].isna().all():
        return

    xs = sub["n"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(xs, sub["clean_AUROC_mean"].values, "o-", color="#2166ac",
            linewidth=2, markersize=6, label="Clean AUROC")
    ax.fill_between(xs,
                    sub["clean_AUROC_p2_5"].values,
                    sub["clean_AUROC_p97_5"].values,
                    alpha=0.12, color="#2166ac")
    ax.plot(xs, sub["leaky_AUROC_mean"].values, "s--", color="firebrick",
            linewidth=2, markersize=6, label="Leaky AUROC (TG4h)")
    if "leaky_AUROC_p2_5" in sub.columns:
        ax.fill_between(xs,
                        sub["leaky_AUROC_p2_5"].values,
                        sub["leaky_AUROC_p97_5"].values,
                        alpha=0.10, color="firebrick")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=0.8)
    ax.fill_between(xs, 0.45, 0.55, alpha=0.06, color="gray", label="Null range")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Sample size (n)", fontsize=9)
    ax.set_ylabel("AUROC", fontsize=9)
    ax.set_title("Clean vs Leaky AUROC vs n\n(null scenario)", fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1]
    infl     = sub["auc_inflation_mean"].values
    infl_sd  = sub["auc_inflation_sd"].values if "auc_inflation_sd" in sub.columns else np.zeros_like(infl)
    colors_b = ["#d73027" if v > 0.30 else "#fc8d59" for v in infl]
    ax.bar(range(len(xs)), infl, color=colors_b, alpha=0.85,
           yerr=infl_sd, capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.6)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"n={x}" for x in xs], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("AUC Inflation (leaky − clean)", fontsize=9)
    ax.set_title("AUC Inflation vs n\n(should stay high — leakage is n-invariant)",
                 fontsize=10)
    ax.set_ylim(bottom=0)
    for i, (v, sd) in enumerate(zip(infl, infl_sd)):
        ax.text(i, v + sd + 0.005, f"+{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Definitional Leakage (TG4h) vs Sample Size\n"
                 "(leakage inflation should be sample-size-invariant)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample size sensitivity: convergence, precision, leakage invariance."
    )
    p.add_argument("--configdir",    default="config/")
    p.add_argument("--model_config", default="config/model_config.yaml")
    p.add_argument("--out",          default="results/tables/sample_size_sensitivity.csv")
    p.add_argument("--figdir",       default="results/figures")
    p.add_argument("--seeds",        type=int, default=30,
                   help="Seeds per (scenario × n). Default 30.")
    p.add_argument("--outer_folds",  type=int, default=5)
    p.add_argument("--n_jobs",       type=int, default=-1,
                   help="Parallel jobs for joblib. -1 = all cores.")
    p.add_argument("--quick",        action="store_true",
                   help="Quick smoke test: 5 seeds, 4 sample sizes (300,750,1500,3000).")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    with open(args.model_config) as f:
        cfg_model = yaml.safe_load(f)

    config_dir = Path(args.configdir)
    scenarios: dict[str, dict] = {}
    for sc_name, cfg_path in SCENARIO_CONFIGS.items():
        p = Path(cfg_path)
        if not p.exists():
            p = config_dir / p.name
        if p.exists():
            with open(p) as f:
                scenarios[sc_name] = yaml.safe_load(f)
        else:
            tqdm.write(f"  WARNING: config not found for {sc_name}: {p}")

    n_seeds   = 5 if args.quick else args.seeds
    sz_list   = [300, 750, 1500, 3000] if args.quick else SAMPLE_SIZES

    tasks_clean: list[tuple] = [
        (sc_name, cfg, n_sz, seed)
        for sc_name, cfg in scenarios.items()
        for n_sz in sz_list
        for seed in range(1, n_seeds + 1)
    ]
    tasks_leaky: list[tuple] = [
        ("null", scenarios["null"], n_sz, seed)
        for n_sz in sz_list
        for seed in range(1, n_seeds + 1)
        if "null" in scenarios
    ]

    n_clean = len(tasks_clean)
    n_leaky = len(tasks_leaky)
    total   = n_clean + n_leaky

    tqdm.write("=" * 70)
    tqdm.write("  Sample Size Sensitivity Analysis (Plan item 36)")
    tqdm.write(f"  Sample sizes: {sz_list}")
    tqdm.write(f"  Scenarios:    {list(scenarios.keys())}")
    tqdm.write(f"  Seeds/cell:   {n_seeds}")
    tqdm.write(f"  Clean runs:   {n_clean}  |  Leaky runs (null): {n_leaky}")
    tqdm.write(f"  Total runs:   {total}  |  CV folds/run: {args.outer_folds}")
    tqdm.write(f"  Parallelism:  n_jobs={args.n_jobs}")
    tqdm.write("=" * 70)

    records: list[dict] = []

    tqdm.write(f"\n{'─'*70}")
    tqdm.write(f"  PHASE 1/2 — Clean pipeline  ({n_clean} runs)")
    tqdm.write(f"{'─'*70}")

    overall_pbar = tqdm(
        total=total,
        desc="Overall",
        ncols=90,
        colour="blue",
        position=0,
        bar_format="  [{elapsed}<{remaining}, {rate_fmt}]  {l_bar}{bar}| {n_fmt}/{total_fmt}",
    )

    size_pbar = tqdm(
        sz_list,
        desc="Sample size",
        ncols=90,
        colour="magenta",
        position=1,
        leave=True,
    )

    for n_sz in size_pbar:
        size_pbar.set_description(f"n = {n_sz:>6,d}")
        size_aucs: dict[str, list[float]] = {sc: [] for sc in scenarios}

        sc_pbar = tqdm(
            scenarios.items(),
            desc="  Scenario",
            ncols=90,
            colour="cyan",
            position=2,
            leave=False,
        )

        for sc_name, cfg_gen in sc_pbar:
            sc_pbar.set_description(f"  {sc_name}")

            seed_pbar = tqdm(
                range(1, n_seeds + 1),
                desc=f"    Seed",
                ncols=88,
                colour="green",
                position=3,
                leave=False,
                bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} seeds  [{elapsed}<{remaining}]  {postfix}",
            )

            for seed in seed_pbar:
                seed_pbar.set_description(f"    Seed {seed:3d}/{n_seeds}")

                result = run_one_seed(
                    sc_name, cfg_gen, cfg_model,
                    n=n_sz, seed=seed,
                    outer_folds=args.outer_folds,
                    run_leaky=False,
                )
                records.append(result)

                if "error" not in result:
                    size_aucs[sc_name].append(result["clean_AUROC"])
                    seed_pbar.set_postfix(
                        AUROC=f"{result['clean_AUROC']:.3f}",
                        in_null="✓" if result.get("in_null_range") else "✗",
                        refresh=True,
                    )
                else:
                    seed_pbar.set_postfix(status="ERR", refresh=True)

                overall_pbar.update(1)
                overall_pbar.set_postfix(
                    n=n_sz,
                    sc=sc_name[:6],
                    done=len(records),
                )

            seed_pbar.close()

            aucs = size_aucs[sc_name]
            if aucs:
                mu = np.mean(aucs)
                sd = np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0
                pct = np.mean([(NULL_RANGE[0] <= a <= NULL_RANGE[1]) for a in aucs]) * 100
                tqdm.write(
                    f"  ✓ n={n_sz:>6,d} | {sc_name:16s} | "
                    f"AUROC={mu:.4f} ± {sd:.4f}  pct_null={pct:.0f}%"
                )
            else:
                tqdm.write(f"  ✗ n={n_sz} | {sc_name}: all seeds errored")

        sc_pbar.close()

    size_pbar.close()

    tqdm.write(f"\n{'─'*70}")
    tqdm.write(f"  PHASE 2/2 — Leaky pipeline (null × {len(sz_list)} sizes)  ({n_leaky} runs)")
    tqdm.write(f"{'─'*70}")

    if "null" in scenarios:
        null_cfg = scenarios["null"]

        leaky_size_pbar = tqdm(
            sz_list,
            desc="Leaky / n",
            ncols=90,
            colour="red",
            position=1,
            leave=True,
        )

        for n_sz in leaky_size_pbar:
            leaky_size_pbar.set_description(f"Leaky n = {n_sz:>6,d}")
            leaky_aucs_this: list[float] = []

            leaky_seed_pbar = tqdm(
                range(1, n_seeds + 1),
                desc="  Leaky seeds",
                ncols=88,
                colour="yellow",
                position=2,
                leave=False,
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} seeds  [{elapsed}<{remaining}]  {postfix}",
            )

            for seed in leaky_seed_pbar:
                leaky_seed_pbar.set_description(f"  Leaky seed {seed:3d}/{n_seeds}")

                result = run_one_seed(
                    "null", null_cfg, cfg_model,
                    n=n_sz, seed=seed,
                    outer_folds=args.outer_folds,
                    run_leaky=True,
                )
                if "error" not in result and "leaky_AUROC" in result:
                    leaky_record = {
                        "scenario":      "null_leaky",
                        "n":             n_sz,
                        "seed":          seed,
                        "clean_AUROC":   result["clean_AUROC"],
                        "clean_Brier":   result.get("clean_Brier", np.nan),
                        "leaky_AUROC":   result["leaky_AUROC"],
                        "leaky_Brier":   result.get("leaky_Brier", np.nan),
                        "auc_inflation": result.get("auc_inflation", np.nan),
                        "in_null_range": result["in_null_range"],
                    }
                    records.append(leaky_record)
                    leaky_aucs_this.append(result["leaky_AUROC"])
                    leaky_seed_pbar.set_postfix(
                        clean=f"{result['clean_AUROC']:.3f}",
                        leaky=f"{result['leaky_AUROC']:.3f}",
                        infl=f"+{result.get('auc_inflation', 0):.3f}",
                        refresh=True,
                    )
                else:
                    leaky_seed_pbar.set_postfix(status="ERR", refresh=True)

                overall_pbar.update(1)
                overall_pbar.set_postfix(
                    n=n_sz, phase="leaky", done=len(records),
                )

            leaky_seed_pbar.close()

            if leaky_aucs_this:
                l_mu = np.mean(leaky_aucs_this)
                tqdm.write(
                    f"  ✓ n={n_sz:>6,d} | {'null_leaky':16s} | "
                    f"leaky_AUROC={l_mu:.4f}  (should be ≥ 0.95)"
                )

        leaky_size_pbar.close()

    overall_pbar.close()

    elapsed = time.time() - t0
    tqdm.write(
        f"\n  Finished {total} runs in {elapsed:.1f}s "
        f"({elapsed / max(total, 1):.1f}s/run avg)\n"
    )

    steps = [
        "Aggregate results",
        "Save raw CSV",
        "Save summary CSV",
        "Save JSON",
        "Print summary table",
        "Plot: AUROC convergence",
        "Plot: Precision vs n",
        "Plot: Null quality vs n",
        "Plot: Leakage vs n",
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
    summary = aggregate_results(records)
    steps_pbar.update(1)

    steps_pbar.set_description("Save raw CSV")
    raw_df   = pd.DataFrame(records)
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
                "sample_sizes": sz_list,
                "scenarios":    list(scenarios.keys()),
                "n_seeds":      n_seeds,
                "null_range":   list(NULL_RANGE),
                "tost_margin":  TOST_MARGIN,
                "summary":      summary.to_dict(orient="records"),
            },
            jf, indent=2,
        )
    tqdm.write(f"  Saved JSON -> {json_path}")
    steps_pbar.update(1)

    steps_pbar.set_description("Print summary table")
    tqdm.write("\n=== Sample Size Sensitivity Summary ===")
    if not summary.empty:
        cols = ["scenario", "n", "n_seeds",
                "clean_AUROC_mean", "clean_AUROC_sd", "ci_width_95",
                "pct_null_range", "tost_equivalent"]
        available = [c for c in cols if c in summary.columns]
        tqdm.write(summary[available].to_string(index=False))

        tqdm.write("\n--- Null Convergence Check ---")
        null_sum = summary[summary["scenario"] == "null"].sort_values("n")
        for _, row in null_sum.iterrows():
            tost = "✓ equiv" if row.get("tost_equivalent") else "✗ inconc"
            tqdm.write(
                f"  n={row['n']:>6,d}: AUROC={row['clean_AUROC_mean']:.4f}"
                f" ±{row['clean_AUROC_sd']:.4f}"
                f"  pct_null={row['pct_null_range']:.0f}%"
                f"  {tost}"
            )

        if "null_leaky" in summary["scenario"].values:
            tqdm.write("\n--- Leakage Persistence Check ---")
            leaky_sum = summary[summary["scenario"] == "null_leaky"].sort_values("n")
            for _, row in leaky_sum.iterrows():
                tqdm.write(
                    f"  n={row['n']:>6,d}: leaky_AUROC={row.get('leaky_AUROC_mean', '?'):.4f}"
                    f"  inflation={row.get('auc_inflation_mean', '?'):.4f}"
                )
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: AUROC convergence")
    plot_auroc_convergence(summary, fig_dir / "sample_size_auroc_convergence.png")
    tqdm.write(f"  Saved -> {fig_dir / 'sample_size_auroc_convergence.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: Precision vs n")
    plot_precision_vs_n(summary, fig_dir / "sample_size_precision.png")
    tqdm.write(f"  Saved -> {fig_dir / 'sample_size_precision.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: Null quality vs n")
    plot_null_quality_vs_n(summary, fig_dir / "sample_size_null_quality.png")
    tqdm.write(f"  Saved -> {fig_dir / 'sample_size_null_quality.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: Leakage vs n")
    plot_leakage_vs_n(summary, fig_dir / "sample_size_leakage.png")
    tqdm.write(f"  Saved -> {fig_dir / 'sample_size_leakage.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Done ✓")
    steps_pbar.update(1)
    steps_pbar.close()

    total_elapsed = time.time() - t0
    tqdm.write(f"\n{'='*70}")
    tqdm.write(f"  Sample size sensitivity COMPLETE  |  total time: {total_elapsed:.1f}s")
    tqdm.write(f"{'='*70}\n")


if __name__ == "__main__":
    main()
