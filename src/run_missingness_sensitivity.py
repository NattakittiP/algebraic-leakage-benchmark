"""
run_missingness_sensitivity.py — Missingness sensitivity analysis (Plan item 38).

Tests the effect of missing data on model performance and whether
imputation inside vs outside CV folds introduces leakage.

Missing data mechanisms:
  MCAR_5pct   : Missing Completely At Random, 5% per feature
  MCAR_10pct  : Missing Completely At Random, 10% per feature
  MCAR_20pct  : Missing Completely At Random, 20% per feature
  MAR_bmi     : Missing At Random — missingness depends on BMI
  MAR_tg0h    : Missing At Random — missingness depends on TG0h

Target variables with missingness:
  hct, tp, hdl, ldl  (never induces WBV/TCR leakage)

Imputation strategies compared:
  fold_impute    : median imputed on training fold only (correct — no leakage)
  global_impute  : median imputed on full dataset before CV (leaky)

Key finding to report:
  "Global imputation constitutes a mild preprocessing leakage. Fold-sealed
   imputation corrects for this bias without sacrificing predictive performance."

Usage:
  python src/run_missingness_sensitivity.py \\
      --config config/generator_null.yaml \\
      --model_config config/model_config.yaml \\
      --out results/tables/missingness_sensitivity.csv \\
      --figdir results/figures \\
      --n 1500 \\
      --seeds 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import CLEAN_FEATURES
from src.utils import FoldSealedScaler, FoldSealedWinsorizer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Missingness scenario definitions
# ---------------------------------------------------------------------------

MISSING_FEATURES = ["hct", "tp", "hdl", "ldl"]

MISSING_SCENARIOS: dict[str, dict] = {
    "complete":    {"mechanism": "none",  "rate": 0.00, "depends_on": None},
    "MCAR_5pct":   {"mechanism": "MCAR",  "rate": 0.05, "depends_on": None},
    "MCAR_10pct":  {"mechanism": "MCAR",  "rate": 0.10, "depends_on": None},
    "MCAR_20pct":  {"mechanism": "MCAR",  "rate": 0.20, "depends_on": None},
    "MAR_bmi":     {"mechanism": "MAR",   "rate": 0.10, "depends_on": "bmi"},
    "MAR_tg0h":    {"mechanism": "MAR",   "rate": 0.10, "depends_on": "tg0h"},
}


# ---------------------------------------------------------------------------
# Missing data injection
# ---------------------------------------------------------------------------

def inject_missingness(
    df: pd.DataFrame,
    scenario: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy of df with NaN injected into MISSING_FEATURES."""
    df = df.copy()
    mech = scenario["mechanism"]
    rate = scenario["rate"]

    if mech == "none" or rate == 0:
        return df

    n = len(df)

    for feat in MISSING_FEATURES:
        if feat not in df.columns:
            continue

        if mech == "MCAR":
            miss_idx = rng.choice(n, size=int(n * rate), replace=False)

        elif mech == "MAR":
            dep_col = scenario["depends_on"]
            if dep_col not in df.columns:
                miss_idx = rng.choice(n, size=int(n * rate), replace=False)
            else:
                dep_vals = df[dep_col].values
                dep_norm = (dep_vals - np.min(dep_vals)) / (np.ptp(dep_vals) + 1e-9)
                probs = np.clip(dep_norm * 2 * rate, 0.02, min(0.5, 2 * rate))
                probs /= probs.mean() / rate
                probs = np.clip(probs, 0, 0.9)
                miss_mask = rng.random(n) < probs
                miss_idx = np.where(miss_mask)[0]
        else:
            miss_idx = np.array([], dtype=int)

        df.iloc[miss_idx, df.columns.get_loc(feat)] = np.nan

    return df


# ---------------------------------------------------------------------------
# Imputation strategies
# ---------------------------------------------------------------------------

def fold_impute(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Median imputation fitted on training fold only (correct, no leakage)."""
    medians = np.nanmedian(X_train, axis=0)
    X_train_imp = X_train.copy()
    X_test_imp  = X_test.copy()
    for j in range(X_train.shape[1]):
        X_train_imp[np.isnan(X_train_imp[:, j]), j] = medians[j]
        X_test_imp[np.isnan(X_test_imp[:, j]), j]   = medians[j]
    return X_train_imp, X_test_imp


def global_impute(X_full: np.ndarray) -> np.ndarray:
    """Median imputation fitted on FULL dataset before split (leaky)."""
    medians = np.nanmedian(X_full, axis=0)
    X_imp = X_full.copy()
    for j in range(X_full.shape[1]):
        X_imp[np.isnan(X_imp[:, j]), j] = medians[j]
    return X_imp


# ---------------------------------------------------------------------------
# One fold evaluation
# ---------------------------------------------------------------------------

def _eval_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    tcr_train: np.ndarray,
    tcr_test: np.ndarray,
    label_q: float,
    seed: int,
    strategy: str,
    X_full_global: Optional[np.ndarray] = None,
    global_idx_train: Optional[np.ndarray] = None,
    global_idx_test: Optional[np.ndarray] = None,
) -> dict:
    """Run one fold with a given imputation strategy."""
    if strategy == "fold_impute":
        X_tr_imp, X_te_imp = fold_impute(X_train, X_test)
    elif strategy == "global_impute":
        assert X_full_global is not None
        X_tr_imp = X_full_global[global_idx_train]
        X_te_imp = X_full_global[global_idx_test]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    win = FoldSealedWinsorizer(lower_pct=1.0, upper_pct=99.0)
    X_tr_w = win.fit_transform(X_tr_imp)
    X_te_w = win.transform(X_te_imp)

    scaler = FoldSealedScaler()
    X_tr_s = scaler.fit_transform(X_tr_w)
    X_te_s = scaler.transform(X_te_w)

    threshold = float(np.percentile(tcr_train, label_q))
    y_tr = (tcr_train <= threshold).astype(int)
    y_te = (tcr_test  <= threshold).astype(int)

    if len(np.unique(y_te)) < 2:
        return {}

    if y_tr.sum() >= 2 and (1 - y_tr).sum() >= 2:
        sm = SMOTE(k_neighbors=5, random_state=seed)
        X_tr_s, y_tr = sm.fit_resample(X_tr_s, y_tr)

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    model.fit(X_tr_s, y_tr)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    return {
        "auc":   float(roc_auc_score(y_te, y_prob)),
        "brier": float(brier_score_loss(y_te, y_prob)),
    }


# ---------------------------------------------------------------------------
# One experiment: fixed missing scenario × strategy × one seed
# ---------------------------------------------------------------------------

def run_one_miss_seed(
    base_cfg: dict,
    model_cfg: dict,
    scenario_name: str,
    scenario: dict,
    strategy: str,
    seed: int,
    n: int,
    outer_folds: int = 5,
) -> dict:
    """Generate data, inject missingness, run clean 5-fold CV with given strategy."""
    rng = np.random.default_rng(seed)

    try:
        df = generate(base_cfg, seed=seed, n=n)
    except Exception as e:
        return {"scenario": scenario_name, "strategy": strategy,
                "seed": seed, "error": str(e)}

    df_miss = inject_missingness(df, scenario, rng)
    X = df_miss[CLEAN_FEATURES].values.astype(float)
    tcr = df["tcr"].values

    label_q = model_cfg.get("label_threshold_percentile", 25.0)
    y_global = (tcr <= np.percentile(tcr, label_q)).astype(int)

    X_global_imp = global_impute(X)

    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    aucs, briers = [], []

    for train_idx, test_idx in tqdm(
        skf.split(X, y_global),
        total=outer_folds,
        desc=f"    Folds ({strategy[:10]})",
        leave=False,
        ncols=88,
        colour="green",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]",
    ):
        fold_res = _eval_fold(
            X_train=X[train_idx],
            X_test=X[test_idx],
            tcr_train=tcr[train_idx],
            tcr_test=tcr[test_idx],
            label_q=label_q,
            seed=seed,
            strategy=strategy,
            X_full_global=X_global_imp,
            global_idx_train=train_idx,
            global_idx_test=test_idx,
        )
        if fold_res:
            aucs.append(fold_res["auc"])
            briers.append(fold_res["brier"])

    if not aucs:
        return {"scenario": scenario_name, "strategy": strategy,
                "seed": seed, "error": "no valid folds"}

    miss_rate_actual = float(df_miss[MISSING_FEATURES].isna().mean().mean())

    return {
        "scenario":         scenario_name,
        "mechanism":        scenario["mechanism"],
        "target_rate":      scenario["rate"],
        "actual_miss_rate": round(miss_rate_actual, 4),
        "strategy":         strategy,
        "seed":             seed,
        "AUROC":            round(float(np.mean(aucs)), 4),
        "Brier":            round(float(np.mean(briers)), 4),
        "n_folds":          len(aucs),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_miss_results(records: list[dict]) -> pd.DataFrame:
    """Summary per (scenario × strategy)."""
    df = pd.DataFrame([r for r in records if "error" not in r])
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (scenario, strategy), grp in df.groupby(["scenario", "strategy"]):
        aucs = grp["AUROC"].values
        rows.append({
            "scenario":         scenario,
            "mechanism":        grp["mechanism"].iloc[0],
            "target_rate":      grp["target_rate"].iloc[0],
            "actual_miss_rate": round(float(grp["actual_miss_rate"].mean()), 4),
            "strategy":         strategy,
            "n_seeds":          len(aucs),
            "AUROC_mean":       round(float(aucs.mean()), 4),
            "AUROC_sd":         round(float(aucs.std(ddof=1)), 4) if len(aucs) > 1 else 0.0,
            "AUROC_p2_5":       round(float(np.percentile(aucs, 2.5)), 4),
            "AUROC_p97_5":      round(float(np.percentile(aucs, 97.5)), 4),
            "Brier_mean":       round(float(grp["Brier"].mean()), 4),
        })

    result = pd.DataFrame(rows)
    order = list(MISSING_SCENARIOS.keys())
    result["_sort"] = result["scenario"].map({s: i for i, s in enumerate(order)})
    result = result.sort_values(["_sort", "strategy"]).drop(columns="_sort").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_missingness_results(summary: pd.DataFrame, out_path: Path):
    """Grouped bar chart: fold vs global imputation AUROC + bias panel."""
    scenarios = list(MISSING_SCENARIOS.keys())
    n_sc = len(scenarios)
    x = np.arange(n_sc)
    width = 0.35

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 6),
        constrained_layout=True
    )

    # ------------------------------------------------------------
    # Panel 1: AUROC fold vs global
    # ------------------------------------------------------------
    ax = axes[0]

    for i, (strategy, color, label) in enumerate([
        ("fold_impute",   "steelblue", "Fold imputation (correct)"),
        ("global_impute", "tomato",    "Global imputation (leaky)"),
    ]):
        grp = summary[summary["strategy"] == strategy].set_index("scenario")
        means = [grp.loc[sc, "AUROC_mean"] if sc in grp.index else np.nan for sc in scenarios]
        sds   = [grp.loc[sc, "AUROC_sd"]   if sc in grp.index else 0.0   for sc in scenarios]

        offset = (i - 0.5) * width

        ax.bar(
            x + offset,
            means,
            width,
            label=label,
            color=color,
            alpha=0.85,
            yerr=sds,
            capsize=4,
            error_kw={"elinewidth": 1.2}
        )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Chance (0.50)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean AUROC (±SD)")
    ax.set_title(
        "AUROC: Fold vs Global Imputation\nby Missingness Scenario",
        fontsize=11,
        pad=12
    )
    ax.set_ylim(0.35, 0.65)
    ax.legend(fontsize=8, loc="upper left")

    # ------------------------------------------------------------
    # Panel 2: Leakage bias
    # ------------------------------------------------------------
    ax = axes[1]

    fold_grp = summary[summary["strategy"] == "fold_impute"].set_index("scenario")
    global_grp = summary[summary["strategy"] == "global_impute"].set_index("scenario")

    biases = [
        global_grp.loc[sc, "AUROC_mean"] - fold_grp.loc[sc, "AUROC_mean"]
        if sc in fold_grp.index and sc in global_grp.index else 0.0
        for sc in scenarios
    ]

    bar_colors = [
        "#d73027" if b > 0.005 else "#4575b4" if b < -0.005 else "#ffffbf"
        for b in biases
    ]

    bars = ax.bar(
        x,
        biases,
        color=bar_colors,
        alpha=0.9,
        edgecolor="gray",
        linewidth=0.5
    )

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC Bias (global − fold)")
    ax.set_title(
        "Imputation Leakage Bias\npositive = global inflates AUC",
        fontsize=11,
        pad=12
    )

    # ตั้ง y-limit ให้พอดีกับ bias ไม่ให้ text หลุดไกล
    max_abs_bias = max(abs(float(b)) for b in biases) if biases else 0.001
    y_lim = max(max_abs_bias * 1.8, 0.001)
    ax.set_ylim(-y_lim, y_lim)

    # ใส่ label แบบ offset เป็น points แทนการบวกค่าแกน y ตรง ๆ
    for bar, bias in zip(bars, biases):
        y = bar.get_height()

        if bias >= 0:
            va = "bottom"
            offset_points = 4
        else:
            va = "top"
            offset_points = -4

        ax.annotate(
            f"{bias:+.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset_points),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
            clip_on=True
        )

    fig.suptitle(
        "Missingness Sensitivity: Impact on Clean Pipeline AUROC\n"
        "Fold-sealed vs Global Imputation Strategies",
        fontsize=13,
        fontweight="bold"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ไม่ใช้ bbox_inches='tight' แบบเดิม เพราะทำให้ canvas ขยายตาม text ที่หลุด
    plt.savefig(out_path, dpi=150, pad_inches=0.2)
    plt.close()


def plot_miss_rate_effect(raw_df: pd.DataFrame, out_path: Path):
    """Line plot: AUROC vs actual missing rate, by strategy."""
    fig, ax = plt.subplots(figsize=(9, 5))

    valid = (raw_df[raw_df["error"].isna()]
             if "error" in raw_df.columns else raw_df.copy())
    valid = valid[valid["mechanism"].isin(["none", "MCAR"])]

    for strategy, color, marker, label in [
        ("fold_impute",   "steelblue", "o", "Fold imputation (correct)"),
        ("global_impute", "tomato",    "s", "Global imputation (leaky)"),
    ]:
        strat_df = valid[valid["strategy"] == strategy]
        agg = strat_df.groupby("target_rate")["AUROC"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["target_rate"] * 100, agg["mean"],
                f"{marker}-", color=color, label=label, linewidth=2, markersize=7)
        ax.fill_between(
            agg["target_rate"] * 100,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.15, color=color,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.50)")
    ax.set_xlabel("Missing data rate (%)")
    ax.set_ylabel("Mean AUROC (±SD across seeds)")
    ax.set_title("AUROC vs Missing Data Rate (MCAR only)\nFold vs Global Imputation")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Missingness sensitivity: MCAR/MAR with fold vs global imputation."
    )
    p.add_argument("--config",       default="config/generator_null.yaml")
    p.add_argument("--model_config", default="config/model_config.yaml")
    p.add_argument("--out",          default="results/tables/missingness_sensitivity.csv")
    p.add_argument("--figdir",       default="results/figures")
    p.add_argument("--n",            type=int, default=1500)
    p.add_argument("--seeds",        type=int, default=20)
    p.add_argument("--outer_folds",  type=int, default=5)
    p.add_argument("--quick",        action="store_true",
                   help="Quick mode: 3 seeds, 3 scenarios")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    n_seeds = 3 if args.quick else args.seeds
    scenarios = dict(list(MISSING_SCENARIOS.items())[:3]) if args.quick else MISSING_SCENARIOS
    strategies = ["fold_impute", "global_impute"]

    n_scenarios  = len(scenarios)
    n_strategies = len(strategies)
    total_runs   = n_scenarios * n_strategies * n_seeds

    tqdm.write("=" * 70)
    tqdm.write("  Missingness Sensitivity Analysis (Plan item 38)")
    tqdm.write(f"  {n_scenarios} scenarios × {n_strategies} strategies × {n_seeds} seeds")
    tqdm.write(f"  Missing features: {MISSING_FEATURES}")
    tqdm.write(f"  Total runs: {total_runs}")
    tqdm.write("=" * 70)

    records: list[dict] = []

    # ── Overall progress bar ──────────────────────────────────────────────────
    overall_pbar = tqdm(
        total=total_runs,
        desc="Overall",
        ncols=90,
        colour="blue",
        position=0,
        bar_format="  [{elapsed}<{remaining}, {rate_fmt}]  {l_bar}{bar}| {n_fmt}/{total_fmt}",
    )

    # ── Outer loop: scenarios ─────────────────────────────────────────────────
    sc_pbar = tqdm(
        scenarios.items(),
        desc="Scenario",
        ncols=90,
        colour="cyan",
        position=1,
        leave=True,
    )

    for scenario_name, scenario in sc_pbar:
        mech = scenario["mechanism"]
        rate = scenario["rate"]
        sc_pbar.set_description(
            f"Scenario [{scenario_name:10s}] {mech} {rate:.0%}"
        )

        # ── Strategy loop ──────────────────────────────────────────────────
        strat_pbar = tqdm(
            strategies,
            desc=f"  Strategy",
            ncols=90,
            colour="magenta",
            position=2,
            leave=False,
            bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} strats  {postfix}",
        )

        for strategy in strat_pbar:
            strat_pbar.set_description(f"  {strategy[:18]}")
            strat_aucs = []

            # ── Seed loop ──────────────────────────────────────────────────
            seed_pbar = tqdm(
                range(1, n_seeds + 1),
                desc=f"    Seeds",
                ncols=88,
                colour="yellow",
                position=3,
                leave=False,
                bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} seeds  [{elapsed}<{remaining}]  {postfix}",
            )

            for seed in seed_pbar:
                seed_pbar.set_description(f"    Seed {seed:3d}/{n_seeds}")

                result = run_one_miss_seed(
                    base_cfg, model_cfg,
                    scenario_name, scenario,
                    strategy=strategy,
                    seed=seed,
                    n=args.n,
                    outer_folds=args.outer_folds,
                )
                records.append(result)

                if "error" not in result:
                    strat_aucs.append(result["AUROC"])
                    seed_pbar.set_postfix(
                        AUROC=f"{result['AUROC']:.3f}",
                        miss=f"{result['actual_miss_rate']:.2%}",
                        refresh=True,
                    )
                else:
                    seed_pbar.set_postfix(status="ERROR", refresh=True)

                overall_pbar.update(1)
                overall_pbar.set_postfix(
                    sc=scenario_name[:8],
                    strat=strategy[:8],
                    done=len(records),
                )

            seed_pbar.close()

            # Per-strategy summary
            if strat_aucs:
                mu = np.mean(strat_aucs)
                sd = np.std(strat_aucs, ddof=1) if len(strat_aucs) > 1 else 0.0
                tqdm.write(
                    f"  ✓ {scenario_name:12s} | {strategy:15s} | "
                    f"AUROC={mu:.4f} ± {sd:.4f}"
                )
            else:
                tqdm.write(f"  ✗ {scenario_name} | {strategy}: all seeds errored")

        strat_pbar.close()

    sc_pbar.close()
    overall_pbar.close()

    elapsed = time.time() - t0
    tqdm.write(f"\n  Finished {total_runs} runs in {elapsed:.1f}s "
               f"({elapsed/max(total_runs,1):.1f}s/run avg)\n")

    # ── Post-processing steps progress bar ────────────────────────────────────
    steps = [
        "Aggregate results",
        "Save raw CSV",
        "Save summary CSV",
        "Save JSON",
        "Print pivot table",
        "Plot: missingness_sensitivity.png",
        "Plot: missingness_rate_effect.png",
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

    # 1. Aggregate
    steps_pbar.set_description("Aggregate results")
    summary = aggregate_miss_results(records)
    steps_pbar.update(1)

    # 2. Save raw CSV
    steps_pbar.set_description("Save raw CSV")
    raw_df = pd.DataFrame(records)
    raw_path = out_path.with_name(out_path.stem + "_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    tqdm.write(f"  Saved raw records ({len(records)} rows) -> {raw_path}")
    steps_pbar.update(1)

    # 3. Save summary CSV
    steps_pbar.set_description("Save summary CSV")
    summary.to_csv(out_path, index=False)
    tqdm.write(f"  Saved summary ({len(summary)} rows) -> {out_path}")
    steps_pbar.update(1)

    # 4. Save JSON
    steps_pbar.set_description("Save JSON")
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "summary": summary.to_dict(orient="records"),
                "scenarios": {k: v for k, v in scenarios.items()},
                "missing_features": MISSING_FEATURES,
                "n_seeds": n_seeds,
            },
            f, indent=2,
        )
    tqdm.write(f"  Saved JSON -> {json_path}")
    steps_pbar.update(1)

    # 5. Print pivot table
    steps_pbar.set_description("Print pivot table")
    tqdm.write("\n=== Missingness Sensitivity Summary ===")
    if not summary.empty:
        pivot = summary.pivot_table(
            index="scenario", columns="strategy", values="AUROC_mean"
        )
        pivot["bias (global-fold)"] = (
            pivot.get("global_impute", pd.Series(dtype=float))
            - pivot.get("fold_impute",  pd.Series(dtype=float))
        )
        tqdm.write(pivot.round(4).to_string())
        tqdm.write("\n--- Imputation Leakage Check ---")
        for sc in scenarios.keys():
            subset = summary[summary["scenario"] == sc]
            fold_row   = subset[subset["strategy"] == "fold_impute"]
            global_row = subset[subset["strategy"] == "global_impute"]
            if not fold_row.empty and not global_row.empty:
                bias = (global_row["AUROC_mean"].values[0]
                        - fold_row["AUROC_mean"].values[0])
                flag = "⚠ INFLATION" if bias > 0.01 else "OK ✓"
                tqdm.write(f"  {sc:14s}: bias={bias:+.4f}  {flag}")
    steps_pbar.update(1)

    # 6. Plot: missingness_sensitivity.png
    steps_pbar.set_description("Plot: missingness_sensitivity.png")
    plot_missingness_results(summary, fig_dir / "missingness_sensitivity.png")
    tqdm.write(f"  Saved -> {fig_dir / 'missingness_sensitivity.png'}")
    steps_pbar.update(1)

    # 7. Plot: missingness_rate_effect.png
    steps_pbar.set_description("Plot: missingness_rate_effect.png")
    raw_valid = (raw_df[raw_df["error"].isna()]
                 if "error" in raw_df.columns else raw_df)
    plot_miss_rate_effect(raw_valid, fig_dir / "missingness_rate_effect.png")
    tqdm.write(f"  Saved -> {fig_dir / 'missingness_rate_effect.png'}")
    steps_pbar.update(1)

    # 8. Done
    steps_pbar.set_description("Done ✓")
    steps_pbar.update(1)
    steps_pbar.close()

    total_elapsed = time.time() - t0
    tqdm.write(f"\n{'='*70}")
    tqdm.write(f"  Missingness sensitivity COMPLETE  |  total time: {total_elapsed:.1f}s")
    tqdm.write(f"{'='*70}\n")


if __name__ == "__main__":
    main()
