"""Outlier/plausibility stress test: cleaning strategy comparison across contamination scenarios."""

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import CLEAN_FEATURES
from src.utils import FoldSealedScaler, FoldSealedWinsorizer

warnings.filterwarnings("ignore")


PLAUSIBILITY_BOUNDS: dict[str, tuple[float, float]] = {
    "age":  (18.0,  90.0),
    "bmi":  (14.0,  60.0),
    "hct":  (20.0,  65.0),
    "tp":   (4.0,   10.0),
    "wbv":  (2.0,   10.0),
    "hdl":  (10.0,  120.0),
    "ldl":  (30.0,  300.0),
    "tg0h": (50.0,  3000.0),
    "tg4h": (0.0,   2000.0),
    "tcr":  (-50.0, 100.0),
}

OUTLIER_SCENARIOS = {
    "clean_0pct":    {"rate": 0.00, "types": []},
    "neg_tg4h_1pct": {"rate": 0.01, "types": ["neg_tg4h"]},
    "neg_tg4h_5pct": {"rate": 0.05, "types": ["neg_tg4h"]},
    "ext_hct_1pct":  {"rate": 0.01, "types": ["extreme_hct"]},
    "ext_hct_5pct":  {"rate": 0.05, "types": ["extreme_hct"]},
    "ext_tp_1pct":   {"rate": 0.01, "types": ["extreme_tp"]},
    "mixed_1pct":    {"rate": 0.01, "types": ["neg_tg4h", "extreme_hct", "extreme_tp"]},
    "mixed_5pct":    {"rate": 0.05, "types": ["neg_tg4h", "extreme_hct", "extreme_tp"]},
}

CLEANING_STRATEGIES = [
    "no_cleaning",
    "global_cleaning",
    "fold_cleaning",
    "prespecified_clean",
]


def inject_outliers(
    df: pd.DataFrame,
    scenario: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject physiologically implausible values into df."""
    df = df.copy()
    rate = scenario["rate"]
    types = scenario["types"]
    n = len(df)

    if rate == 0 or not types:
        return df

    n_contam = max(1, int(n * rate))

    for otype in types:
        idx = rng.choice(n, size=n_contam, replace=False)

        if otype == "neg_tg4h":
            df.loc[idx, "tg4h"] = rng.uniform(-200, -1, size=n_contam)

        elif otype == "extreme_tg4h":
            df.loc[idx, "tg4h"] = rng.uniform(1500, 3000, size=n_contam)

        elif otype == "extreme_hct":
            n_low  = n_contam // 2
            n_high = n_contam - n_low
            df.loc[idx[:n_low],  "hct"] = rng.uniform(8.0,  19.9, size=n_low)
            df.loc[idx[n_low:],  "hct"] = rng.uniform(65.1, 80.0, size=n_high)
            df.loc[idx, "wbv"] = np.round(
                0.12 * df.loc[idx, "hct"] + 0.17 * (df.loc[idx, "tp"] - 2.07), 3
            )

        elif otype == "extreme_tp":
            n_low  = n_contam // 2
            n_high = n_contam - n_low
            df.loc[idx[:n_low],  "tp"] = rng.uniform(1.0,  3.9,  size=n_low)
            df.loc[idx[n_low:],  "tp"] = rng.uniform(10.1, 14.0, size=n_high)
            df.loc[idx, "wbv"] = np.round(
                0.12 * df.loc[idx, "hct"] + 0.17 * (df.loc[idx, "tp"] - 2.07), 3
            )

    return df


def count_implausible(df: pd.DataFrame) -> dict:
    """Count rows with at least one implausible value per variable."""
    return {
        col: int(((df[col] < lo) | (df[col] > hi)).sum())
        for col, (lo, hi) in PLAUSIBILITY_BOUNDS.items()
        if col in df.columns
        and int(((df[col] < lo) | (df[col] > hi)).sum()) > 0
    }


def apply_global_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clip all features to plausibility bounds BEFORE any split (leaky)."""
    df = df.copy()
    for col, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def apply_fold_cleaning(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Clip to pre-specified plausibility bounds, applied per-fold."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for j, feat in enumerate(feature_names):
        if feat in PLAUSIBILITY_BOUNDS:
            lo, hi = PLAUSIBILITY_BOUNDS[feat]
            X_tr[:, j] = np.clip(X_tr[:, j], lo, hi)
            X_te[:, j] = np.clip(X_te[:, j], lo, hi)
    return X_tr, X_te


def apply_prespecified_clean(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Best practice: pre-registered bounds, applied fold-locally."""
    return apply_fold_cleaning(X_train, X_test, feature_names)


def _eval_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    tcr_train: np.ndarray,
    tcr_test: np.ndarray,
    label_q: float,
    seed: int,
    strategy: str,
    feature_names: list[str],
    X_full_global: np.ndarray,
    global_idx_train: np.ndarray,
    global_idx_test: np.ndarray,
) -> dict:
    """One outer fold with a given cleaning strategy."""
    if strategy == "no_cleaning":
        X_tr = X_train.copy()
        X_te = X_test.copy()
    elif strategy == "global_cleaning":
        X_tr = X_full_global[global_idx_train]
        X_te = X_full_global[global_idx_test]
    elif strategy == "fold_cleaning":
        X_tr, X_te = apply_fold_cleaning(X_train, X_test, feature_names)
    elif strategy == "prespecified_clean":
        X_tr, X_te = apply_prespecified_clean(X_train, X_test, feature_names)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    win = FoldSealedWinsorizer(lower_pct=1.0, upper_pct=99.0)
    X_tr_w = win.fit_transform(X_tr)
    X_te_w = win.transform(X_te)

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


def run_one_outlier_seed(
    base_cfg: dict,
    model_cfg: dict,
    scenario_name: str,
    scenario: dict,
    strategy: str,
    seed: int,
    n: int,
    outer_folds: int = 5,
) -> dict:
    """Generate data, inject outliers, run clean 5-fold CV with given strategy."""
    rng = np.random.default_rng(seed + 10000)

    try:
        df_clean = generate(base_cfg, seed=seed, n=n)
    except Exception as e:
        return {"scenario": scenario_name, "strategy": strategy,
                "seed": seed, "error": str(e)}

    df = inject_outliers(df_clean, scenario, rng)
    X = df[CLEAN_FEATURES].values.astype(float)
    tcr = df["tcr"].values

    label_q = model_cfg.get("label_threshold_percentile", 25.0)
    y_global = (tcr <= np.percentile(tcr, label_q)).astype(int)

    df_global = apply_global_cleaning(df)
    X_global = df_global[CLEAN_FEATURES].values.astype(float)

    implaus_counts = count_implausible(df)
    n_implausible = sum(implaus_counts.values())

    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    aucs, briers = [], []

    for train_idx, test_idx in tqdm(
        skf.split(X, y_global),
        total=outer_folds,
        desc=f"    Folds ({strategy[:12]})",
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
            feature_names=CLEAN_FEATURES,
            X_full_global=X_global,
            global_idx_train=train_idx,
            global_idx_test=test_idx,
        )
        if fold_res:
            aucs.append(fold_res["auc"])
            briers.append(fold_res["brier"])

    if not aucs:
        return {"scenario": scenario_name, "strategy": strategy,
                "seed": seed, "error": "no valid folds"}

    return {
        "scenario":        scenario_name,
        "contam_rate":     scenario["rate"],
        "outlier_types":   "+".join(scenario["types"]) if scenario["types"] else "none",
        "strategy":        strategy,
        "seed":            seed,
        "n_implausible":   n_implausible,
        "implaus_pct":     round(100 * n_implausible / (n * len(CLEAN_FEATURES)), 2),
        "AUROC":           round(float(np.mean(aucs)), 4),
        "Brier":           round(float(np.mean(briers)), 4),
        "n_folds":         len(aucs),
    }


def aggregate_outlier_results(records: list[dict]) -> pd.DataFrame:
    """Summary per (scenario × strategy)."""
    df = pd.DataFrame([r for r in records if "error" not in r])
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (scenario, strategy), grp in df.groupby(["scenario", "strategy"]):
        aucs = grp["AUROC"].values
        rows.append({
            "scenario":      scenario,
            "contam_rate":   grp["contam_rate"].iloc[0],
            "outlier_types": grp["outlier_types"].iloc[0],
            "strategy":      strategy,
            "n_seeds":       len(aucs),
            "AUROC_mean":    round(float(aucs.mean()), 4),
            "AUROC_sd":      round(float(aucs.std(ddof=1)), 4) if len(aucs) > 1 else 0.0,
            "AUROC_p2_5":    round(float(np.percentile(aucs, 2.5)), 4),
            "AUROC_p97_5":   round(float(np.percentile(aucs, 97.5)), 4),
            "Brier_mean":    round(float(grp["Brier"].mean()), 4),
        })

    result = pd.DataFrame(rows)
    sc_order   = list(OUTLIER_SCENARIOS.keys())
    strat_order = CLEANING_STRATEGIES
    result["_sc_sort"]   = result["scenario"].map({s: i for i, s in enumerate(sc_order)})
    result["_st_sort"]   = result["strategy"].map({s: i for i, s in enumerate(strat_order)})
    result = (result.sort_values(["_sc_sort", "_st_sort"])
              .drop(columns=["_sc_sort", "_st_sort"])
              .reset_index(drop=True))
    return result


def plot_outlier_heatmap(summary: pd.DataFrame, out_path: Path):
    """Heatmap: AUROC by scenario × strategy."""
    sc_order    = [s for s in OUTLIER_SCENARIOS.keys() if s in summary["scenario"].values]
    strat_order = [s for s in CLEANING_STRATEGIES if s in summary["strategy"].values]

    matrix = np.full((len(sc_order), len(strat_order)), np.nan)
    for i, sc in enumerate(sc_order):
        for j, st in enumerate(strat_order):
            row = summary[(summary["scenario"] == sc) & (summary["strategy"] == st)]
            if not row.empty:
                matrix[i, j] = row["AUROC_mean"].values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.40, vmax=0.60, aspect="auto")

    ax.set_xticks(range(len(strat_order)))
    ax.set_xticklabels(strat_order, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(sc_order)))
    ax.set_yticklabels(sc_order, fontsize=9)

    for i in range(len(sc_order)):
        for j in range(len(strat_order)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "black" if 0.44 <= val <= 0.56 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean AUROC")
    ax.set_title(
        "Outlier Stress Test: AUROC by Scenario × Cleaning Strategy\n"
        "(green ≈ 0.50 = correct null recovery)",
        fontsize=11,
    )
    ax.set_xlabel("Cleaning Strategy")
    ax.set_ylabel("Outlier Scenario")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_strategy_comparison(summary: pd.DataFrame, out_path: Path):
    """Bar chart: AUROC and SD by strategy, grouped by scenario."""
    mixed_sc = [s for s in summary["scenario"].values if "mixed" in s]
    if not mixed_sc:
        mixed_sc = list(summary["scenario"].unique()[:4])

    strats  = [s for s in CLEANING_STRATEGIES if s in summary["strategy"].values]
    n_sc    = len(mixed_sc)
    x       = np.arange(len(strats))
    width   = 0.8 / max(n_sc, 1)
    colors  = plt.cm.Set2(np.linspace(0, 1, n_sc))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, metric in enumerate(["AUROC_mean", "AUROC_sd"]):
        ax = axes[ax_idx]
        for i, sc in enumerate(mixed_sc):
            sc_df = summary[summary["scenario"] == sc].set_index("strategy")
            vals  = [sc_df.loc[st, metric] if st in sc_df.index else np.nan for st in strats]
            offset = (i - (n_sc - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=sc,
                   color=colors[i], alpha=0.85, edgecolor="gray", linewidth=0.5)

        if metric == "AUROC_mean":
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Chance (0.50)")
            ax.set_ylabel("Mean AUROC")
            ax.set_title("AUROC by Cleaning Strategy\n(mixed outlier scenarios)")
            ax.set_ylim(0.35, 0.65)
        else:
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
            ax.set_ylabel("AUROC SD across seeds")
            ax.set_title("AUROC Variability by Cleaning Strategy\n(lower = more stable)")
            ax.set_ylim(bottom=0)

        ax.set_xticks(x)
        ax.set_xticklabels(strats, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=7)

    plt.suptitle("Outlier Stress Test — Cleaning Strategy Comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_contamination_rate_effect(summary: pd.DataFrame, out_path: Path):
    """Line plot: AUROC vs contamination rate by cleaning strategy."""
    neg_scenarios = {s: d for s, d in OUTLIER_SCENARIOS.items()
                     if "neg_tg4h" in d["types"] or s == "clean_0pct"}
    sc_in_summary = [s for s in neg_scenarios if s in summary["scenario"].values]
    if len(sc_in_summary) < 2:
        sc_in_summary = list(summary["scenario"].unique()[:3])

    fig, ax = plt.subplots(figsize=(9, 5))
    strat_styles = {
        "no_cleaning":        ("o-", "#d73027"),
        "global_cleaning":    ("s--", "#fc8d59"),
        "fold_cleaning":      ("^-", "#4575b4"),
        "prespecified_clean": ("D-", "#1a9850"),
    }

    for strategy, (linestyle, color) in strat_styles.items():
        sc_df = summary[
            (summary["strategy"] == strategy) &
            (summary["scenario"].isin(sc_in_summary))
        ].sort_values("contam_rate")
        if sc_df.empty:
            continue
        ax.plot(sc_df["contam_rate"] * 100, sc_df["AUROC_mean"],
                linestyle, color=color, label=strategy, linewidth=2, markersize=7)
        ax.fill_between(
            sc_df["contam_rate"] * 100,
            sc_df["AUROC_mean"] - sc_df["AUROC_sd"],
            sc_df["AUROC_mean"] + sc_df["AUROC_sd"],
            alpha=0.10, color=color,
        )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="Chance (0.50)")
    ax.set_xlabel("Contamination rate (%)")
    ax.set_ylabel("Mean AUROC (±SD)")
    ax.set_title("Effect of Contamination Rate on AUROC\nby Cleaning Strategy (neg TG4h)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Outlier/plausibility stress test: cleaning strategy comparison."
    )
    p.add_argument("--config",       default="config/generator_null.yaml")
    p.add_argument("--model_config", default="config/model_config.yaml")
    p.add_argument("--out",          default="results/tables/outlier_stress_test.csv")
    p.add_argument("--figdir",       default="results/figures")
    p.add_argument("--n",            type=int, default=1500)
    p.add_argument("--seeds",        type=int, default=15)
    p.add_argument("--outer_folds",  type=int, default=5)
    p.add_argument("--quick",        action="store_true",
                   help="Quick mode: 3 seeds, 4 scenarios, 3 strategies")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    n_seeds = 3 if args.quick else args.seeds

    if args.quick:
        scenarios_to_run = {
            k: v for k, v in OUTLIER_SCENARIOS.items()
            if k in ("clean_0pct", "neg_tg4h_1pct", "neg_tg4h_5pct", "mixed_5pct")
        }
        strategies_to_run = ["no_cleaning", "fold_cleaning", "prespecified_clean"]
    else:
        scenarios_to_run = OUTLIER_SCENARIOS
        strategies_to_run = CLEANING_STRATEGIES

    n_sc     = len(scenarios_to_run)
    n_st     = len(strategies_to_run)
    total_runs = n_sc * n_st * n_seeds

    tqdm.write("=" * 70)
    tqdm.write("  Outlier / Plausibility Stress Test (Plan item 39)")
    tqdm.write(f"  {n_sc} scenarios × {n_st} strategies × {n_seeds} seeds")
    tqdm.write(f"  Total runs: {total_runs}")
    tqdm.write(f"  Plausibility bounds: {list(PLAUSIBILITY_BOUNDS.keys())}")
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

    sc_pbar = tqdm(
        scenarios_to_run.items(),
        desc="Scenario",
        ncols=90,
        colour="red",
        position=1,
        leave=True,
    )

    for scenario_name, scenario in sc_pbar:
        rate  = scenario["rate"]
        types = scenario["types"]
        sc_pbar.set_description(
            f"Scenario [{scenario_name:16s}] rate={rate:.0%}"
        )

        strat_pbar = tqdm(
            strategies_to_run,
            desc="  Strategy",
            ncols=90,
            colour="magenta",
            position=2,
            leave=False,
            bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} strats  {postfix}",
        )

        for strategy in strat_pbar:
            strat_pbar.set_description(f"  {strategy[:20]}")
            strat_aucs = []

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

                result = run_one_outlier_seed(
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
                        implaus=f"{result['n_implausible']}",
                        refresh=True,
                    )
                else:
                    seed_pbar.set_postfix(status="ERROR", refresh=True)

                overall_pbar.update(1)
                overall_pbar.set_postfix(
                    sc=scenario_name[:10],
                    strat=strategy[:8],
                    done=len(records),
                )

            seed_pbar.close()

            if strat_aucs:
                mu = np.mean(strat_aucs)
                sd = np.std(strat_aucs, ddof=1) if len(strat_aucs) > 1 else 0.0
                flag = ("⚠ DEGRADED" if mu < 0.44
                        else "⬆ INFLATED" if mu > 0.56
                        else "OK ✓")
                tqdm.write(
                    f"  ✓ {scenario_name:18s} | {strategy:20s} | "
                    f"AUROC={mu:.4f} ± {sd:.4f}  {flag}"
                )
            else:
                tqdm.write(f"  ✗ {scenario_name} | {strategy}: all seeds errored")

        strat_pbar.close()

    sc_pbar.close()
    overall_pbar.close()

    elapsed = time.time() - t0
    tqdm.write(f"\n  Finished {total_runs} runs in {elapsed:.1f}s "
               f"({elapsed/max(total_runs,1):.1f}s/run avg)\n")

    steps = [
        "Aggregate results",
        "Save raw CSV",
        "Save summary CSV",
        "Save JSON",
        "Print pivot table",
        "Plot: outlier_stress_heatmap.png",
        "Plot: outlier_strategy_comparison.png",
        "Plot: outlier_contamination_rate.png",
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
    summary = aggregate_outlier_results(records)
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
                "plausibility_bounds": {k: list(v) for k, v in PLAUSIBILITY_BOUNDS.items()},
                "scenarios": {
                    k: {"rate": v["rate"], "types": v["types"]}
                    for k, v in scenarios_to_run.items()
                },
                "n_seeds": n_seeds,
            },
            jf, indent=2,
        )
    tqdm.write(f"  Saved JSON -> {json_path}")
    steps_pbar.update(1)

    steps_pbar.set_description("Print pivot table")
    tqdm.write("\n=== Outlier Stress Test Summary ===")
    if not summary.empty:
        try:
            pivot = summary.pivot_table(
                index="scenario", columns="strategy", values="AUROC_mean"
            )
            tqdm.write(pivot.round(4).to_string())
        except Exception:
            tqdm.write(
                summary[["scenario", "strategy", "AUROC_mean", "AUROC_sd"]]
                .round(4).to_string(index=False)
            )
        tqdm.write("\n--- Outlier Impact Check ---")
        for sc in list(scenarios_to_run.keys()):
            subset = summary[summary["scenario"] == sc]
            if not subset.empty:
                best  = subset.loc[subset["AUROC_mean"].sub(0.5).abs().idxmin(), "strategy"]
                worst = subset.loc[subset["AUROC_mean"].sub(0.5).abs().idxmax(), "strategy"]
                tqdm.write(f"  {sc:18s}: best={best}  worst={worst}")
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: outlier_stress_heatmap.png")
    plot_outlier_heatmap(summary, fig_dir / "outlier_stress_heatmap.png")
    tqdm.write(f"  Saved -> {fig_dir / 'outlier_stress_heatmap.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: outlier_strategy_comparison.png")
    plot_strategy_comparison(summary, fig_dir / "outlier_strategy_comparison.png")
    tqdm.write(f"  Saved -> {fig_dir / 'outlier_strategy_comparison.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Plot: outlier_contamination_rate.png")
    plot_contamination_rate_effect(summary, fig_dir / "outlier_contamination_rate.png")
    tqdm.write(f"  Saved -> {fig_dir / 'outlier_contamination_rate.png'}")
    steps_pbar.update(1)

    steps_pbar.set_description("Done ✓")
    steps_pbar.update(1)
    steps_pbar.close()

    total_elapsed = time.time() - t0
    tqdm.write(f"\n{'='*70}")
    tqdm.write(f"  Outlier stress test COMPLETE  |  total time: {total_elapsed:.1f}s")
    tqdm.write(f"{'='*70}\n")


if __name__ == "__main__":
    main()
