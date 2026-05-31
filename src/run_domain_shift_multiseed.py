"""Multi-seed domain shift evaluation across 10 independent training seeds."""

from __future__ import annotations

import argparse
import sys
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_clean_pipeline import CLEAN_FEATURES
from src.generate_synthetic_data import generate
from src.calibration import calibration_summary

warnings.filterwarnings("ignore")

LEAKY_FEATURES = CLEAN_FEATURES + ["tg4h"]

SHIFT_TYPES = ["tg0h_shift", "bmi_shift", "noise_shift", "prevalence_shift"]

SHIFT_LABELS = {
    "tg0h_shift":       "TG0h distribution shift",
    "bmi_shift":        "BMI distribution shift",
    "noise_shift":      "TCR noise increase",
    "prevalence_shift": "Prevalence shift",
}

BASE_SEEDS = list(range(10))


def make_shift_config(base_cfg: dict, shift_type: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if shift_type == "tg0h_shift":
        cfg["tg0h_mu_log"] = base_cfg.get("tg0h_mu_log", 5.5) + 0.3
        cfg["tg0h_shift"]  = base_cfg.get("tg0h_shift", 350.0) + 100.0
    elif shift_type == "bmi_shift":
        bmi_sd = base_cfg.get("bmi_sd", 3.0)
        cfg["bmi_mu"] = base_cfg.get("bmi_mu", 24.0) + 2 * bmi_sd
    elif shift_type == "noise_shift":
        cfg["tcr_sd"] = base_cfg.get("tcr_sd", 18.6) * 1.5
    elif shift_type == "prevalence_shift":
        cfg["_prevalence_q"] = 35.0
    return cfg


def train_pipeline(
    df: pd.DataFrame,
    features: list[str],
    label_q: float,
    rf_cfg: dict,
    seed: int,
    test_size: float = 0.20,
):
    """Train RF and return (model, scaler, source_auroc, threshold)."""
    tcr = df["tcr"].values
    threshold = float(np.percentile(tcr, label_q))
    y = (tcr <= threshold).astype(int)
    X = df[features].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)
    src_auroc = float(roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1]))
    return model, scaler, src_auroc, threshold


def evaluate_on_shift(
    model,
    scaler,
    df_shift: pd.DataFrame,
    features: list[str],
    threshold: float,
    prevalence_q: float | None = None,
) -> dict:
    tcr_shift = df_shift["tcr"].values
    label_threshold = (
        float(np.percentile(tcr_shift, prevalence_q))
        if prevalence_q is not None
        else threshold
    )
    y_shift = (tcr_shift <= label_threshold).astype(int)

    if len(np.unique(y_shift)) < 2:
        return {"error": "single class"}

    X_s = scaler.transform(df_shift[features].values)
    prob = model.predict_proba(X_s)[:, 1]

    auroc = float(roc_auc_score(y_shift, prob))
    brier = float(brier_score_loss(y_shift, prob))
    cal   = calibration_summary(y_shift, prob, label="shift")

    return {
        "AUROC":      round(auroc, 4),
        "Brier":      round(brier, 4),
        "ECE":        round(cal["ece"], 4),
        "cal_slope":  round(cal["cal_slope"], 3),
        "prevalence": round(float(y_shift.mean()), 4),
    }


def summarise(df_per_seed: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± SD over seeds, grouped by pipeline × shift_type."""
    metrics = ["source_AUROC", "shifted_AUROC", "delta_AUROC",
               "shifted_Brier", "shifted_ECE", "shifted_cal_slope"]
    rows = []
    for (pipeline, shift_type), grp in df_per_seed.groupby(["pipeline", "shift_type"]):
        row = {"pipeline": pipeline, "shift_type": shift_type,
               "shift_label": SHIFT_LABELS[shift_type], "n_seeds": len(grp)}
        for m in metrics:
            if m in grp.columns:
                row[f"{m}_mean"] = round(grp[m].mean(), 4)
                row[f"{m}_sd"]   = round(grp[m].std(ddof=1), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_multiseed_comparison(summary: pd.DataFrame, fig_dir: Path):
    """Bar chart with error bars (mean ± SD) for shifted AUROC and cal_slope."""
    leaky = summary[summary["pipeline"] == "leaky_null"].set_index("shift_type")
    clean = summary[summary["pipeline"] == "clean_wbv"].set_index("shift_type")

    x = np.arange(len(SHIFT_TYPES))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(x - width/2,
           [leaky.loc[s, "shifted_AUROC_mean"] for s in SHIFT_TYPES],
           width,
           yerr=[leaky.loc[s, "shifted_AUROC_sd"] for s in SHIFT_TYPES],
           label="Leaky RF (null)", color="#d6604d", alpha=0.85, capsize=4)
    ax.bar(x + width/2,
           [clean.loc[s, "shifted_AUROC_mean"] for s in SHIFT_TYPES],
           width,
           yerr=[clean.loc[s, "shifted_AUROC_sd"] for s in SHIFT_TYPES],
           label="Clean WBV+ RF", color="#2166ac", alpha=0.85, capsize=4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([SHIFT_LABELS[s] for s in SHIFT_TYPES],
                       rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Shifted AUROC (mean ± SD, 10 seeds)")
    ax.set_title("AUROC under distributional shift\n(mean ± SD across 10 seeds)")
    ax.set_ylim(0.3, 1.10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width/2,
           [leaky.loc[s, "shifted_cal_slope_mean"] for s in SHIFT_TYPES],
           width,
           yerr=[leaky.loc[s, "shifted_cal_slope_sd"] for s in SHIFT_TYPES],
           label="Leaky RF", color="#d6604d", alpha=0.85, capsize=4)
    ax.bar(x + width/2,
           [clean.loc[s, "shifted_cal_slope_mean"] for s in SHIFT_TYPES],
           width,
           yerr=[clean.loc[s, "shifted_cal_slope_sd"] for s in SHIFT_TYPES],
           label="Clean WBV+ RF", color="#2166ac", alpha=0.85, capsize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Perfect (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([SHIFT_LABELS[s] for s in SHIFT_TYPES],
                       rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Calibration slope (mean ± SD, 10 seeds)")
    ax.set_title("Calibration slope under distributional shift\n(1.0 = perfect)")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Multi-seed domain-shift comparison: TG4h-leaky RF vs clean WBV+ RF\n"
        "Mean ± SD across 10 independent training seeds",
        fontsize=11,
    )
    plt.tight_layout()
    out = fig_dir / "domain_shift_multiseed_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved multi-seed comparison figure -> {out}")


def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 90)
    print("MULTI-SEED DOMAIN SHIFT SUMMARY  (mean ± SD, 10 seeds)")
    print("=" * 90)
    for pipeline, grp in summary.groupby("pipeline"):
        print(f"\n  Pipeline: {pipeline}")
        print(f"  {'Shift':30s}  {'Src AUROC':>14}  {'Shifted AUROC':>16}  {'Cal Slope':>12}")
        print("  " + "-" * 78)
        for _, row in grp.iterrows():
            print(
                f"  {row['shift_label']:30s}  "
                f"{row['source_AUROC_mean']:.3f} ± {row['source_AUROC_sd']:.3f}  "
                f"{row['shifted_AUROC_mean']:.3f} ± {row['shifted_AUROC_sd']:.3f}  "
                f"  {row['shifted_cal_slope_mean']:.3f} ± {row['shifted_cal_slope_sd']:.3f}"
            )
    print("=" * 90)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed domain shift evaluation.")
    p.add_argument("--null_data",    required=True)
    p.add_argument("--wbv_data",     required=True)
    p.add_argument("--null_config",  required=True)
    p.add_argument("--wbv_config",   required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--shift_seed",   type=int, default=2027,
                   help="Fixed seed for shifted data generation (default 2027)")
    p.add_argument("--n_seeds",      type=int, default=10,
                   help="Number of independent training seeds (default 10)")
    p.add_argument("--out_dir",      default="results/tables")
    p.add_argument("--figdir",       default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()

    df_null = pd.read_csv(args.null_data, keep_default_na=False,
                          na_values=["NA", "NaN", "nan", ""])
    df_wbv  = pd.read_csv(args.wbv_data,  keep_default_na=False,
                          na_values=["NA", "NaN", "nan", ""])

    with open(args.null_config)  as fh: null_cfg   = yaml.safe_load(fh)
    with open(args.wbv_config)   as fh: wbv_cfg    = yaml.safe_load(fh)
    with open(args.model_config) as fh: model_cfg  = yaml.safe_load(fh)

    label_q = model_cfg.get("label_threshold_percentile", 25.0)
    rf_cfg  = model_cfg.get("random_forest", {})

    seeds = list(range(args.n_seeds))

    print("Pre-generating shifted datasets (fixed shift_seed={})...".format(args.shift_seed))
    shifted_null: dict[str, pd.DataFrame] = {}
    shifted_wbv:  dict[str, pd.DataFrame] = {}
    for shift_type in SHIFT_TYPES:
        null_scfg = make_shift_config(null_cfg, shift_type)
        wbv_scfg  = make_shift_config(wbv_cfg,  shift_type)
        shifted_null[shift_type] = generate(null_scfg, seed=args.shift_seed, n=len(df_null))
        shifted_wbv[shift_type]  = generate(wbv_scfg,  seed=args.shift_seed, n=len(df_wbv))
    print("  Done.\n")

    all_rows: list[dict] = []

    seed_pbar = tqdm(seeds, desc="Training seeds", ncols=90, colour="cyan")
    for seed in seed_pbar:
        seed_pbar.set_description(f"Seed {seed:2d}")

        leaky_model, leaky_scaler, leaky_src, leaky_thr = train_pipeline(
            df_null, LEAKY_FEATURES, label_q, rf_cfg, seed=seed
        )

        clean_model, clean_scaler, clean_src, clean_thr = train_pipeline(
            df_wbv, CLEAN_FEATURES, label_q, rf_cfg, seed=seed
        )

        seed_pbar.set_postfix(leaky_src=f"{leaky_src:.3f}", clean_src=f"{clean_src:.3f}")

        for shift_type in SHIFT_TYPES:
            prevalence_q = 35.0 if shift_type == "prevalence_shift" else None

            leaky_eval = evaluate_on_shift(
                leaky_model, leaky_scaler,
                shifted_null[shift_type], LEAKY_FEATURES, leaky_thr, prevalence_q
            )
            clean_eval = evaluate_on_shift(
                clean_model, clean_scaler,
                shifted_wbv[shift_type],  CLEAN_FEATURES, clean_thr, prevalence_q
            )

            for pipeline_key, pipeline_label, src_auroc, ev in [
                ("leaky_null", "Leaky RF (null)",  leaky_src, leaky_eval),
                ("clean_wbv",  "Clean RF (WBV+)",  clean_src, clean_eval),
            ]:
                if "error" in ev:
                    continue
                all_rows.append({
                    "seed":               seed,
                    "pipeline":           pipeline_key,
                    "pipeline_label":     pipeline_label,
                    "shift_type":         shift_type,
                    "shift_label":        SHIFT_LABELS[shift_type],
                    "source_AUROC":       round(src_auroc, 4),
                    "shifted_AUROC":      round(ev["AUROC"], 4),
                    "delta_AUROC":        round(ev["AUROC"] - src_auroc, 4),
                    "shifted_Brier":      round(ev["Brier"], 4),
                    "shifted_ECE":        round(ev["ECE"],   4),
                    "shifted_cal_slope":  round(ev["cal_slope"], 3),
                })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_per_seed = pd.DataFrame(all_rows)
    per_seed_path = out_dir / "domain_shift_multiseed.csv"
    df_per_seed.to_csv(per_seed_path, index=False)
    print(f"\nSaved per-seed data -> {per_seed_path}")

    summary = summarise(df_per_seed)
    summary_path = out_dir / "domain_shift_multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    print_summary(summary)

    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_multiseed_comparison(summary, fig_dir)

    print("\n" + "=" * 90)
    print("LATEX-READY NUMBERS (mean ± SD) for Table update")
    print("=" * 90)
    for pipeline, grp in summary.groupby("pipeline"):
        print(f"\n  {pipeline}:")
        for _, row in grp.iterrows():
            print(
                f"    {row['shift_label']:30s}: "
                f"Shifted AUROC = ${row['shifted_AUROC_mean']:.3f} \\pm {row['shifted_AUROC_sd']:.3f}$, "
                f"Cal Slope = ${row['shifted_cal_slope_mean']:.3f} \\pm {row['shifted_cal_slope_sd']:.3f}$, "
                f"Brier = ${row['shifted_Brier_mean']:.3f} \\pm {row['shifted_Brier_sd']:.3f}$, "
                f"ECE = ${row['shifted_ECE_mean']:.3f} \\pm {row['shifted_ECE_sd']:.3f}$"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
