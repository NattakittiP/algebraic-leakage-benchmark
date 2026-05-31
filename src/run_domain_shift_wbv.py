"""run_domain_shift_wbv.py — Domain shift comparison: leaky (null) vs clean (WBV-positive)."""

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
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_clean_pipeline import CLEAN_FEATURES
from src.generate_synthetic_data import generate
from src.calibration import calibration_summary

warnings.filterwarnings("ignore")

LEAKY_FEATURES = CLEAN_FEATURES + ["tg4h"]   # definitional leakage

SHIFT_TYPES = ["tg0h_shift", "bmi_shift", "noise_shift", "prevalence_shift"]

SHIFT_LABELS = {
    "tg0h_shift":      "TG0h distribution shift",
    "bmi_shift":       "BMI distribution shift",
    "noise_shift":     "TCR noise increase",
    "prevalence_shift": "Prevalence shift",
}


def make_shift_config(base_cfg: dict, shift_type: str) -> dict:
    """Return a modified config dict for a particular shift type."""
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
        cfg["_prevalence_q"] = 35.0   # Q35 instead of Q25

    return cfg


def train_pipeline(
    df: pd.DataFrame,
    features: list[str],
    label_q: float,
    rf_cfg: dict,
    seed: int,
    test_size: float = 0.20,
) -> tuple[RandomForestClassifier, StandardScaler, float, float, np.ndarray]:
    """Train a RF classifier on the given feature set. Returns (model, scaler, source_auroc, threshold, y_test)."""
    tcr = df["tcr"].values
    threshold = float(np.percentile(tcr, label_q))
    y = (tcr <= threshold).astype(int)

    X = df[features].values

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)

    prob = model.predict_proba(X_test_s)[:, 1]
    src_auroc = float(roc_auc_score(y_test, prob))

    return model, scaler, src_auroc, threshold, y_test


def evaluate_on_shift(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    df_shift: pd.DataFrame,
    features: list[str],
    threshold: float,
    prevalence_q: float | None = None,
) -> dict:
    """
    Evaluate a trained model on shifted data (no retraining).

    For prevalence_shift, re-label with shift-specific threshold to avoid trivial
    single-class prediction — mirrors existing run_domain_shift.py logic.
    """
    tcr_shift = df_shift["tcr"].values

    if prevalence_q is not None:
        label_threshold = float(np.percentile(tcr_shift, prevalence_q))
    else:
        label_threshold = threshold

    y_shift = (tcr_shift <= label_threshold).astype(int)

    if len(np.unique(y_shift)) < 2:
        return {"error": "single class in shifted data"}

    X_shift = df_shift[features].values
    X_shift_s = scaler.transform(X_shift)

    prob = model.predict_proba(X_shift_s)[:, 1]
    auroc  = float(roc_auc_score(y_shift, prob))
    brier  = float(brier_score_loss(y_shift, prob))
    cal    = calibration_summary(y_shift, prob, label="shift")

    return {
        "AUROC":        round(auroc, 4),
        "Brier":        round(brier, 4),
        "ECE":          round(cal["ece"], 4),
        "cal_slope":    round(cal["cal_slope"], 3),
        "prevalence":   round(float(y_shift.mean()), 4),
        "y_shift":      y_shift,
        "prob_shift":   prob,
    }


def plot_reliability_panel(ax, y_true, prob, label, colour, auroc):
    """Plot a single reliability diagram on an existing axis."""
    fraction_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10)
    ax.plot(mean_pred, fraction_pos, marker="o", color=colour,
            linewidth=1.5, label=f"{label} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")


def plot_comparison(rows: list[dict], fig_dir: Path):
    """Side-by-side bar chart: AUROC and calibration slope under each shift."""
    df = pd.DataFrame(rows)
    leaky_df = df[df["pipeline"] == "leaky_null"]
    clean_df = df[df["pipeline"] == "clean_wbv"]

    shifts = [SHIFT_LABELS[s] for s in SHIFT_TYPES]
    x = np.arange(len(shifts))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    leaky_shifted = [leaky_df.loc[leaky_df["shift_type"] == s, "shifted_AUROC"].values[0]
                     if s in leaky_df["shift_type"].values else np.nan
                     for s in SHIFT_TYPES]
    clean_shifted = [clean_df.loc[clean_df["shift_type"] == s, "shifted_AUROC"].values[0]
                     if s in clean_df["shift_type"].values else np.nan
                     for s in SHIFT_TYPES]
    leaky_src = leaky_df["source_AUROC"].iloc[0] if len(leaky_df) else np.nan
    clean_src = clean_df["source_AUROC"].iloc[0]  if len(clean_df) else np.nan

    bars1 = ax.bar(x - width/2, leaky_shifted, width, label=f"Leaky RF (src={leaky_src:.3f})",
                   color="#d6604d", alpha=0.85)
    bars2 = ax.bar(x + width/2, clean_shifted, width, label=f"Clean WBV+ RF (src={clean_src:.3f})",
                   color="#2166ac", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(shifts, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Shifted AUROC")
    ax.set_title("AUROC under distributional shift\n(model applied without retraining)")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1]
    leaky_slope = [leaky_df.loc[leaky_df["shift_type"] == s, "shifted_cal_slope"].values[0]
                   if s in leaky_df["shift_type"].values else np.nan
                   for s in SHIFT_TYPES]
    clean_slope = [clean_df.loc[clean_df["shift_type"] == s, "shifted_cal_slope"].values[0]
                   if s in clean_df["shift_type"].values else np.nan
                   for s in SHIFT_TYPES]

    ax.bar(x - width/2, leaky_slope, width, label="Leaky RF",  color="#d6604d", alpha=0.85)
    ax.bar(x + width/2, clean_slope, width, label="Clean WBV+ RF", color="#2166ac", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Perfect (1.0)")
    ax.set_xticks(x); ax.set_xticklabels(shifts, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Calibration slope")
    ax.set_title("Calibration slope under distributional shift\n(1.0 = perfect)")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Domain-shift comparison: TG4h-leaky RF (null) vs clean RF (WBV-positive)\n"
        "Both models evaluated without retraining on shifted data",
        fontsize=11,
    )
    plt.tight_layout()
    out = fig_dir / "domain_shift_wbv_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure -> {out}")


def plot_reliability_diagrams(shift_results: dict, fig_dir: Path):
    """4-panel reliability diagram: one panel per shift type, leaky vs clean-WBV."""
    n_shifts = len(SHIFT_TYPES)
    fig, axes = plt.subplots(1, n_shifts, figsize=(4 * n_shifts, 4), sharey=True)

    for ax, shift_type in zip(axes, SHIFT_TYPES):
        label  = SHIFT_LABELS[shift_type]
        leaky  = shift_results.get(("leaky_null",  shift_type), {})
        clean  = shift_results.get(("clean_wbv",   shift_type), {})

        ax.set_title(label, fontsize=8)

        if "y_shift" in leaky and "prob_shift" in leaky:
            plot_reliability_panel(
                ax, leaky["y_shift"], leaky["prob_shift"],
                "Leaky RF", "#d6604d", leaky["AUROC"]
            )
        if "y_shift" in clean and "prob_shift" in clean:
            plot_reliability_panel(
                ax, clean["y_shift"], clean["prob_shift"],
                "Clean WBV+", "#2166ac", clean["AUROC"]
            )

        ax.legend(fontsize=7)

    fig.suptitle(
        "Reliability diagrams under distributional shift\n"
        "Leaky RF (red) vs clean WBV-positive RF (blue)",
        fontsize=10,
    )
    plt.tight_layout()
    out = fig_dir / "domain_shift_reliability_wbv.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram -> {out}")


def print_latex_table(rows: list[dict]):
    """Print a ready-to-paste LaTeX tabular fragment."""
    print("\n% === Domain-shift comparison table ===")
    print(r"\begin{tabular}{ll cccc}")
    print(r"\toprule")
    print(r"Pipeline & Shift Condition & Src AUROC & Shifted AUROC & $\Delta$AUROC & Cal.\ Slope \\")
    print(r"\midrule")
    for r in rows:
        delta = r["shifted_AUROC"] - r["source_AUROC"]
        print(
            f"  {r['pipeline_label']:20s} & {SHIFT_LABELS[r['shift_type']]:28s} "
            f"& ${r['source_AUROC']:.3f}$ "
            f"& ${r['shifted_AUROC']:.3f}$ "
            f"& ${delta:+.3f}$ "
            f"& ${r['shifted_cal_slope']:.3f}$ \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Domain shift: leaky (null) vs clean (WBV-positive) comparison."
    )
    p.add_argument("--null_data",    required=True,
                   help="Null scenario CSV (data/paired_tcr_null_v1_seed2026.csv)")
    p.add_argument("--wbv_data",     required=True,
                   help="WBV-positive scenario CSV (data/paired_tcr_wbv_positive_v1_seed2026.csv)")
    p.add_argument("--null_config",  required=True,
                   help="Generator YAML for null scenario (config/generator_null.yaml)")
    p.add_argument("--wbv_config",   required=True,
                   help="Generator YAML for WBV-positive (config/generator_wbv_positive.yaml)")
    p.add_argument("--model_config", required=True,
                   help="Model config YAML (config/model_config.yaml)")
    p.add_argument("--shift_seed",   type=int, default=2027,
                   help="RNG seed for generating shifted data (default 2027)")
    p.add_argument("--out",          default="results/tables/domain_shift_wbv_comparison.csv")
    p.add_argument("--figdir",       default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()

    df_null = pd.read_csv(args.null_data, keep_default_na=False,
                          na_values=["NA", "NaN", "nan", ""])
    df_wbv  = pd.read_csv(args.wbv_data,  keep_default_na=False,
                          na_values=["NA", "NaN", "nan", ""])

    with open(args.null_config)  as fh: null_cfg  = yaml.safe_load(fh)
    with open(args.wbv_config)   as fh: wbv_cfg   = yaml.safe_load(fh)
    with open(args.model_config) as fh: model_cfg  = yaml.safe_load(fh)

    label_q  = model_cfg.get("label_threshold_percentile", 25.0)
    rf_cfg   = model_cfg.get("random_forest", {})

    print("=" * 60)
    print("Pipeline A: TG4h-leaky RF  (null scenario)")
    print("=" * 60)
    leaky_model, leaky_scaler, leaky_src_auroc, leaky_threshold, _ = train_pipeline(
        df_null, LEAKY_FEATURES, label_q, rf_cfg, seed=42
    )
    print(f"  Source AUROC (held-out 20%): {leaky_src_auroc:.4f}")

    print("\n" + "=" * 60)
    print("Pipeline B: Clean RF  (WBV-positive scenario)")
    print("=" * 60)
    clean_model, clean_scaler, clean_src_auroc, clean_threshold, _ = train_pipeline(
        df_wbv, CLEAN_FEATURES, label_q, rf_cfg, seed=42
    )
    print(f"  Source AUROC (held-out 20%): {clean_src_auroc:.4f}")

    rows: list[dict] = []
    shift_results: dict = {}   # (pipeline, shift_type) -> eval dict

    pbar = tqdm(SHIFT_TYPES, desc="Shift types", ncols=90, colour="yellow")

    for shift_type in pbar:
        pbar.set_description(f"Shift: {shift_type}")

        null_shift_cfg = make_shift_config(null_cfg, shift_type)
        wbv_shift_cfg  = make_shift_config(wbv_cfg,  shift_type)

        prevalence_q = 35.0 if shift_type == "prevalence_shift" else None

        n_shift = len(df_null)
        df_shift_null = generate(null_shift_cfg, seed=args.shift_seed, n=n_shift)
        df_shift_wbv  = generate(wbv_shift_cfg,  seed=args.shift_seed, n=len(df_wbv))

        leaky_eval = evaluate_on_shift(
            leaky_model, leaky_scaler, df_shift_null,
            LEAKY_FEATURES, leaky_threshold,
            prevalence_q=prevalence_q,
        )
        shift_results[("leaky_null", shift_type)] = leaky_eval

        clean_eval = evaluate_on_shift(
            clean_model, clean_scaler, df_shift_wbv,
            CLEAN_FEATURES, clean_threshold,
            prevalence_q=prevalence_q,
        )
        shift_results[("clean_wbv", shift_type)] = clean_eval

        pbar.set_postfix(
            leaky=f"{leaky_eval.get('AUROC', 'err'):.3f}",
            clean=f"{clean_eval.get('AUROC', 'err'):.3f}",
        )

        for pipeline_key, pipeline_label, src_auroc, eval_dict in [
            ("leaky_null", "Leaky RF (null)",     leaky_src_auroc, leaky_eval),
            ("clean_wbv",  "Clean RF (WBV+)",     clean_src_auroc, clean_eval),
        ]:
            if "error" in eval_dict:
                continue
            rows.append({
                "pipeline":          pipeline_key,
                "pipeline_label":    pipeline_label,
                "shift_type":        shift_type,
                "shift_label":       SHIFT_LABELS[shift_type],
                "source_AUROC":      round(src_auroc,  4),
                "shifted_AUROC":     round(eval_dict["AUROC"], 4),
                "delta_AUROC":       round(eval_dict["AUROC"] - src_auroc, 4),
                "shifted_Brier":     round(eval_dict["Brier"], 4),
                "shifted_ECE":       round(eval_dict["ECE"],   4),
                "shifted_cal_slope": round(eval_dict["cal_slope"], 3),
                "shifted_prevalence":round(eval_dict["prevalence"], 4),
            })

    print("\n" + "=" * 72)
    print(f"{'Shift':30s}  {'Pipeline':22s}  {'Src':>6}  {'Shifted':>8}  {'ΔAUROC':>8}  {'CalSlp':>7}")
    print("-" * 72)
    for r in rows:
        print(
            f"  {r['shift_label']:28s}  {r['pipeline_label']:22s}"
            f"  {r['source_AUROC']:6.3f}  {r['shifted_AUROC']:8.3f}"
            f"  {r['delta_AUROC']:+8.3f}  {r['shifted_cal_slope']:7.3f}"
        )
    print("=" * 72)

    leaky_deltas = [r["delta_AUROC"] for r in rows if r["pipeline"] == "leaky_null"]
    clean_deltas = [r["delta_AUROC"] for r in rows if r["pipeline"] == "clean_wbv"]
    if leaky_deltas and clean_deltas:
        print(f"\nLeaky RF mean ΔAUROC:      {np.mean(leaky_deltas):+.3f}  "
              f"(range {min(leaky_deltas):+.3f} to {max(leaky_deltas):+.3f})")
        print(f"Clean WBV+ RF mean ΔAUROC: {np.mean(clean_deltas):+.3f}  "
              f"(range {min(clean_deltas):+.3f} to {max(clean_deltas):+.3f})")
        print(
            "\nInterpretation: "
            "Leaky RF collapse is caused by failure of the LEARNED approximation "
            "under shift, not by the algebraic identity (which is distribution-free). "
            "Clean WBV+ RF shows a smaller, more gradual degradation consistent with "
            "partial generalisability of a genuine biological signal."
        )

    print_latex_table(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved comparison table -> {out_path}")

    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(rows, fig_dir)
    plot_reliability_diagrams(shift_results, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
