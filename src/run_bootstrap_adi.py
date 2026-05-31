"""run_bootstrap_adi.py — Bootstrap confidence intervals for the Attribution Distortion Index (ADI)."""

from __future__ import annotations

import argparse
import sys
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed — cannot compute SHAP-based ADI.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_clean_pipeline import CLEAN_FEATURES

warnings.filterwarnings("ignore")

LEAKY_FEATURES  = CLEAN_FEATURES + ["tg4h"]   # definitional leakage feature set
SHARED_FEATURES = CLEAN_FEATURES

PAPER_FEATURES  = ["bmi", "age", "hdl", "tg0h", "ldl", "wbv", "hct", "tp", "sex"]


def _shap_mean_abs(model, X: np.ndarray, clf_name: str) -> np.ndarray:
    """Return mean |SHAP| vector (length = n_features) for the positive class."""
    if "RF" in clf_name or "XGB" in clf_name:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
    else:
        # LogisticRegression — LinearExplainer
        background = shap.sample(X, min(50, len(X)))
        explainer  = shap.LinearExplainer(model, background,
                                          feature_perturbation="correlation_dependent")
        sv = explainer.shap_values(X)

    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    return np.abs(sv).mean(axis=0)


def feature_ranks(mean_abs: np.ndarray, feature_names: list[str]) -> dict[str, int]:
    """Return {feature: rank} where rank 1 = highest importance."""
    order = np.argsort(mean_abs)[::-1]
    return {feature_names[int(i)]: int(r + 1) for r, i in enumerate(order)}


def compute_adi_one_resample(
    X_clean: np.ndarray,
    X_leaky: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    cfg: dict,
    shap_sample: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Train clean + leaky classifiers on a bootstrap resample, return {feature: ADI}."""
    n = len(y)
    idx = rng.integers(0, n, size=n)          # bootstrap draw (with replacement)

    X_c_boot = X_clean[idx]
    X_l_boot = X_leaky[idx]
    y_boot   = y[idx]

    if len(np.unique(y_boot)) < 2:
        return {}                              # degenerate resample — skip

    # Scale inside the resample (fold-sealed analogue)
    sc_c = StandardScaler().fit(X_c_boot)
    sc_l = StandardScaler().fit(X_l_boot)
    Xc_s = sc_c.transform(X_c_boot)
    Xl_s = sc_l.transform(X_l_boot)

    # SHAP subsample (for speed)
    shap_idx = rng.integers(0, n, size=min(shap_sample, n))
    Xc_shap  = sc_c.transform(X_clean[shap_idx])
    Xl_shap  = sc_l.transform(X_leaky[shap_idx])

    rf_cfg  = cfg.get("random_forest", {})
    lr_cfg  = cfg.get("logistic_regression", {})
    xgb_cfg = cfg.get("xgboost", {})

    if clf_name == "RF":
        clf_c = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            random_state=int(rng.integers(0, 2**31)),
            n_jobs=-1,
        )
        clf_l = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            random_state=int(rng.integers(0, 2**31)),
            n_jobs=-1,
        )

    elif clf_name == "LR":
        clf_c = LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver="lbfgs",
            random_state=42,
        )
        clf_l = LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver="lbfgs",
            random_state=42,
        )

    elif clf_name == "XGB":
        if not HAS_XGB:
            return {}
        clf_c = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 200),
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            eval_metric="logloss",
            random_state=int(rng.integers(0, 2**31)),
            verbosity=0,
        )
        clf_l = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 200),
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            eval_metric="logloss",
            random_state=int(rng.integers(0, 2**31)),
            verbosity=0,
        )
    else:
        return {}

    clf_c.fit(Xc_s, y_boot)
    clf_l.fit(Xl_s, y_boot)

    ma_clean = _shap_mean_abs(clf_c, Xc_shap, clf_name)
    ma_leaky = _shap_mean_abs(clf_l, Xl_shap, clf_name)

    ranks_c = feature_ranks(ma_clean, CLEAN_FEATURES)
    ranks_l = feature_ranks(ma_leaky, LEAKY_FEATURES)

    adi = {}
    for feat in SHARED_FEATURES:
        r_c = ranks_c.get(feat)
        r_l = ranks_l.get(feat)
        if r_c is not None and r_l is not None:
            adi[feat] = r_c - r_l
    return adi


def compute_observed_adi(
    X_clean: np.ndarray,
    X_leaky: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    cfg: dict,
    shap_sample: int,
    seed: int,
) -> dict[str, int]:
    """ADI on the full (non-bootstrapped) dataset."""
    rng = np.random.default_rng(seed)
    sc_c = StandardScaler().fit(X_clean)
    sc_l = StandardScaler().fit(X_leaky)

    rf_cfg  = cfg.get("random_forest", {})
    lr_cfg  = cfg.get("logistic_regression", {})
    xgb_cfg = cfg.get("xgboost", {})

    if clf_name == "RF":
        clf_c = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            random_state=seed, n_jobs=-1)
        clf_l = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            random_state=seed, n_jobs=-1)
    elif clf_name == "LR":
        clf_c = LogisticRegression(C=lr_cfg.get("C", 1.0),
                                   max_iter=lr_cfg.get("max_iter", 1000),
                                   solver="lbfgs", random_state=42)
        clf_l = LogisticRegression(C=lr_cfg.get("C", 1.0),
                                   max_iter=lr_cfg.get("max_iter", 1000),
                                   solver="lbfgs", random_state=42)
    elif clf_name == "XGB":
        if not HAS_XGB:
            return {}
        clf_c = XGBClassifier(n_estimators=xgb_cfg.get("n_estimators", 200),
                              max_depth=xgb_cfg.get("max_depth", 4),
                              learning_rate=xgb_cfg.get("learning_rate", 0.05),
                              eval_metric="logloss", random_state=seed, verbosity=0)
        clf_l = XGBClassifier(n_estimators=xgb_cfg.get("n_estimators", 200),
                              max_depth=xgb_cfg.get("max_depth", 4),
                              learning_rate=xgb_cfg.get("learning_rate", 0.05),
                              eval_metric="logloss", random_state=seed, verbosity=0)
    else:
        return {}

    Xc_s = sc_c.transform(X_clean)
    Xl_s = sc_l.transform(X_leaky)
    clf_c.fit(Xc_s, y)
    clf_l.fit(Xl_s, y)

    shap_idx = rng.integers(0, len(y), size=min(shap_sample, len(y)))
    ma_clean = _shap_mean_abs(clf_c, sc_c.transform(X_clean[shap_idx]), clf_name)
    ma_leaky = _shap_mean_abs(clf_l, sc_l.transform(X_leaky[shap_idx]), clf_name)

    ranks_c = feature_ranks(ma_clean, CLEAN_FEATURES)
    ranks_l = feature_ranks(ma_leaky, LEAKY_FEATURES)

    adi = {}
    for feat in SHARED_FEATURES:
        r_c = ranks_c.get(feat)
        r_l = ranks_l.get(feat)
        if r_c is not None and r_l is not None:
            adi[feat] = r_c - r_l
    return adi


def aggregate_boot(boot_records: list[dict]) -> dict[str, dict]:
    """Aggregate per-bootstrap ADI dicts into summary statistics per feature."""
    from collections import defaultdict
    collector: dict[str, list[int]] = defaultdict(list)
    for record in boot_records:
        for feat, val in record.items():
            collector[feat].append(val)

    out = {}
    for feat, vals in collector.items():
        arr = np.array(vals, dtype=float)
        out[feat] = {
            "n_boot":    int(len(arr)),
            "mean":      float(np.mean(arr)),
            "sd":        float(np.std(arr, ddof=1)),
            "ci_lower":  float(np.percentile(arr, 2.5)),
            "ci_upper":  float(np.percentile(arr, 97.5)),
            "pct_neg":   float((arr < 0).mean() * 100),
            "pct_zero":  float((arr == 0).mean() * 100),
            "pct_pos":   float((arr > 0).mean() * 100),
        }
    return out


def plot_adi_ci(
    summary_by_clf: dict[str, dict[str, dict]],
    observed_by_clf: dict[str, dict[str, int]],
    features_order: list[str],
    out_path: Path,
):
    """Forest plot: one panel per classifier; horizontal bar = 95% CI; star = observed ADI."""
    classifiers = list(summary_by_clf.keys())
    n_clf = len(classifiers)

    fig, axes = plt.subplots(1, n_clf, figsize=(5 * n_clf, 6), sharey=True)
    if n_clf == 1:
        axes = [axes]

    colours = {"RF": "#2166ac", "LR": "#d6604d", "XGB": "#4dac26"}

    for ax, clf_name in zip(axes, classifiers):
        summary = summary_by_clf[clf_name]
        observed = observed_by_clf.get(clf_name, {})

        feat_list = [f for f in features_order if f in summary]
        n_feat = len(feat_list)
        y_pos = np.arange(n_feat)

        colour = colours.get(clf_name, "steelblue")

        for i, feat in enumerate(feat_list):
            s = summary[feat]
            obs = observed.get(feat, s["mean"])

            lo  = s["ci_lower"]
            hi  = s["ci_upper"]
            mid = s["mean"]

            ax.barh(i, hi - lo, left=lo, height=0.45,
                    color=colour, alpha=0.40)
            ax.scatter(mid, i, color=colour, s=40, zorder=3)
            ax.scatter(obs, i, marker="*", color="black", s=80, zorder=4,
                       label="observed" if i == 0 else "")

        ax.axvline(0, color="gray", linewidth=1.0, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_list, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("ADI  (rank$_\\mathrm{clean}$ − rank$_\\mathrm{leaky}$)", fontsize=9)
        ax.set_title(f"{clf_name}", fontsize=11)

        if clf_name == classifiers[0]:
            ax.legend(fontsize=8)

        ax.axvspan(ax.get_xlim()[0], 0, alpha=0.05, color="red",
                   label="suppressed (ADI < 0)")
        ax.axvspan(0, ax.get_xlim()[1], alpha=0.05, color="green",
                   label="promoted (ADI > 0)")

    fig.suptitle(
        "Bootstrap 95 % CIs for Attribution Distortion Index (ADI)\n"
        "Null scenario, TG4h-leaky vs clean pipeline",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CI forest plot -> {out_path}")


def print_latex_table(
    summary: dict[str, dict],
    observed: dict[str, int],
    clf_name: str,
    features_order: list[str],
):
    """Print a ready-to-paste LaTeX tabular fragment."""
    print(f"\n% === Bootstrap ADI table — {clf_name} ===")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\toprule")
    print(r"Feature & Obs.\ ADI & Boot.\ Mean & 95\,\% CI & $P(\text{ADI}<0)$\,\% \\")
    print(r"\midrule")
    for feat in features_order:
        if feat not in summary:
            continue
        s   = summary[feat]
        obs = observed.get(feat, float("nan"))
        direction = "↓" if obs < 0 else ("↑" if obs > 0 else "=")
        print(
            f"  {feat.upper():10s} & ${obs:+d}$ {direction} "
            f"& ${s['mean']:+.2f}$ "
            f"& $[{s['ci_lower']:+.1f},\\;{s['ci_upper']:+.1f}]$ "
            f"& ${s['pct_neg']:.0f}$ \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap CIs for ADI.")
    p.add_argument("--data",        required=True,
                   help="Null scenario CSV (data/paired_tcr_null_v1_seed2026.csv)")
    p.add_argument("--config",      required=True,
                   help="Model config YAML (config/model_config.yaml)")
    p.add_argument("--out",         default="results/tables/bootstrap_adi.csv")
    p.add_argument("--figdir",      default="results/figures")
    p.add_argument("--n_boot",      type=int, default=500,
                   help="Number of bootstrap resamples (default 500)")
    p.add_argument("--shap_sample", type=int, default=200,
                   help="SHAP subsample size per resample (default 200)")
    p.add_argument("--seed",        type=int, default=2026)
    p.add_argument("--classifiers", nargs="+", default=["RF", "LR", "XGB"],
                   choices=["RF", "LR", "XGB"],
                   help="Classifiers to evaluate (default: RF LR XGB)")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data, keep_default_na=False,
                     na_values=["NA", "NaN", "nan", ""])

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # Labels: Q1 TCR threshold from FULL dataset
    # (acceptable here because we are measuring SHAP rank stability, not AUROC)
    tcr = df["tcr"].values
    threshold = float(np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0)))
    y = (tcr <= threshold).astype(int)

    X_clean = df[CLEAN_FEATURES].values
    X_leaky = df[LEAKY_FEATURES].values

    print(f"Dataset: n={len(df)}, pos_rate={y.mean():.3f}")
    print(f"Bootstrap resamples: {args.n_boot}")
    print(f"SHAP subsample per resample: {args.shap_sample}")
    print(f"Classifiers: {args.classifiers}")
    print(f"Clean features ({len(CLEAN_FEATURES)}): {CLEAN_FEATURES}")
    print(f"Leaky features ({len(LEAKY_FEATURES)}): {LEAKY_FEATURES}")

    all_rows   : list[dict] = []
    summary_by_clf  : dict[str, dict] = {}
    observed_by_clf : dict[str, dict] = {}

    for clf_name in args.classifiers:
        if clf_name == "XGB" and not HAS_XGB:
            print(f"Skipping XGB — xgboost not installed.")
            continue

        print(f"\n{'='*60}")
        print(f"Classifier: {clf_name}")
        print(f"{'='*60}")

        print("  Computing observed ADI (full dataset)…")
        obs_adi = compute_observed_adi(
            X_clean, X_leaky, y, clf_name, cfg, args.shap_sample, args.seed
        )
        observed_by_clf[clf_name] = obs_adi
        print(f"  Observed ADI: {obs_adi}")

        rng = np.random.default_rng(args.seed)
        boot_records: list[dict] = []

        pbar = tqdm(
            range(args.n_boot),
            desc=f"Bootstrap {clf_name}",
            ncols=90,
            colour="cyan",
        )
        for b in pbar:
            record = compute_adi_one_resample(
                X_clean, X_leaky, y,
                clf_name, cfg,
                args.shap_sample, rng,
            )
            if record:
                boot_records.append(record)

            if (b + 1) % 50 == 0:
                pbar.set_postfix(valid=len(boot_records))

        print(f"  Valid bootstrap resamples: {len(boot_records)} / {args.n_boot}")

        summary = aggregate_boot(boot_records)
        summary_by_clf[clf_name] = summary

        for feat in SHARED_FEATURES:
            if feat not in summary:
                continue
            s = summary[feat]
            all_rows.append({
                "classifier":    clf_name,
                "feature":       feat,
                "observed_adi":  obs_adi.get(feat),
                "mean_boot_adi": round(s["mean"], 3),
                "sd_boot_adi":   round(s["sd"], 3),
                "ci_lower_95":   round(s["ci_lower"], 2),
                "ci_upper_95":   round(s["ci_upper"], 2),
                "pct_negative":  round(s["pct_neg"], 1),
                "pct_zero":      round(s["pct_zero"], 1),
                "pct_positive":  round(s["pct_pos"], 1),
                "n_boot_valid":  s["n_boot"],
            })

        print_latex_table(summary, obs_adi, clf_name, PAPER_FEATURES)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved full table -> {out_path}")

    if "RF" in summary_by_clf:
        rf_rows = [r for r in all_rows if r["classifier"] == "RF"
                   and r["feature"] in PAPER_FEATURES]
        summary_path = out_path.with_name("bootstrap_adi_summary.csv")
        pd.DataFrame(rf_rows).to_csv(summary_path, index=False)
        print(f"Saved RF summary -> {summary_path}")

    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_adi_ci(
        summary_by_clf,
        observed_by_clf,
        PAPER_FEATURES,
        fig_dir / "bootstrap_adi_ci.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
