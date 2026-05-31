"""Unified SHAP + ADI analysis for tab:shap_adi and tab:cross_classifier_shap."""

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn.base

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — XGB classifier will be skipped.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed — SHAP analysis will be skipped.")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_clean_pipeline import CLEAN_FEATURES

warnings.filterwarnings("ignore")

LEAKY_FEATURES = CLEAN_FEATURES + ["tg4h"]


def get_shap_values(model, X: np.ndarray, model_name: str, sample_size: int) -> np.ndarray:
    """Compute SHAP values for positive class; consistent across both tables."""
    if not HAS_SHAP:
        return np.zeros((min(sample_size, len(X)), X.shape[1]))

    Xs = X[:sample_size]

    if "Forest" in model_name:
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(Xs)
    elif "XGB" in model_name:
        dmat = xgb.DMatrix(Xs.astype(np.float32))
        c = model.get_booster().predict(dmat, pred_contribs=True)
        return c[:, :-1]   # drop bias column
    else:
        exp = shap.LinearExplainer(model, Xs, feature_perturbation="correlation_dependent")
        sv = exp.shap_values(Xs)

    if isinstance(sv, list):
        return sv[1]
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def rank_features(shap_values: np.ndarray, feature_names: list[str]) -> dict[str, int]:
    """Return {feature: rank} sorted by mean |SHAP| descending (rank 1 = most important)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return {feature_names[int(i)]: int(r + 1) for r, i in enumerate(order)}


def mean_shap(shap_values: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    return {f: float(np.abs(shap_values[:, i]).mean()) for i, f in enumerate(feature_names)}


def run_one_classifier(
    name: str,
    clf,
    Xc: np.ndarray,
    Xl: np.ndarray,
    y: np.ndarray,
    sample_size: int,
) -> dict:
    """Train clean+leaky models, compute SHAP, return ranks and ADI."""
    clf_c = sklearn.base.clone(clf)
    sc = StandardScaler()
    Xcs = sc.fit_transform(Xc)
    clf_c.fit(Xcs, y)
    sv_c = get_shap_values(clf_c, Xcs, name, sample_size)
    rc = rank_features(sv_c, CLEAN_FEATURES)
    mc = mean_shap(sv_c, CLEAN_FEATURES)

    clf_l = sklearn.base.clone(clf)
    sl = StandardScaler()
    Xls = sl.fit_transform(Xl)
    clf_l.fit(Xls, y)
    sv_l = get_shap_values(clf_l, Xls, name, sample_size)
    rl = rank_features(sv_l, LEAKY_FEATURES)
    ml = mean_shap(sv_l, LEAKY_FEATURES)

    # ADI = rank_clean - rank_leaky (negative = suppressed, positive = promoted)
    adi = {f: rc[f] - rl[f] for f in CLEAN_FEATURES if f in rl}

    return {
        "classifier": name,
        "rc": rc, "rl": rl, "mc": mc, "ml": ml,
        "adi": adi,
        "sv_c": sv_c, "sv_l": sv_l,
        "Xcs": Xcs, "Xls": Xls,
    }


def print_shap_adi_table(rf_result: dict):
    """Print the tab:shap_adi table (RF only) for manual LaTeX update."""
    rc = rf_result["rc"]
    rl = rf_result["rl"]
    adi = rf_result["adi"]
    mc = rf_result["mc"]

    print("\n" + "="*65)
    print("tab:shap_adi — RF Attribution Distortion Index")
    print("(use these values to update the LaTeX table)")
    print("="*65)
    print(f"{'Feature':<12} {'Rank(clean)':>11} {'Rank(leaky)':>11} {'ADI':>6}  {'Mean|SHAP|(clean)':>18}")
    print("-"*65)

    print(f"{'TG4h':<12} {'--- (absent)':>11} {'1':>11} {'leaky-only':>6}")
    print()

    for feat in sorted(CLEAN_FEATURES, key=lambda f: rc[f]):
        r_clean = rc[feat]
        r_leaky = rl.get(feat, "?")
        a = adi.get(feat, "?")
        m = mc.get(feat, 0.0)
        print(f"{feat:<12} {r_clean:>11} {r_leaky:>11} {a:>+6}  {m:>18.4f}")


def print_cross_classifier_table(results: list[dict]):
    """Print the cross-classifier ADI table for manual LaTeX update."""
    print("\n" + "="*90)
    print("tab:cross_classifier_shap — Cross-Classifier ADI")
    print("Columns: Feature | RF Rank(L) | RF ADI | LR Rank(L) | LR ADI | XGB Rank(L) | XGB ADI")
    print("="*90)

    all_feats = ["tg4h"] + CLEAN_FEATURES
    for feat in all_feats:
        parts = [f"{feat:<12}"]
        for res in results:
            if feat == "tg4h":
                rl_rank = res["rl"].get("tg4h", "?")
                parts.append(f"  {rl_rank:>4}  {'leaky-only':>10}")
            else:
                rl_rank = res["rl"].get(feat, "?")
                adi_val = res["adi"].get(feat, "?")
                adi_str = f"{adi_val:+d}" if isinstance(adi_val, int) else str(adi_val)
                parts.append(f"  {rl_rank:>4}  {adi_str:>6}")
        print("".join(parts))


def save_figures(results: list[dict], figdir: Path, sample_size: int):
    figdir.mkdir(parents=True, exist_ok=True)

    rf = results[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mc = rf["mc"]
    feats_c = sorted(CLEAN_FEATURES, key=lambda f: -mc[f])
    axes[0].barh(feats_c, [mc[f] for f in feats_c], color="steelblue")
    axes[0].set_xlabel("Mean |SHAP|"); axes[0].set_title("RF — Clean pipeline")
    axes[0].invert_yaxis()
    ml = rf["ml"]
    feats_l = sorted(LEAKY_FEATURES, key=lambda f: -ml[f])
    colors = ["crimson" if f == "tg4h" else "steelblue" for f in feats_l]
    axes[1].barh(feats_l, [ml[f] for f in feats_l], color=colors)
    axes[1].set_xlabel("Mean |SHAP|"); axes[1].set_title("RF — Leaky (TG4h included)")
    axes[1].invert_yaxis()
    plt.suptitle("Feature Attribution: Clean vs Leaky RF", fontsize=13)
    plt.tight_layout()
    p = figdir / "shap_comparison_bars.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {p}")

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        ml = r["ml"]
        feats = sorted(LEAKY_FEATURES, key=lambda f: -ml[f])
        bc = ["crimson" if f == "tg4h" else "steelblue" for f in feats]
        ax.barh(feats, [ml[f] for f in feats], color=bc)
        ax.set_xlabel("Mean |SHAP|"); ax.set_title(r["classifier"] + "\n(Leaky)")
        ax.invert_yaxis()
    plt.suptitle("Cross-Classifier Attribution (Leaky)", fontsize=12)
    plt.tight_layout()
    p2 = figdir / "cross_classifier_shap_leaky.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {p2}")

    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        mc = r["mc"]
        feats = sorted(CLEAN_FEATURES, key=lambda f: -mc[f])
        ax.barh(feats, [mc[f] for f in feats], color="steelblue")
        ax.set_xlabel("Mean |SHAP|"); ax.set_title(r["classifier"] + "\n(Clean)")
        ax.invert_yaxis()
    plt.suptitle("Cross-Classifier Attribution (Clean)", fontsize=12)
    plt.tight_layout()
    p3 = figdir / "cross_classifier_shap_clean.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {p3}")


def save_csvs(results: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = results[0]
    rows = []
    rows.append({"feature": "tg4h", "rank_clean": None, "rank_leaky": rf["rl"]["tg4h"],
                 "mean_shap_clean": None, "adi": None, "note": "leaky-only antecedent"})
    for feat in sorted(CLEAN_FEATURES, key=lambda f: rf["rc"][f]):
        rows.append({
            "feature": feat,
            "rank_clean": rf["rc"][feat],
            "rank_leaky": rf["rl"].get(feat),
            "mean_shap_clean": round(rf["mc"][feat], 5),
            "adi": rf["adi"].get(feat),
            "note": "",
        })
    p1 = out_dir / "unified_shap_adi_rf.csv"
    pd.DataFrame(rows).to_csv(p1, index=False)
    print(f"  Saved -> {p1}")

    rows2 = []
    for res in results:
        for feat in CLEAN_FEATURES:
            rows2.append({
                "classifier": res["classifier"],
                "feature": feat,
                "rank_leaky": res["rl"].get(feat),
                "adi": res["adi"].get(feat),
                "mean_shap_leaky": round(res["ml"].get(feat, 0.0), 5),
            })
        rows2.append({
            "classifier": res["classifier"], "feature": "tg4h",
            "rank_leaky": res["rl"].get("tg4h"),
            "adi": None,
            "mean_shap_leaky": round(res["ml"].get("tg4h", 0.0), 5),
        })
    p2 = out_dir / "unified_shap_cross_clf.csv"
    pd.DataFrame(rows2).to_csv(p2, index=False)
    print(f"  Saved -> {p2}")


def main():
    p = argparse.ArgumentParser(description="Unified SHAP/ADI for tab:shap_adi + tab:cross_classifier_shap")
    p.add_argument("--data",        default="data/paired_tcr_null_v1_seed2026.csv")
    p.add_argument("--config",      default="config/model_config.yaml")
    p.add_argument("--out_dir",     default="results/tables")
    p.add_argument("--figdir",      default="results/figures")
    p.add_argument("--seed",        type=int, default=2026,
                   help="Random seed for all classifiers (default: 2026)")
    p.add_argument("--sample_size", type=int, default=500,
                   help="Number of rows used for SHAP computation (default: 500)")
    args = p.parse_args()

    print(f"\n{'='*65}")
    print(f" run_unified_shap.py")
    print(f" seed={args.seed}  sample_size={args.sample_size}")
    print(f"{'='*65}\n")

    df = pd.read_csv(args.data, keep_default_na=False, na_values=["NA","NaN","nan",""])
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcr = df["tcr"].values
    thr = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y   = (tcr <= thr).astype(int)
    print(f"Dataset: n={len(df)}, prevalence={y.mean():.3f}, threshold={thr:.2f}")

    Xc = df[CLEAN_FEATURES].values
    Xl = df[LEAKY_FEATURES].values

    classifiers = [
        ("RandomForest",
         RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)),
        ("LogisticRegression",
         LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                            random_state=args.seed)),
    ]
    if HAS_XGB:
        classifiers.append(
            ("XGBoost",
             XGBClassifier(n_estimators=200, random_state=args.seed,
                           base_score=0.5, eval_metric="logloss", verbosity=0))
        )
    else:
        print("XGBoost not available — skipping XGB.\n")

    results = []
    for name, clf in classifiers:
        print(f"\n── {name} ──")
        res = run_one_classifier(name, clf, Xc, Xl, y, args.sample_size)
        results.append(res)
        print(f"  TG4h rank (leaky): {res['rl'].get('tg4h')}")
        print(f"  TG4h mean|SHAP|:   {res['ml'].get('tg4h',0):.5f}")
        print(f"  Key ADIs: HDL={res['adi'].get('hdl'):+d}  Age={res['adi'].get('age'):+d}  "
              f"BMI={res['adi'].get('bmi'):+d}  TG0h={res['adi'].get('tg0h'):+d}")

    print_shap_adi_table(results[0])
    print_cross_classifier_table(results)

    print("\n── Saving outputs ──")
    save_csvs(results, Path(args.out_dir))
    save_figures(results, Path(args.figdir), args.sample_size)

    print("\n✓ Done. Use the printed ADI values above to update main.tex tables.")
    print("  Files written to results/tables/unified_shap_*.csv")


if __name__ == "__main__":
    main()
