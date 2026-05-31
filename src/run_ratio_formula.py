"""Ratio-type formula benchmark (TGR = TG4h / TG0h), analogous to UACR."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_clean_pipeline import CLEAN_FEATURES

RATIO_CLEAN = CLEAN_FEATURES                     # includes tg0h (A-component)
RATIO_LEAKY = CLEAN_FEATURES + ["tg4h"]          # adds B-component → algebraic identity


def cv_auroc(X: np.ndarray, y: np.ndarray, clf, n_splits: int, seed: int):
    """Stratified k-fold AUROC; returns (mean, sd) across folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        m = clone(clf)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m.fit(Xtr, y[tr])
        prob = m.predict_proba(Xte)[:, 1]
        scores.append(roc_auc_score(y[te], prob))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def run_one_seed(df: pd.DataFrame, seed: int, n_splits: int = 5) -> list[dict]:
    """Run clean vs leaky benchmark for ratio formula on one seed."""
    ratio = df["tg4h"].values / df["tg0h"].values
    thr = np.percentile(ratio, 75)
    y = (ratio >= thr).astype(int)

    Xc = df[RATIO_CLEAN].values.astype(float)
    Xl = df[RATIO_LEAKY].values.astype(float)

    classifiers = [
        ("LR",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                   random_state=seed)),
        ("RF",  RandomForestClassifier(n_estimators=200, random_state=seed,
                                       n_jobs=-1)),
    ]
    if HAS_XGB:
        classifiers.append(
            ("XGB", XGBClassifier(n_estimators=200, random_state=seed,
                                  base_score=0.5, eval_metric="logloss",
                                  verbosity=0))
        )

    rows = []
    for name, clf in classifiers:
        ac, sc_ = cv_auroc(Xc, y, clf, n_splits, seed)
        al, sl  = cv_auroc(Xl, y, clf, n_splits, seed)
        rows.append({
            "seed":        seed,
            "classifier":  name,
            "clean_auroc": ac,
            "clean_sd":    sc_,
            "leaky_auroc": al,
            "leaky_sd":    sl,
            "delta_auc":   al - ac,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True,
                   help="Path to paired TCR null CSV (same synthetic cohort)")
    p.add_argument("--out",      default="results/tables/ratio_formula_benchmark.csv")
    p.add_argument("--seeds",    type=int, default=30)
    p.add_argument("--n_splits", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False,
                     na_values=["NA", "NaN", "nan", ""])
    print(f"Data loaded: {len(df)} rows")
    print(f"Formula:     TGR = TG4h / TG0h  (ratio-type, UACR-analogous)")
    print(f"Leaky feat:  tg4h  (reconstructs Y given tg0h algebraically)")
    print(f"Running {args.seeds} seeds × {args.n_splits}-fold CV …\n")

    all_rows = []
    for s in range(args.seeds):
        seed = 2000 + s
        all_rows.extend(run_one_seed(df, seed, args.n_splits))
        if (s + 1) % 10 == 0:
            print(f"  Completed {s+1}/{args.seeds} seeds")

    results = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nFull results → {out}")

    print("\n=== Ratio Formula (TGR = TG4h / TG0h) Benchmark ===")
    print(f"{'Classifier':8s}  {'Clean AUROC':>14s}  {'Leaky AUROC':>14s}  {'ΔAUC':>8s}")
    print("-" * 52)
    summary_rows = []
    for clf_name in results["classifier"].unique():
        sub = results[results["classifier"] == clf_name]
        mc = sub["clean_auroc"].mean()
        sc_ = sub["clean_auroc"].std()
        ml = sub["leaky_auroc"].mean()
        sl = sub["leaky_auroc"].std()
        md = sub["delta_auc"].mean()
        sd_d = sub["delta_auc"].std()
        print(f"{clf_name:8s}   {mc:.4f}±{sc_:.4f}   {ml:.4f}±{sl:.4f}   +{md:.4f}")
        summary_rows.append({
            "classifier":         clf_name,
            "clean_auroc_mean":   round(mc,  4),
            "clean_auroc_sd":     round(sc_, 4),
            "leaky_auroc_mean":   round(ml,  4),
            "leaky_auroc_sd":     round(sl,  4),
            "delta_auc_mean":     round(md,  4),
            "delta_auc_sd":       round(sd_d, 4),
        })

    summary = pd.DataFrame(summary_rows)
    sout = out.parent / "ratio_formula_summary.csv"
    summary.to_csv(sout, index=False)
    print(f"\nSummary → {sout}")
    return summary


if __name__ == "__main__":
    main()
