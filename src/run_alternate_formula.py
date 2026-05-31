"""Negative control: alternate paired formula ATC = TG0h - TG4h (absolute change)."""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_clean_pipeline import CLEAN_FEATURES
warnings.filterwarnings("ignore")

ATC_CLEAN = CLEAN_FEATURES              # includes tg0h (algebraic partner)
ATC_LEAKY = CLEAN_FEATURES + ["tg4h"]  # adds the leaky component


def cv_auroc(X, y, clf, n_splits=5, seed=42):
    """Stratified k-fold AUROC; returns (mean, sd) across folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        import sklearn.base
        m = sklearn.base.clone(clf)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        scores.append(roc_auc_score(y[te], p))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def run_one_seed(df, seed, n_splits=5):
    """Run clean vs leaky benchmark for ATC formula on one seed."""
    atc = df["tg0h"].values - df["tg4h"].values
    thr = np.percentile(atc, 75)
    y = (atc >= thr).astype(int)

    Xc = df[ATC_CLEAN].values
    Xl = df[ATC_LEAKY].values

    clfs = [
        ("LR",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                   random_state=seed)),
        ("RF",  RandomForestClassifier(n_estimators=200, random_state=seed,
                                       n_jobs=-1)),
    ]
    if HAS_XGB:
        clfs.append(("XGB", XGBClassifier(n_estimators=200, random_state=seed,
                                          base_score=0.5,
                                          eval_metric="logloss", verbosity=0)))
    rows = []
    for name, clf in clfs:
        ac, sc = cv_auroc(Xc, y, clf, n_splits, seed)
        al, sl = cv_auroc(Xl, y, clf, n_splits, seed)
        rows.append({"seed": seed, "classifier": name,
                     "clean_auroc": ac, "clean_sd": sc,
                     "leaky_auroc": al, "leaky_sd": sl,
                     "delta_auc": al - ac})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True)
    p.add_argument("--config",   required=True)
    p.add_argument("--out",      default="results/tables/alternate_formula_benchmark.csv")
    p.add_argument("--seeds",    type=int, default=30)
    p.add_argument("--n_splits", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False,
                     na_values=["NA","NaN","nan",""])
    print(f"Data: {len(df)} rows")
    print(f"Formula: ATC = TG0h - TG4h  (absolute change)")
    print(f"Leaky predictor: TG4h  |  Clean: {ATC_CLEAN}")
    print(f"Running {args.seeds} seeds x {args.n_splits}-fold CV...")

    all_rows = []
    for s in range(args.seeds):
        seed = 1000 + s
        all_rows.extend(run_one_seed(df, seed, args.n_splits))
        if (s + 1) % 10 == 0:
            print(f"  Completed seed {s+1}/{args.seeds}")

    results = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nFull results -> {out}")

    print("\n=== ATC Formula — Alternate Paired Formula Benchmark ===")
    print(f"{'Classifier':20s} {'Clean AUROC':>14s} {'Leaky AUROC':>14s} {'dAUC':>10s}")
    print("-" * 62)
    for clf in results["classifier"].unique():
        sub = results[results["classifier"] == clf]
        mc  = sub["clean_auroc"].mean()
        sc  = sub["clean_auroc"].std()
        ml  = sub["leaky_auroc"].mean()
        sl  = sub["leaky_auroc"].std()
        md  = sub["delta_auc"].mean()
        print(f"{clf:20s}  {mc:.4f}+/-{sc:.4f}   {ml:.4f}+/-{sl:.4f}   +{md:.4f}")

    summary = (results.groupby("classifier")
               .agg(clean_auroc_mean=("clean_auroc","mean"),
                    clean_auroc_sd=("clean_auroc","std"),
                    leaky_auroc_mean=("leaky_auroc","mean"),
                    leaky_auroc_sd=("leaky_auroc","std"),
                    delta_auc_mean=("delta_auc","mean"),
                    delta_auc_sd=("delta_auc","std"))
               .reset_index())
    sout = out.parent / "alternate_formula_summary.csv"
    summary.to_csv(sout, index=False)
    print(f"\nSummary -> {sout}")
    return summary


if __name__ == "__main__":
    main()
