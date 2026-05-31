"""Cross-classifier SHAP/ADI comparison."""
from __future__ import annotations
import argparse, sys, warnings
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
try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_clean_pipeline import CLEAN_FEATURES
warnings.filterwarnings("ignore")
LEAKY_FEATURES = CLEAN_FEATURES + ["tg4h"]
SAMPLE_SIZE = 400

def get_shap_rf(model, X):
    Xs = X[:SAMPLE_SIZE]
    exp = shap.TreeExplainer(model)
    sv = exp.shap_values(Xs)
    if isinstance(sv, list): return sv[1]
    if sv.ndim == 3: return sv[:, :, 1]
    return sv

def get_shap_xgb(model, X):
    Xs = X[:SAMPLE_SIZE].astype(np.float32)
    dmat = xgb.DMatrix(Xs)
    c = model.get_booster().predict(dmat, pred_contribs=True)
    return c[:, :-1]

def get_shap_lr(model, X):
    Xs = X[:SAMPLE_SIZE]
    exp = shap.LinearExplainer(model, Xs,
                               feature_perturbation="correlation_dependent")
    sv = exp.shap_values(Xs)
    if isinstance(sv, list): return sv[1]
    if sv.ndim == 3: return sv[:, :, 1]
    return sv

def get_shap(model, X, name):
    if not HAS_SHAP:
        return np.zeros((min(SAMPLE_SIZE, len(X)), X.shape[1]))
    if "Forest" in name: return get_shap_rf(model, X)
    if "XGB" in name: return get_shap_xgb(model, X)
    return get_shap_lr(model, X)

def rank_feats(sv, names):
    m = np.abs(sv).mean(axis=0)
    o = np.argsort(m)[::-1]
    return {names[int(i)]: int(r+1) for r, i in enumerate(o)}

def run_clf(name, clf, Xc, Xl, y):
    import sklearn.base
    c = sklearn.base.clone(clf)
    sc = StandardScaler()
    Xcs = sc.fit_transform(Xc)
    c.fit(Xcs, y)
    sv_c = get_shap(c, Xcs, name)
    rc = rank_feats(sv_c, CLEAN_FEATURES)
    mc = {f: float(np.abs(sv_c[:, i]).mean()) for i, f in enumerate(CLEAN_FEATURES)}
    l = sklearn.base.clone(clf)
    sl = StandardScaler()
    Xls = sl.fit_transform(Xl)
    l.fit(Xls, y)
    sv_l = get_shap(l, Xls, name)
    rl = rank_feats(sv_l, LEAKY_FEATURES)
    ml = {f: float(np.abs(sv_l[:, i]).mean()) for i, f in enumerate(LEAKY_FEATURES)}
    adi = {f: rc[f] - rl[f] for f in rc if f in rl}
    return {
        "classifier": name,
        "tg4h_rank_leaky": rl.get("tg4h"),
        "tg4h_mean_abs_shap": round(ml.get("tg4h", 0.0), 5),
        "wbv_rank_clean": rc.get("wbv"),
        "wbv_rank_leaky": rl.get("wbv"),
        "wbv_shap_clean": round(mc.get("wbv", 0.0), 5),
        "wbv_shap_leaky": round(ml.get("wbv", 0.0), 5),
        "adi_wbv": adi.get("wbv"),
        "adi_bmi": adi.get("bmi"),
        "adi_age": adi.get("age"),
        "adi_hdl": adi.get("hdl"),
        "rc": rc, "rl": rl, "mc": mc, "ml": ml,
    }

def make_charts(rows, figdir):
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, rows):
        feats = LEAKY_FEATURES
        ms = [r["ml"].get(f, 0) for f in feats]
        o = np.argsort(ms)[::-1]
        sf = [feats[i] for i in o]
        sm = [ms[i] for i in o]
        bc = ["crimson" if f == "tg4h" else "steelblue" for f in sf]
        ax.barh(sf, sm, color=bc)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(r["classifier"] + "\n(Leaky — TG4h in)")
        ax.invert_yaxis()
    plt.suptitle("Cross-Classifier Attribution Under Definitional Leakage"
                 "\n(crimson=TG4h; n=1500 null scenario)", fontsize=12)
    plt.tight_layout()
    p = Path(figdir) / "cross_classifier_shap_leaky.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved leaky chart ->", p)
    fig2, axes2 = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1: axes2 = [axes2]
    for ax, r in zip(axes2, rows):
        feats = CLEAN_FEATURES
        ms = [r["mc"].get(f, 0) for f in feats]
        o = np.argsort(ms)[::-1]
        sf = [feats[i] for i in o]
        sm = [ms[i] for i in o]
        ax.barh(sf, sm, color="steelblue")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(r["classifier"] + "\n(Clean — no leak)")
        ax.invert_yaxis()
    plt.suptitle("Cross-Classifier Attribution: Clean Pipeline"
                 "\n(WBV near-zero in null scenario)", fontsize=12)
    plt.tight_layout()
    p2 = Path(figdir) / "cross_classifier_shap_clean.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved clean chart ->", p2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="results/tables/cross_classifier_shap.csv")
    p.add_argument("--figdir", default="results/figures")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    df = pd.read_csv(args.data, keep_default_na=False,
                     na_values=["NA","NaN","nan",""])
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcr = df["tcr"].values
    thr = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= thr).astype(int)
    print(f"Prevalence: {y.mean():.3f}  n={len(y)}")
    Xc = df[CLEAN_FEATURES].values
    Xl = df[LEAKY_FEATURES].values
    clfs = [
        ("RandomForest",
         RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)),
        ("LogisticRegression",
         LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                            random_state=args.seed)),
    ]
    if HAS_XGB:
        clfs.append(("XGBoost",
                     XGBClassifier(n_estimators=200, random_state=args.seed,
                                   base_score=0.5, eval_metric="logloss",
                                   verbosity=0)))
    rows = []
    for name, clf in clfs:
        print(f"\n── {name} ──")
        res = run_clf(name, clf, Xc, Xl, y)
        rows.append(res)
        print(f"  TG4h rank leaky:    {res['tg4h_rank_leaky']}")
        print(f"  TG4h mean|SHAP|:    {res['tg4h_mean_abs_shap']:.5f}")
        print(f"  WBV rank c->l:      {res['wbv_rank_clean']}->{res['wbv_rank_leaky']}")
        print(f"  ADI wbv={res['adi_wbv']} bmi={res['adi_bmi']}"
              f" age={res['adi_age']} hdl={res['adi_hdl']}")
    cols = ["classifier","tg4h_rank_leaky","tg4h_mean_abs_shap",
            "wbv_rank_clean","wbv_rank_leaky",
            "wbv_shap_clean","wbv_shap_leaky",
            "adi_wbv","adi_bmi","adi_age","adi_hdl"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[cols].to_csv(out, index=False)
    print(f"\nSaved summary -> {out}")
    rr = []
    for r in rows:
        for feat in CLEAN_FEATURES:
            rr.append({"classifier": r["classifier"], "feature": feat,
                       "rank_clean": r["rc"].get(feat),
                       "rank_leaky": r["rl"].get(feat),
                       "adi": r["rc"].get(feat,99)-r["rl"].get(feat,99)})
        rr.append({"classifier": r["classifier"], "feature": "tg4h",
                   "rank_clean": None, "rank_leaky": r["rl"].get("tg4h"),
                   "adi": None})
    rcsv = out.parent / "cross_classifier_shap_ranks.csv"
    pd.DataFrame(rr).to_csv(rcsv, index=False)
    print(f"Saved ranks   -> {rcsv}")
    make_charts(rows, args.figdir)
    return rows

if __name__ == "__main__":
    main()
