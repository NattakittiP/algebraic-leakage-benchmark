"""Literal test of Proposition 2: additive Gaussian measurement error on TG4h.

For each seed the null dataset is generated at the standard parameters, TG4h is
computed exactly, and TG4h_obs = TG4h + eps, eps ~ N(0, sigma_eps^2) is used as the
tenth feature of the leaky pipeline (fold-sealed protocol, LR).  sigma_eps is
specified in mg/dL as multiples of 18.6 (the nominal TCR SD, for comparability
with S3); the propagated error in reconstructed TCR units is
sigma_eta = 100 * sigma_eps * sqrt(E[TG0h^-2]).
Usage: python src/run_measurement_noise_prop2.py --seeds 20 --n 1500 --out results/tables/measurement_noise_prop2.csv
"""
import sys, argparse, contextlib, io, yaml, numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import CLEAN_FEATURES, build_models, fold_sealed_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
MULTS = [0.5, 1.0, 1.5, 2.0, 3.0]; D0, VD = 1.94, 0.767
def run(seed, n, cfgg, cfgm):
    with contextlib.redirect_stderr(io.StringIO()):
        df = generate(cfgg, seed=seed, n=n)
    model = build_models(cfgm["models"])["LogisticRegression"]
    tcr = df["tcr"].values; ystrat = (tcr <= np.percentile(tcr, 25)).astype(int)
    rng = np.random.default_rng(10_000 + seed); out = []
    k = 100*np.sqrt(np.mean(1/df["tg0h"].values**2))
    for m in MULTS:
        se = m*18.6
        X = np.column_stack([df[CLEAN_FEATURES].values, df["tg4h"].values + rng.normal(0, se, n)])
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, ystrat):
            Xtr, Xte, ytr, thr = fold_sealed_preprocess(X[tr], X[te], None, tcr[tr], cfgm)
            yte = (tcr[te] <= thr).astype(int)
            aucs.append(roc_auc_score(yte, clone(model).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
        r = k*se/18.6
        out.append(dict(seed=seed, mult=m, sigma_eps=se, sigma_eta_over_sigma_tcr=r,
                        oracle_ref=stats.norm.cdf(D0/np.sqrt(1+2*r**2/VD)), leaky_auroc=np.mean(aucs)))
    return out
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n", type=int, default=1500); p.add_argument("--out", required=True)
    a = p.parse_args()
    cfgg = yaml.safe_load(open("config/generator_null.yaml")); cfgm = yaml.safe_load(open("config/model_config.yaml"))
    recs = sum(Parallel(n_jobs=2)(delayed(run)(s, a.n, cfgg, cfgm) for s in range(1, a.seeds+1)), [])
    d = pd.DataFrame(recs); d.to_csv(a.out, index=False)
    print(d.groupby("mult").agg(sigma_eps=("sigma_eps","first"), ratio=("sigma_eta_over_sigma_tcr","mean"),
          oracle=("oracle_ref","mean"), auc_mean=("leaky_auroc","mean"), auc_sd=("leaky_auroc","std")).round(4))
