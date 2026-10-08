"""S1 re-computation: clean (nested_cv) and TG4h-leaky AUROC per scenario x seed.
Leaky pipeline = identical fold-sealed protocol (winsorise, scale, Q1 threshold
from training fold, SMOTE in training fold) with TG4h added as a tenth feature."""
import sys, yaml, numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import CLEAN_FEATURES, build_models, nested_cv, fold_sealed_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import contextlib, io
SC = {"null":"config/generator_null.yaml","weak_signal":"config/generator_weak_signal.yaml",
      "moderate_signal":"config/generator_moderate_signal.yaml","wbv_positive":"config/generator_wbv_positive.yaml"}
cfgm = yaml.safe_load(open("config/model_config.yaml"))
def leaky_auc(df, model, seed):
    X = df[CLEAN_FEATURES+["tg4h"]].values; tcr = df["tcr"].values
    ystrat = (tcr <= np.percentile(tcr,25)).astype(int)
    aucs=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,ystrat):
        Xtr,Xte,ytr,thr = fold_sealed_preprocess(X[tr],X[te],None,tcr[tr],cfgm)
        yte=(tcr[te]<=thr).astype(int)
        m=clone(model).fit(Xtr,ytr); aucs.append(roc_auc_score(yte,m.predict_proba(Xte)[:,1]))
    return float(np.mean(aucs))
def one(sc, seed, mname):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        cfg = yaml.safe_load(open(SC[sc])); df = generate(cfg, seed=seed, n=1500)
        model = build_models(cfgm["models"])[mname]
        c = nested_cv(df, mname, model, cfgm, outer_folds=5, seed=seed)["AUROC_mean"]
        l = leaky_auc(df, model, seed)
    return dict(scenario=sc, seed=seed, model=mname, clean=c, leaky=l)
if __name__=="__main__":
    mname=sys.argv[1]; nseeds=int(sys.argv[2]); out=sys.argv[3]
    recs=Parallel(n_jobs=int(sys.argv[4]) if len(sys.argv)>4 else 2)(delayed(one)(sc,s,mname) for sc in SC for s in range(1,nseeds+1))
    d=pd.DataFrame(recs); d.to_csv(out,index=False)
    g=d.groupby("scenario")[["clean","leaky"]].agg(["mean","std"]).round(4); print(g)
