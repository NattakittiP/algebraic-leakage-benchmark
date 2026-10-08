"""Table S9 Panel C check: 30 repeated fold-sealed 5-fold CVs of the clean pipeline
on the single primary null dataset (seed 2026, n = 1,500), four classifiers."""
import sys, yaml, contextlib, io, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_synthetic_data import generate
from src.run_clean_pipeline import build_models, nested_cv
CG = yaml.safe_load(open("config/generator_null.yaml")); CM = yaml.safe_load(open("config/model_config.yaml"))
with contextlib.redirect_stderr(io.StringIO()):
    DF = generate(CG, seed=2026, n=1500)
def task(a):
    m, s = a
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        r = nested_cv(DF, m, build_models(CM["models"])[m], CM, outer_folds=5, seed=s)
    return dict(model=m, cv_seed=s, auroc=r["AUROC_mean"])
if __name__ == "__main__":
    out = sys.argv[1]
    jobs = [(m, s) for m in ["LogisticRegression", "RandomForest", "SVM", "XGBoost"] for s in range(1, 31)]
    with Pool(2) as p: recs = p.map(task, jobs)
    pd.DataFrame(recs).to_csv(out, index=False); print("saved", out)
