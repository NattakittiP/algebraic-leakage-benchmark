"""Null-scenario S2 sweep (Table S5) with per-task checkpointing.
Re-uses run_one_seed() from run_sample_size_sensitivity.py (clean + leaky, LR, 5-fold).
Usage: python src/run_s5_null_checkpointed.py OUT.jsonl [n_jobs]"""
import sys, json, os, yaml, contextlib, io
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_sample_size_sensitivity import run_one_seed, SAMPLE_SIZES
CG = yaml.safe_load(open("config/generator_null.yaml")); CM = yaml.safe_load(open("config/model_config.yaml"))
def task(a):
    n, s = a
    with contextlib.redirect_stderr(io.StringIO()):
        return run_one_seed("null", CG, CM, n, s, outer_folds=5, run_leaky=True)
if __name__ == "__main__":
    out = sys.argv[1]; nj = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["n"], r["seed"]))
    todo = [(n, s) for n in SAMPLE_SIZES for s in range(1, 31) if (n, s) not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with Pool(nj) as p, open(out, "a") as f:
        for r in p.imap_unordered(task, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["n"], r["seed"], r.get("clean_AUROC"), r.get("leaky_AUROC"), flush=True)
