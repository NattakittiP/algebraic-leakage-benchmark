"""Run all 8+ leakage types and measure AUC inflation relative to a clean baseline."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import FoldSealedScaler, FoldSealedWinsorizer
from src.metrics import compute_auc_inflation, build_benchmark_row
from src.run_clean_pipeline import CLEAN_FEATURES, build_models

warnings.filterwarnings("ignore")


FEATURES_TG4H_LEAK   = CLEAN_FEATURES + ["tg4h"]
FEATURES_TCR_LEAK    = CLEAN_FEATURES + ["tcr"]
FEATURES_COMBINED    = CLEAN_FEATURES + ["tg4h"]


def quick_auc_cv(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Fast outer-CV AUROC. Returns (mean_AUC, sd_AUC)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        import sklearn.base
        clf = sklearn.base.clone(model)
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], prob))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def get_top_shap_feature(
    model,
    X: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> str:
    """Return the feature with highest mean |SHAP| value."""
    if not HAS_SHAP:
        return "N/A (shap not installed)"
    try:
        if "XGB" in model_name or "Forest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X[:200])
        else:
            explainer = shap.LinearExplainer(model, X[:200], feature_perturbation="correlation_dependent")
            shap_vals = explainer.shap_values(X[:200])

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = int(np.argmax(mean_abs))
        return feature_names[top_idx]
    except Exception:
        return "N/A"


def run_clean(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Honest nested CV baseline (imports from run_clean_pipeline)."""
    from src.run_clean_pipeline import nested_cv
    result = nested_cv(df, model_name, model, cfg, outer_folds=5, seed=seed)
    return result


def run_tg4h_leakage(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: add TG4h as predictor (TG4h is algebraically embedded in TCR)."""
    features = FEATURES_TG4H_LEAK
    X = df[features].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    auc_mean, auc_sd = quick_auc_cv(X_scaled, y, model, seed=seed)

    top_feature = "N/A"
    if HAS_SHAP:
        import sklearn.base
        clf = sklearn.base.clone(model)
        clf.fit(X_scaled, y)
        top_feature = get_top_shap_feature(clf, X_scaled, features, model_name)

    return {
        "model": model_name,
        "pipeline": "tg4h_leakage",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "top_SHAP_feature": top_feature,
    }


def run_tcr_leakage(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: add TCR directly as a predictor (TCR IS the label source, AUC ≈ 1.00)."""
    features = FEATURES_TCR_LEAK
    X = df[features].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    auc_mean, auc_sd = quick_auc_cv(X_scaled, y, model, seed=seed)

    return {
        "model": model_name,
        "pipeline": "tcr_leakage",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "top_SHAP_feature": "tcr",
    }


def run_global_scaling(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: fit StandardScaler on full dataset before CV split."""
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    auc_mean, auc_sd = quick_auc_cv(X_scaled, y, model, seed=seed)
    return {
        "model": model_name,
        "pipeline": "global_scaling",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "top_SHAP_feature": "N/A",
    }


def run_global_winsorization(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: winsorize on full dataset before CV split."""
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    lower = np.percentile(X, 1, axis=0)
    upper = np.percentile(X, 99, axis=0)
    X_win = np.clip(X, lower, upper)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_win)

    auc_mean, auc_sd = quick_auc_cv(X_scaled, y, model, seed=seed)
    return {
        "model": model_name,
        "pipeline": "global_winsorization",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "top_SHAP_feature": "N/A",
    }


def run_global_label_threshold(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: use Q1 TCR from full dataset (pre-computed label)."""
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values

    threshold_global = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold_global).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    auc_mean, auc_sd = quick_auc_cv(X_scaled, y, model, seed=seed)
    return {
        "model": model_name,
        "pipeline": "global_label_threshold",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "top_SHAP_feature": "N/A",
    }


def run_smote_before_cv(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: apply SMOTE to the full dataset before splitting into folds."""
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    auc_mean, auc_sd = quick_auc_cv(X_res, y_res, model, seed=seed)
    return {
        "model": model_name,
        "pipeline": "smote_before_cv",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "top_SHAP_feature": "N/A",
    }


def run_feature_selection_leak(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: feature selection using ANOVA F-score on full dataset before CV."""
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    k = min(5, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_sel = selector.fit_transform(X, y)
    selected_features = [CLEAN_FEATURES[i] for i in selector.get_support(indices=True)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    auc_mean, auc_sd = quick_auc_cv(X_scaled, y, model, seed=seed)
    return {
        "model": model_name,
        "pipeline": "feature_selection_leakage",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "selected_features": selected_features,
        "top_SHAP_feature": "N/A",
    }


def run_combined_leakage(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: TG4h predictor + global preprocessing + SMOTE before CV."""
    features = FEATURES_COMBINED
    X = df[features].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    lower = np.percentile(X, 1, axis=0)
    upper = np.percentile(X, 99, axis=0)
    X_win = np.clip(X, lower, upper)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_win)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    auc_mean, auc_sd = quick_auc_cv(X_res, y_res, model, seed=seed)

    top_feature = "N/A"
    if HAS_SHAP:
        import sklearn.base
        clf = sklearn.base.clone(model)
        clf.fit(X_res, y_res)
        top_feature = get_top_shap_feature(clf, X_res, features, model_name)

    return {
        "model": model_name,
        "pipeline": "combined_leakage",
        "AUROC_mean": round(auc_mean, 4),
        "AUROC_sd": round(auc_sd, 4),
        "top_SHAP_feature": top_feature,
    }


PIPELINE_RUNNERS = {
    "tg4h_leakage":             run_tg4h_leakage,
    "tcr_leakage":              run_tcr_leakage,
    "global_scaling":           run_global_scaling,
    "global_winsorization":     run_global_winsorization,
    "global_label_threshold":   run_global_label_threshold,
    "smote_before_cv":          run_smote_before_cv,
    "feature_selection_leakage": run_feature_selection_leak,
    "combined_leakage":         run_combined_leakage,
}


def run_all_leakage_scenarios(
    df: pd.DataFrame,
    cfg: dict,
    model_name: str,
    model,
    seed: int = 42,
) -> list[dict]:
    """Run clean + all 8 leakage scenarios for one model."""
    results = []

    tqdm.write(f"  [{model_name}] running clean baseline ...")
    clean_res = run_clean(df, model, model_name, cfg, seed)
    clean_auc = clean_res.get("AUROC_mean", 0.5)
    clean_res["AUC_inflation"] = 0.0
    results.append(clean_res)
    tqdm.write(f"  [{model_name}] clean AUROC = {clean_auc:.4f}")

    leakage_pbar = tqdm(
        PIPELINE_RUNNERS.items(),
        total=len(PIPELINE_RUNNERS),
        desc=f"  Leakage types [{model_name}]",
        ncols=90,
        colour="red",
        leave=True,
    )
    for pipeline_name, runner in leakage_pbar:
        leakage_pbar.set_description(f"  [{model_name}] {pipeline_name}")
        try:
            res = runner(df, model, model_name, cfg, seed)
            res["AUC_inflation"] = round(
                compute_auc_inflation(res["AUROC_mean"], clean_auc), 4
            )
            results.append(res)
            inflation = res["AUC_inflation"]
            leakage_pbar.set_postfix(
                AUROC=f"{res['AUROC_mean']:.4f}",
                inflation=f"+{inflation:.4f}",
                refresh=True,
            )
        except Exception as e:
            tqdm.write(f"    ERROR [{pipeline_name}]: {e}")
            results.append({
                "model": model_name,
                "pipeline": pipeline_name,
                "AUROC_mean": None,
                "error": str(e),
            })

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Run leakage benchmark for all 8 leakage types.")
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="results/tables/leakage_benchmark.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=None, help="Run only this model (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data, keep_default_na=False, na_values=["NA", "NaN", "nan", ""])
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = build_models(cfg.get("models", {}))
    if args.model:
        models = {k: v for k, v in models.items() if k == args.model}

    all_results = []
    model_pbar = tqdm(
        models.items(),
        total=len(models),
        desc="Models (leakage benchmark)",
        ncols=90,
        colour="magenta",
    )
    for model_name, model in model_pbar:
        model_pbar.set_description(f"Model: {model_name}")
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"  Model: {model_name}")
        tqdm.write(f"{'='*50}")
        results = run_all_leakage_scenarios(df, cfg, model_name, model, seed=args.seed)
        all_results.extend(results)

    rows = []
    for r in all_results:
        rows.append({
            "model": r.get("model"),
            "pipeline": r.get("pipeline"),
            "AUROC_mean": r.get("AUROC_mean"),
            "AUROC_sd": r.get("AUROC_sd"),
            "PR_AUC_mean": r.get("PR_AUC_mean"),
            "Brier_mean": r.get("Brier_mean"),
            "AUC_inflation": r.get("AUC_inflation"),
            "top_SHAP_feature": r.get("top_SHAP_feature", ""),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved leakage benchmark -> {out_path}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed JSON -> {json_path}")

    print("\n=== Leakage Benchmark Summary (LogisticRegression) ===")
    for r in all_results:
        if r.get("model") == "LogisticRegression" and r.get("AUROC_mean") is not None:
            infl = r.get("AUC_inflation", 0)
            print(f"  {r['pipeline']:30s} AUROC={r['AUROC_mean']:.4f}  inflation={infl:+.4f}")


if __name__ == "__main__":
    main()
