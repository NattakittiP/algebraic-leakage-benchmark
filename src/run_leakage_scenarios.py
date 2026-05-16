"""
run_leakage_scenarios.py — Run all 8+ leakage types and measure AUC inflation.

Leakage types implemented:
  1. clean            — honest baseline (no leakage)
  2. tg4h_leakage     — add TG4h as predictor (definitional leakage)
  3. tcr_leakage      — add TCR as predictor (target leakage)
  4. global_scaling   — fit scaler on full dataset before CV split
  5. global_winsor    — winsorize on full dataset before split
  6. global_label     — use Q1 TCR from full dataset (label leakage)
  7. smote_before_cv  — oversample before splitting into folds
  8. feature_sel_leak — select features from full data before CV
  9. combined_leakage — TG4h + global preprocessing + SMOTE

Outputs:
  - Benchmark table CSV (one row per leakage type × model)
  - AUC Inflation for each type relative to clean baseline

Usage:
  python src/run_leakage_scenarios.py \\
      --data data/paired_tcr_null_v1_seed2026.csv \\
      --config config/model_config.yaml \\
      --out results/tables/leakage_benchmark.csv
"""

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


# ---------------------------------------------------------------------------
# Feature sets per leakage type
# ---------------------------------------------------------------------------

FEATURES_TG4H_LEAK   = CLEAN_FEATURES + ["tg4h"]          # adds definitionally-leaky TG4h
FEATURES_TCR_LEAK    = CLEAN_FEATURES + ["tcr"]            # adds target variable
FEATURES_COMBINED    = CLEAN_FEATURES + ["tg4h"]           # same as tg4h for combined


# ---------------------------------------------------------------------------
# Helper: quick CV evaluation
# ---------------------------------------------------------------------------

def quick_auc_cv(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Fast outer-CV AUROC (no inner CV, no nested preprocessing).

    Used for leakage scenarios where the leakage itself is the preprocessing.
    Returns (mean_AUC, sd_AUC).
    """
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
            shap_vals = explainer.shap_values(X[:200])   # sample for speed
        else:
            explainer = shap.LinearExplainer(model, X[:200], feature_perturbation="correlation_dependent")
            shap_vals = explainer.shap_values(X[:200])

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # positive class for binary
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = int(np.argmax(mean_abs))
        return feature_names[top_idx]
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Leakage pipeline runners
# ---------------------------------------------------------------------------

def run_clean(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Honest nested CV baseline (imports from run_clean_pipeline)."""
    from src.run_clean_pipeline import nested_cv
    result = nested_cv(df, model_name, model, cfg, outer_folds=5, seed=seed)
    return result


def run_tg4h_leakage(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: add TG4h as predictor.

    TG4h is mathematically embedded in TCR = (TG0h - TG4h) / TG0h × 100.
    Including TG4h causes near-perfect reconstruction of the label.
    Expected AUROC: ~0.90+
    """
    features = FEATURES_TG4H_LEAK
    X = df[features].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    # Global preprocessing (leaky: all data before split)
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
    """Leakage: add TCR directly as a predictor.

    TCR IS the quantity used to derive the binary label → AUC ≈ 1.00.
    """
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
        "top_SHAP_feature": "tcr",  # obviously TCR dominates
    }


def run_global_scaling(df: pd.DataFrame, model, model_name: str, cfg: dict, seed: int) -> dict:
    """Leakage: fit StandardScaler on full dataset before CV split."""
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values
    threshold = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y = (tcr <= threshold).astype(int)

    # LEAKY: fit scaler on ALL data
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

    # LEAKY: winsorize on all data
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

    # LEAKY: threshold from full data
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

    # LEAKY: SMOTE before any fold split
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

    # LEAKY: select features from full data
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

    # All leakage combined
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

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

    # Clean baseline first
    tqdm.write(f"  [{model_name}] running clean baseline ...")
    clean_res = run_clean(df, model, model_name, cfg, seed)
    clean_auc = clean_res.get("AUROC_mean", 0.5)
    clean_res["AUC_inflation"] = 0.0
    results.append(clean_res)
    tqdm.write(f"  [{model_name}] clean AUROC = {clean_auc:.4f}")

    # Each leakage type
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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

    # Flatten to CSV rows
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
