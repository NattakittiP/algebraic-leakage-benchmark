"""Clean fold-sealed ML pipeline (honest baseline). All preprocessing fitted on training folds only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not found — XGB model will be skipped.")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import FoldSealedScaler, FoldSealedWinsorizer
from src.calibration import calibration_summary
from src.metrics import compute_auc, compute_pr_auc

# Clean predictors: do NOT include tcr or tg4h
CLEAN_FEATURES = ["age", "sex", "bmi", "hct", "tp", "wbv", "hdl", "ldl", "tg0h"]


def build_models(cfg: dict) -> dict:
    """Return dict of model_name -> unfitted sklearn estimator."""
    lr_cfg = cfg.get("logistic_regression", {})
    rf_cfg = cfg.get("random_forest", {})
    svm_cfg = cfg.get("svm", {})
    xgb_cfg = cfg.get("xgboost", {})

    models = {
        "LogisticRegression": LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver="lbfgs",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            C=svm_cfg.get("C", 1.0),
            kernel=svm_cfg.get("kernel", "rbf"),
            probability=True,
            random_state=42,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 200),
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

    return models


def fold_sealed_preprocess(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    tcr_train: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply all preprocessing with parameters derived from training fold only.

    Steps: Winsorise → Scale → Compute label threshold (Q1 TCR) → Binarise → SMOTE.
    Returns X_train_proc, X_test_proc, y_train_binary, label_threshold.
    """
    win_cfg = cfg.get("winsorizer", {})
    lower_pct = win_cfg.get("lower_pct", 1.0)
    upper_pct = win_cfg.get("upper_pct", 99.0)

    winsorizer = FoldSealedWinsorizer(lower_pct=lower_pct, upper_pct=upper_pct)
    X_train_w = winsorizer.fit_transform(X_train)
    X_test_w = winsorizer.transform(X_test)

    scaler = FoldSealedScaler()
    X_train_s = scaler.fit_transform(X_train_w)
    X_test_s = scaler.transform(X_test_w)

    label_q = cfg.get("label_threshold_percentile", 25.0)
    threshold = float(np.percentile(tcr_train, label_q))

    y_train_bin = (tcr_train <= threshold).astype(int)

    smote_cfg = cfg.get("smote", {})
    if smote_cfg.get("enabled", True) and y_train_bin.sum() >= 2 and (1 - y_train_bin).sum() >= 2:
        smote = SMOTE(
            k_neighbors=smote_cfg.get("k_neighbors", 5),
            random_state=smote_cfg.get("random_state", 42),
        )
        X_train_s, y_train_bin = smote.fit_resample(X_train_s, y_train_bin)

    return X_train_s, X_test_s, y_train_bin, threshold


def nested_cv(
    df: pd.DataFrame,
    model_name: str,
    model,
    cfg: dict,
    outer_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Run nested 5×5 CV for one model. Returns dict with per-fold and aggregate metrics."""
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    # Global Q1 used for stratification only, not for training labels.
    global_q1 = np.percentile(tcr, cfg.get("label_threshold_percentile", 25.0))
    y_stratify = (tcr <= global_q1).astype(int)

    fold_results = []

    fold_pbar = tqdm(
        enumerate(outer_cv.split(X, y_stratify)),
        total=outer_folds,
        desc=f"  CV folds [{model_name}]",
        ncols=90,
        colour="blue",
        leave=True,
    )

    for fold_idx, (train_idx, test_idx) in fold_pbar:
        fold_pbar.set_postfix(fold=f"{fold_idx+1}/{outer_folds}", refresh=True)

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        tcr_train = tcr[train_idx]
        tcr_test = tcr[test_idx]

        X_train_proc, X_test_proc, y_train_bin, threshold = fold_sealed_preprocess(
            X_train_raw, X_test_raw, None, tcr_train, cfg
        )

        y_test_bin = (tcr_test <= threshold).astype(int)

        if len(np.unique(y_test_bin)) < 2:
            tqdm.write(f"  Fold {fold_idx+1}: only one class in test — skipping")
            continue

        import sklearn.base
        clf = sklearn.base.clone(model)
        clf.fit(X_train_proc, y_train_bin)
        y_prob = clf.predict_proba(X_test_proc)[:, 1]

        auc = float(roc_auc_score(y_test_bin, y_prob))
        pr_auc = float(average_precision_score(y_test_bin, y_prob))
        brier = float(brier_score_loss(y_test_bin, y_prob))

        fold_results.append({
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "label_threshold": round(threshold, 3),
            "prevalence_test": round(y_test_bin.mean(), 4),
            "AUROC": round(auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Brier": round(brier, 4),
        })
        fold_pbar.set_postfix(
            fold=f"{fold_idx+1}/{outer_folds}",
            AUROC=f"{auc:.4f}",
            refresh=True,
        )

    if not fold_results:
        return {"model": model_name, "error": "No valid folds"}

    aucs = np.array([r["AUROC"] for r in fold_results])
    pr_aucs = np.array([r["PR_AUC"] for r in fold_results])
    briers = np.array([r["Brier"] for r in fold_results])

    return {
        "model": model_name,
        "pipeline": "clean",
        "AUROC_mean": round(float(aucs.mean()), 4),
        "AUROC_sd": round(float(aucs.std(ddof=1)), 4),
        "PR_AUC_mean": round(float(pr_aucs.mean()), 4),
        "PR_AUC_sd": round(float(pr_aucs.std(ddof=1)), 4),
        "Brier_mean": round(float(briers.mean()), 4),
        "Brier_sd": round(float(briers.std(ddof=1)), 4),
        "n_folds": len(fold_results),
        "fold_details": fold_results,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run clean fold-sealed ML pipeline.")
    p.add_argument("--data", required=True, help="Path to synthetic CSV")
    p.add_argument("--config", required=True, help="Path to model_config.yaml")
    p.add_argument("--out", default="results/tables/clean_results.csv",
                   help="Output CSV path for aggregate results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data, keep_default_na=False, na_values=["NA", "NaN", "nan", ""])
    print(f"Loaded {len(df)} records | features: {CLEAN_FEATURES}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("models", {})
    cv_cfg = cfg.get("cv", {})
    outer_folds = cv_cfg.get("outer_folds", 5)

    models = build_models(model_cfg)
    all_results = []

    model_pbar = tqdm(
        models.items(),
        total=len(models),
        desc="Models",
        ncols=90,
        colour="magenta",
    )
    for name, model in model_pbar:
        model_pbar.set_description(f"Model: {name}")
        result = nested_cv(
            df, name, model, cfg,
            outer_folds=outer_folds,
            seed=args.seed,
        )
        all_results.append(result)
        auroc = result.get("AUROC_mean", float("nan"))
        sd    = result.get("AUROC_sd", 0)
        model_pbar.set_postfix(AUROC=f"{auroc:.4f}±{sd:.4f}")
        tqdm.write(f"  ✓ {name}: AUROC = {auroc:.4f} ± {sd:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        rows.append({
            "model": r["model"],
            "pipeline": r.get("pipeline", "clean"),
            "AUROC_mean": r.get("AUROC_mean"),
            "AUROC_sd": r.get("AUROC_sd"),
            "PR_AUC_mean": r.get("PR_AUC_mean"),
            "PR_AUC_sd": r.get("PR_AUC_sd"),
            "Brier_mean": r.get("Brier_mean"),
            "Brier_sd": r.get("Brier_sd"),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved results -> {out_path}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed JSON -> {json_path}")

    print("\n=== Clean Pipeline Summary ===")
    for r in all_results:
        print(f"  {r['model']}: AUROC = {r.get('AUROC_mean','N/A'):.4f}")
    print("\nExpected (null scenario): AUROC ≈ 0.48–0.52")


if __name__ == "__main__":
    main()
