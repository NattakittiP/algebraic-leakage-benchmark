"""
metrics.py — Custom benchmark metrics for leakage detection.

Three core metrics:
  1. AUC Inflation            = AUC_leaky − AUC_clean
  2. Attribution Distortion   = rank_shift of each feature between clean/leaky SHAP
  3. False Attribution Rate   = P(WBV ranked top-k | true WBV effect = 0)

All functions accept plain numpy arrays / dicts; no sklearn dependency required
so they can be imported in tests without a heavy ML environment.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC with fallback for degenerate label distributions."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under Precision-Recall curve (average precision)."""
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Core leakage benchmark metrics
# ---------------------------------------------------------------------------

def compute_auc_inflation(
    auc_leaky: float,
    auc_clean: float,
) -> float:
    """AUC Inflation = AUC_leaky − AUC_clean.

    A positive value indicates artificial performance boost due to leakage.
    Values near 0 indicate the leakage type did not inflate performance.
    """
    return float(auc_leaky - auc_clean)


def compute_attribution_distortion(
    shap_ranks_clean: Dict[str, int],
    shap_ranks_leaky: Dict[str, int],
) -> Dict[str, int]:
    """Attribution Distortion = Rank_clean(feature) − Rank_leaky(feature).

    Parameters
    ----------
    shap_ranks_clean : {feature_name: rank} from the clean model (1 = most important)
    shap_ranks_leaky : {feature_name: rank} from the leaky model

    Returns
    -------
    Dict mapping feature_name -> distortion (positive = rose in leaky ranking)
    """
    all_features = set(shap_ranks_clean) | set(shap_ranks_leaky)
    max_rank = max(
        max(shap_ranks_clean.values(), default=0),
        max(shap_ranks_leaky.values(), default=0),
    ) + 1  # fallback rank if feature absent in one model

    distortion = {}
    for feat in all_features:
        r_clean = shap_ranks_clean.get(feat, max_rank)
        r_leaky = shap_ranks_leaky.get(feat, max_rank)
        distortion[feat] = int(r_clean - r_leaky)  # positive = feature ascended

    return distortion


def compute_false_attribution_rate(
    shap_ranks_per_run: List[Dict[str, int]],
    target_feature: str = "WBV",
    top_k: int = 3,
) -> float:
    """False Attribution Rate = P(target_feature ranked in top-k across runs).

    Used in the null scenario where the target feature (WBV) has zero true effect.
    A high FAR means the pipeline falsely promotes WBV to top importance.

    Parameters
    ----------
    shap_ranks_per_run : list of {feature: rank} dicts, one per seed / fold
    target_feature     : feature that should have ~zero importance
    top_k              : threshold for "top importance"

    Returns
    -------
    Proportion of runs where target_feature ranks ≤ top_k (false positive rate)
    """
    if not shap_ranks_per_run:
        return float("nan")
    count = sum(
        1
        for ranks in shap_ranks_per_run
        if ranks.get(target_feature, 9999) <= top_k
    )
    return float(count / len(shap_ranks_per_run))


# ---------------------------------------------------------------------------
# Decision Curve Analysis (net benefit)
# ---------------------------------------------------------------------------

def compute_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Net Benefit curve for Decision Curve Analysis.

    NB(pt) = (TP/N) - (FP/N) * (pt / (1 - pt))

    Parameters
    ----------
    y_true     : binary outcome array
    y_prob     : predicted probabilities
    thresholds : probability thresholds to evaluate (default linspace 0.01, 0.99)

    Returns
    -------
    (thresholds, net_benefit) as numpy arrays
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    n = len(y_true)
    net_benefit = np.zeros(len(thresholds))

    for i, pt in enumerate(thresholds):
        predicted_pos = y_prob >= pt
        tp = np.sum((predicted_pos == 1) & (y_true == 1))
        fp = np.sum((predicted_pos == 1) & (y_true == 0))
        net_benefit[i] = tp / n - (fp / n) * (pt / (1.0 - pt))

    return thresholds, net_benefit


# ---------------------------------------------------------------------------
# Benchmark summary table builder
# ---------------------------------------------------------------------------

def build_benchmark_row(
    pipeline_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_clean: Optional[float] = None,
    top_shap_feature: Optional[str] = None,
    shap_ranks_clean: Optional[Dict[str, int]] = None,
    shap_ranks_leaky: Optional[Dict[str, int]] = None,
) -> dict:
    """Compute all benchmark metrics for one pipeline and return as a dict row.

    Parameters
    ----------
    pipeline_name    : label for this leakage scenario
    y_true           : binary ground-truth labels
    y_prob           : predicted probabilities from the model
    auc_clean        : AUC from the corresponding clean pipeline (for inflation)
    top_shap_feature : name of the feature with highest mean |SHAP| in this run
    shap_ranks_clean : SHAP rank dict from clean model (for distortion)
    shap_ranks_leaky : SHAP rank dict from this leaky model (for distortion)
    """
    auc = compute_auc(y_true, y_prob)
    pr_auc = compute_pr_auc(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))

    row: dict = {
        "pipeline": pipeline_name,
        "AUROC": round(auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Brier": round(brier, 4),
        "top_SHAP_feature": top_shap_feature or "",
    }

    if auc_clean is not None:
        row["AUC_inflation"] = round(compute_auc_inflation(auc, auc_clean), 4)

    if shap_ranks_clean and shap_ranks_leaky:
        distortion = compute_attribution_distortion(shap_ranks_clean, shap_ranks_leaky)
        row["WBV_rank_distortion"] = distortion.get("WBV", 0)

    return row


# ---------------------------------------------------------------------------
# Scenario result aggregator
# ---------------------------------------------------------------------------

def aggregate_seed_results(
    records: List[dict],
    auc_key: str = "AUROC",
    ci: float = 0.95,
) -> dict:
    """Aggregate AUROC (or other metric) across seeds.

    Parameters
    ----------
    records : list of dicts, each containing auc_key
    auc_key : which metric to aggregate
    ci      : coverage for percentile CI

    Returns
    -------
    dict with mean, sd, median, ci_lower, ci_upper, pct_in_null_range
    """
    vals = np.array([r[auc_key] for r in records if auc_key in r], dtype=float)
    if len(vals) == 0:
        return {}

    alpha = (1 - ci) / 2
    return {
        f"{auc_key}_mean": float(np.mean(vals)),
        f"{auc_key}_sd": float(np.std(vals, ddof=1)),
        f"{auc_key}_median": float(np.median(vals)),
        f"{auc_key}_ci_lower": float(np.percentile(vals, 100 * alpha)),
        f"{auc_key}_ci_upper": float(np.percentile(vals, 100 * (1 - alpha))),
        f"{auc_key}_pct_null_range": float(
            np.mean((vals >= 0.45) & (vals <= 0.55)) * 100
        ),
        "n_seeds": int(len(vals)),
    }
