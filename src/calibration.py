"""
calibration.py — Calibration analysis module.

Provides:
  - Brier score
  - Expected Calibration Error (ECE)
  - Calibration slope / intercept (logistic calibration regression)
  - Platt scaling (sigmoid recalibration)
  - Isotonic regression recalibration
  - Calibration curve data for plotting

All functions are fold-safe and operate on arrays passed in.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    return float(brier_score_loss(y_true, y_prob))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true   : binary ground-truth labels
    y_prob   : predicted probabilities
    n_bins   : number of probability bins
    strategy : 'uniform' (equal-width) or 'quantile' (equal-frequency)
    """
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(y_true)

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(bins) - 2 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """Calibration slope and intercept via logistic regression on log-odds.

    A perfectly calibrated model has slope=1, intercept=0.

    Returns
    -------
    (slope, intercept)
    """
    eps = 1e-7
    log_odds = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))

    # Fit logistic regression: logit(P(Y=1)) = intercept + slope * log_odds(predicted)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), y_true)

    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


# ---------------------------------------------------------------------------
# Calibration curve data
# ---------------------------------------------------------------------------

def get_calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fraction_of_positives, mean_predicted_value) for a reliability diagram.

    Wraps sklearn.calibration.calibration_curve.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    return frac_pos, mean_pred


# ---------------------------------------------------------------------------
# Recalibration methods
# ---------------------------------------------------------------------------

def platt_scaling(
    y_true_cal: np.ndarray,
    y_prob_cal: np.ndarray,
    y_prob_test: np.ndarray,
) -> np.ndarray:
    """Platt scaling: fit sigmoid on calibration set, apply to test set.

    Parameters
    ----------
    y_true_cal  : binary labels in calibration split
    y_prob_cal  : raw model scores in calibration split
    y_prob_test : raw model scores in test set to recalibrate

    Returns
    -------
    Recalibrated probabilities for test set
    """
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(y_prob_cal.reshape(-1, 1), y_true_cal)
    return lr.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]


def isotonic_regression_calibration(
    y_true_cal: np.ndarray,
    y_prob_cal: np.ndarray,
    y_prob_test: np.ndarray,
) -> np.ndarray:
    """Isotonic regression recalibration.

    Parameters
    ----------
    y_true_cal  : binary labels in calibration split
    y_prob_cal  : raw model scores in calibration split
    y_prob_test : raw model scores in test set to recalibrate

    Returns
    -------
    Recalibrated probabilities for test set (clipped to [0, 1])
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob_cal, y_true_cal)
    return np.clip(ir.predict(y_prob_test), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Comprehensive calibration summary
# ---------------------------------------------------------------------------

def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    n_bins: int = 10,
) -> dict:
    """Return all calibration metrics as a single dict."""
    slope, intercept = compute_calibration_slope_intercept(y_true, y_prob)
    return {
        "label": label,
        "brier": round(compute_brier_score(y_true, y_prob), 4),
        "ece": round(compute_ece(y_true, y_prob, n_bins=n_bins), 4),
        "cal_slope": round(slope, 4),
        "cal_intercept": round(intercept, 4),
    }
