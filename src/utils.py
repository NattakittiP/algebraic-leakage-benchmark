"""
utils.py — Shared utility functions imported by all pipeline scripts.

All statistical helpers are fold-safe: they accept only the data slice
passed in, never touching global state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Distribution samplers
# ---------------------------------------------------------------------------

def truncated_normal(
    mu: float,
    sigma: float,
    low: float,
    high: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a truncated Normal distribution.

    Parameters
    ----------
    mu, sigma : location and scale of the underlying Normal
    low, high : hard truncation bounds (inclusive)
    n         : number of samples
    rng       : numpy random Generator (for reproducibility)
    """
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)


def shifted_lognormal_tg0h(
    n: int,
    rng: np.random.Generator,
    shift: float = 350.0,
    mu_log: float = 5.5,
    sigma_log: float = 0.45,
    low: float = 350.0,
    high: float = 1750.0,
    max_iter: int = 20,
) -> np.ndarray:
    """Sample TG0h from a shifted LogNormal distribution.

    TG0h is baseline triglyceride (mg/dL), restricted to hypertriglyceridaemia
    range [350, 1750].  We draw from LogNormal and keep only samples inside
    [low, high], re-sampling until the quota is filled.

    Parameters
    ----------
    shift     : additive offset so that values start at 350
    mu_log    : mean of log(X - shift)
    sigma_log : SD   of log(X - shift)
    low, high : final truncation bounds
    """
    samples: list[np.ndarray] = []
    needed = n
    for _ in range(max_iter):
        raw = rng.lognormal(mean=mu_log, sigma=sigma_log, size=needed * 3)
        raw = raw + shift
        valid = raw[(raw >= low) & (raw <= high)]
        samples.append(valid[:needed])
        needed -= len(valid[:needed])
        if needed <= 0:
            break
    if needed > 0:
        # Fill any remaining slots with clipped values to avoid infinite loops
        filler = np.clip(
            rng.lognormal(mean=mu_log, sigma=sigma_log, size=needed) + shift,
            low,
            high,
        )
        samples.append(filler)
    return np.concatenate(samples)[:n]


# ---------------------------------------------------------------------------
# Fold-sealed preprocessing helpers
# ---------------------------------------------------------------------------

class FoldSealedScaler:
    """StandardScaler that MUST be fit on training data only.

    Usage inside CV loop:
        scaler = FoldSealedScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
    """

    def __init__(self) -> None:
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FoldSealedScaler":
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0, ddof=0)
        self._std[self._std == 0] = 1.0  # avoid division by zero
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FoldSealedScaler must be fit before transform.")
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def fold_sealed_scaler() -> FoldSealedScaler:
    """Factory that returns a fresh FoldSealedScaler instance."""
    return FoldSealedScaler()


class FoldSealedWinsorizer:
    """Winsorizer that clips at percentiles computed from training data only."""

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self._lower_vals: Optional[np.ndarray] = None
        self._upper_vals: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FoldSealedWinsorizer":
        self._lower_vals = np.percentile(X, self.lower_pct, axis=0)
        self._upper_vals = np.percentile(X, self.upper_pct, axis=0)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FoldSealedWinsorizer must be fit before transform.")
        return np.clip(X, self._lower_vals, self._upper_vals)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def fold_sealed_winsorizer(
    lower_pct: float = 1.0, upper_pct: float = 99.0
) -> FoldSealedWinsorizer:
    return FoldSealedWinsorizer(lower_pct=lower_pct, upper_pct=upper_pct)


# ---------------------------------------------------------------------------
# Label threshold (fold-sealed)
# ---------------------------------------------------------------------------

def fold_sealed_label_threshold(tcr_train: np.ndarray, q: float = 25.0) -> float:
    """Return the Q-th percentile of TCR computed from training fold only.

    The resulting threshold is used to binarise low_TCR labels.
    NEVER call this on the full dataset before splitting.
    """
    return float(np.percentile(tcr_train, q))


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def tost_equivalence_test(
    sample: np.ndarray,
    reference_mean: float,
    reference_sd: float,
    delta: float = 0.1,
    alpha: float = 0.05,
) -> dict:
    """Two One-Sided Tests (TOST) for equivalence with a reference distribution.

    Tests whether sample mean is within [ref_mean - delta*ref_sd,
    ref_mean + delta*ref_sd].

    Returns
    -------
    dict with keys: 'equivalent' (bool), 'p_lower', 'p_upper', 'margin'
    """
    n = len(sample)
    se = reference_sd / np.sqrt(n)
    margin = delta * reference_sd

    t_lower = (sample.mean() - (reference_mean - margin)) / (sample.std(ddof=1) / np.sqrt(n))
    t_upper = ((reference_mean + margin) - sample.mean()) / (sample.std(ddof=1) / np.sqrt(n))

    p_lower = stats.t.sf(t_lower, df=n - 1)
    p_upper = stats.t.sf(t_upper, df=n - 1)

    return {
        "equivalent": bool(p_lower < alpha and p_upper < alpha),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "margin": float(margin),
        "sample_mean": float(sample.mean()),
        "reference_mean": float(reference_mean),
    }


def bca_bootstrap(
    data: np.ndarray,
    statistic,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float, float]:
    """BCa bootstrap confidence interval.

    Parameters
    ----------
    data      : 1-D array of observations
    statistic : callable(array) -> scalar
    n_boot    : number of bootstrap replicates
    ci        : coverage (e.g. 0.95)
    rng_seed  : for reproducibility

    Returns
    -------
    (point_estimate, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(rng_seed)
    n = len(data)
    theta_hat = statistic(data)

    # Bootstrap replicates
    boot = np.array(
        [statistic(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)]
    )

    # Bias-correction factor z0
    z0 = stats.norm.ppf(np.mean(boot < theta_hat))

    # Acceleration factor a (jackknife)
    jack = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    alpha = (1 - ci) / 2
    z_alpha = stats.norm.ppf(alpha)
    z_1alpha = stats.norm.ppf(1 - alpha)

    a1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    lower = float(np.percentile(boot, 100 * a1))
    upper = float(np.percentile(boot, 100 * a2))
    return float(theta_hat), lower, upper


def compute_pearson_spearman_partial_r(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariate_cols: Optional[list[str]] = None,
) -> dict:
    """Compute Pearson r, Spearman ρ, and (optionally) partial Pearson r.

    Partial r is computed by regressing both x and y on covariates and
    correlating the residuals.
    """
    x = df[x_col].values
    y = df[y_col].values

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    result = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }

    if covariate_cols:
        Z = df[covariate_cols].values
        # Residualise x
        res_x = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        # Residualise y
        res_y = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        partial_r, partial_p = stats.pearsonr(res_x, res_y)
        result["partial_r"] = float(partial_r)
        result["partial_p"] = float(partial_p)

    return result


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded numpy Generator (preferred over legacy RandomState)."""
    return np.random.default_rng(seed)


def describe_series(arr: np.ndarray, name: str = "") -> dict:
    """Return descriptive stats dict for an array."""
    return {
        "name": name,
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }
