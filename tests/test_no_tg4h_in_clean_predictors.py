"""
test_no_tg4h_in_clean_predictors.py
CRITICAL: Assert that TG4h and TCR are NOT in the clean feature set.

Including either of these in the clean pipeline constitutes definitional leakage
and would invalidate the benchmark's clean baseline.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_clean_pipeline import CLEAN_FEATURES


def test_tg4h_not_in_clean_features():
    """TG4h must be excluded from clean predictors (definitional leakage)."""
    assert "tg4h" not in CLEAN_FEATURES, (
        "CRITICAL: 'tg4h' found in CLEAN_FEATURES! "
        "TG4h is derived from TCR and including it causes definitional leakage."
    )


def test_tcr_not_in_clean_features():
    """TCR (the label source) must be excluded from clean predictors (target leakage)."""
    assert "tcr" not in CLEAN_FEATURES, (
        "CRITICAL: 'tcr' found in CLEAN_FEATURES! "
        "TCR is used to derive the binary label and including it causes target leakage."
    )


def test_clean_features_not_empty():
    """Clean feature set must have at least one predictor."""
    assert len(CLEAN_FEATURES) > 0, "CLEAN_FEATURES must not be empty"


def test_wbv_in_clean_features():
    """WBV must be in clean features (it is the key biomarker under investigation)."""
    assert "wbv" in CLEAN_FEATURES, (
        "WBV should be in CLEAN_FEATURES — it is the primary biomarker being evaluated."
    )


def test_tg0h_in_clean_features():
    """TG0h (baseline TG) should be in clean features."""
    assert "tg0h" in CLEAN_FEATURES, "TG0h (baseline triglyceride) should be a clean predictor"


def test_clean_features_subset_of_known():
    """All clean features must be biologically valid baseline predictors."""
    valid_predictors = {"age", "sex", "bmi", "hct", "tp", "wbv", "hdl", "ldl", "tg0h"}
    for feat in CLEAN_FEATURES:
        assert feat in valid_predictors, (
            f"Unexpected feature '{feat}' in CLEAN_FEATURES. "
            f"Valid predictors are: {valid_predictors}"
        )
