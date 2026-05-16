"""
test_generator_ranges.py
Test that all generated variables stay within their specified bounds.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate_synthetic_data import generate


@pytest.fixture(scope="module")
def null_df():
    """Generate a small null dataset for testing."""
    import yaml
    with open("config/generator_null.yaml") as f:
        cfg = yaml.safe_load(f)
    return generate(cfg, seed=42, n=500)


def test_age_range(null_df):
    assert null_df["age"].min() >= 18.0
    assert null_df["age"].max() <= 75.0


def test_bmi_range(null_df):
    assert null_df["bmi"].min() >= 16.0
    assert null_df["bmi"].max() <= 33.0


def test_hct_range(null_df):
    assert null_df["hct"].min() >= 30.0
    assert null_df["hct"].max() <= 55.0


def test_tp_range(null_df):
    assert null_df["tp"].min() >= 5.0
    assert null_df["tp"].max() <= 8.9


def test_hdl_range(null_df):
    assert null_df["hdl"].min() >= 25.0
    assert null_df["hdl"].max() <= 87.0


def test_ldl_range(null_df):
    assert null_df["ldl"].min() >= 70.0
    assert null_df["ldl"].max() <= 215.0


def test_tg0h_range(null_df):
    assert null_df["tg0h"].min() >= 350.0
    assert null_df["tg0h"].max() <= 1750.0


def test_tcr_range(null_df):
    assert null_df["tcr"].min() >= -10.0
    assert null_df["tcr"].max() <= 99.9


def test_tg4h_positive(null_df):
    assert (null_df["tg4h"] > 0).all(), "All TG4h values must be positive"


def test_tg4h_upper_bound(null_df):
    assert null_df["tg4h"].max() <= 1300.0, "TG4h must not exceed 1300 mg/dL"


def test_sex_binary(null_df):
    assert set(null_df["sex"].unique()).issubset({0, 1}), "Sex must be 0 or 1"


def test_sex_prevalence(null_df):
    p_male = null_df["sex"].mean()
    assert 0.40 <= p_male <= 0.58, f"Male prevalence {p_male:.3f} out of expected range"


def test_n_rows(null_df):
    assert len(null_df) == 500


def test_wbv_non_negative(null_df):
    assert (null_df["wbv"] > 0).all(), "WBV must be positive"


def test_record_id_unique(null_df):
    assert null_df["record_id"].nunique() == len(null_df), "record_id must be unique"
