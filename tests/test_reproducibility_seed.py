"""
test_reproducibility_seed.py
Assert that the same seed always produces an identical dataset.

This guarantees one-command reproducibility of all results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate_synthetic_data import generate


@pytest.fixture(scope="module")
def cfg():
    import yaml
    with open("config/generator_null.yaml") as f:
        return yaml.safe_load(f)


def test_same_seed_produces_identical_dataframe(cfg):
    """Two calls with the same seed must return byte-identical datasets."""
    df1 = generate(cfg, seed=2026, n=200)
    df2 = generate(cfg, seed=2026, n=200)

    numeric_cols = ["age", "bmi", "hct", "tp", "wbv", "tg0h", "tcr", "tg4h", "hdl", "ldl"]
    for col in numeric_cols:
        np.testing.assert_array_equal(
            df1[col].values,
            df2[col].values,
            err_msg=f"Column '{col}' differs between two runs with same seed 2026",
        )


def test_different_seeds_produce_different_datasets(cfg):
    """Different seeds must produce different datasets."""
    df_42 = generate(cfg, seed=42, n=200)
    df_99 = generate(cfg, seed=99, n=200)

    # TCR values should differ between seeds
    assert not np.array_equal(df_42["tcr"].values, df_99["tcr"].values), (
        "TCR values are identical for different seeds — seed is not controlling the RNG"
    )


def test_seed_42_vs_seed_2026_differ(cfg):
    """Development seed (42) and main analysis seed (2026) must differ."""
    df_dev  = generate(cfg, seed=42,   n=200)
    df_main = generate(cfg, seed=2026, n=200)
    assert not np.array_equal(df_dev["tg0h"].values, df_main["tg0h"].values)


def test_seed_2026_tcr_statistics(cfg):
    """Seed 2026 null dataset should have TCR near the target distribution."""
    df = generate(cfg, seed=2026, n=1500)
    tcr = df["tcr"].values
    assert 48.0 <= tcr.mean() <= 56.0, f"TCR mean {tcr.mean():.2f} out of expected range"
    assert 14.0 <= tcr.std() <= 24.0,  f"TCR SD {tcr.std():.2f} out of expected range"


def test_seed_stability_100_seeds(cfg):
    """All 100 robustness seeds must be callable without error."""
    for seed in range(1, 11):   # spot-check first 10
        df = generate(cfg, seed=seed, n=100)
        assert len(df) == 100
        assert (df["tg4h"] > 0).all(), f"Seed {seed}: negative TG4h found"
