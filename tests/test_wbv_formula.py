"""
test_wbv_formula.py
Assert that WBV = 0.12 × Hct + 0.17 × (TP − 2.07) is implemented exactly.

This is the de Simone (1990) formula and must not be altered.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate_synthetic_data import compute_wbv, generate


def test_wbv_formula_scalar():
    """Verify formula on a single known input."""
    hct = 41.7
    tp  = 6.88
    expected = 0.12 * 41.7 + 0.17 * (6.88 - 2.07)
    result = compute_wbv(np.array([hct]), np.array([tp]))[0]
    assert abs(result - expected) < 1e-6, (
        f"WBV formula incorrect: expected {expected:.6f}, got {result:.6f}"
    )


def test_wbv_formula_vectorised():
    """Verify formula applied to arrays."""
    rng = np.random.default_rng(42)
    hct = rng.uniform(30, 55, 100)
    tp  = rng.uniform(5, 8.9, 100)

    expected = 0.12 * hct + 0.17 * (tp - 2.07)
    result = compute_wbv(hct, tp)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_wbv_dataset_matches_formula():
    """WBV in generated dataset must exactly match the formula applied to Hct/TP."""
    import yaml
    with open("config/generator_null.yaml") as f:
        cfg = yaml.safe_load(f)

    df = generate(cfg, seed=42, n=300)

    recomputed = 0.12 * df["hct"] + 0.17 * (df["tp"] - 2.07)
    diff = (df["wbv"] - recomputed).abs()
    assert diff.max() < 0.01, (
        f"WBV in dataset does not match formula. Max error: {diff.max():.6f} "
        "(tolerance 0.01 accommodates np.round(wbv, 3) rounding artefact)"
    )


def test_wbv_coefficient_hct():
    """Verify Hct coefficient is exactly 0.12."""
    hct = np.array([1.0, 0.0])
    tp  = np.array([0.0, 0.0])
    wbv = compute_wbv(hct, tp)
    assert abs(wbv[0] - wbv[1] - 0.12) < 1e-9, "Hct coefficient must be 0.12"


def test_wbv_coefficient_tp():
    """Verify TP coefficient is exactly 0.17."""
    hct = np.array([0.0, 0.0])
    tp  = np.array([3.07, 2.07])   # difference of 1.0
    wbv = compute_wbv(hct, tp)
    assert abs(wbv[0] - wbv[1] - 0.17) < 1e-9, "TP coefficient must be 0.17"


def test_wbv_intercept_offset():
    """Verify the −2.07 offset in the TP term."""
    hct = np.array([0.0])
    tp  = np.array([2.07])
    wbv = compute_wbv(hct, tp)
    # 0.12 * 0 + 0.17 * (2.07 - 2.07) = 0
    assert abs(wbv[0]) < 1e-9, "WBV(hct=0, tp=2.07) should be 0"


def test_wbv_physiological_range():
    """WBV values should fall in a physiologically plausible range (≈3.5–6.0 mPa·s)."""
    import yaml
    with open("config/generator_null.yaml") as f:
        cfg = yaml.safe_load(f)
    df = generate(cfg, seed=2026, n=1500)
    assert df["wbv"].min() > 2.0
    assert df["wbv"].max() < 8.0
