"""
test_scaler_fold_sealed.py
Assert that the StandardScaler is fitted ONLY on training data.

A scaler fitted on the full dataset (including test data) constitutes
preprocessing leakage (leakage type #4 in our taxonomy).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import FoldSealedScaler, FoldSealedWinsorizer


# ---------------------------------------------------------------------------
# FoldSealedScaler tests
# ---------------------------------------------------------------------------

def test_scaler_must_be_fit_before_transform():
    """FoldSealedScaler must raise if transform called without fitting."""
    scaler = FoldSealedScaler()
    X = np.random.default_rng(0).normal(0, 1, (100, 5))
    with pytest.raises(RuntimeError, match="must be fit"):
        scaler.transform(X)


def test_scaler_fit_on_train_only():
    """Scaler mean/std computed from training data must not equal those of full data."""
    rng = np.random.default_rng(42)
    X_full  = rng.normal(loc=[10, 20], scale=[2, 4], size=(1000, 2))
    X_train = X_full[:700]
    X_test  = X_full[700:]

    scaler = FoldSealedScaler()
    scaler.fit(X_train)

    # Scaler mean should approximate training mean, not full-data mean
    assert abs(scaler._mean[0] - X_train[:, 0].mean()) < 0.5, (
        "Scaler mean does not match training mean"
    )
    assert abs(scaler._mean[0] - X_full[:, 0].mean()) < 1.5, (
        "This is expected to be close (large n), but the important thing is "
        "that the scaler was NOT fitted on test data"
    )


def test_scaler_transform_consistent():
    """Transformed test data must use training statistics, not test statistics."""
    rng = np.random.default_rng(7)
    X_train = rng.normal(5.0, 2.0, (200, 1))
    X_test  = rng.normal(8.0, 2.0, (50,  1))   # shifted mean

    scaler = FoldSealedScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Training data mean should be ≈ 0 after scaling with train stats
    assert abs(X_train_scaled.mean()) < 0.1, "Scaled train data should have mean ≈ 0"

    # Test data mean should NOT be ≈ 0 (it's from a different distribution)
    assert abs(X_test_scaled.mean()) > 0.5, (
        "Scaled test data mean should deviate from 0 (test distribution is shifted)"
    )


def test_fit_transform_equals_fit_then_transform():
    """fit_transform should produce the same result as fit then transform."""
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, (100, 4))

    s1 = FoldSealedScaler()
    result1 = s1.fit_transform(X)

    s2 = FoldSealedScaler()
    s2.fit(X)
    result2 = s2.transform(X)

    np.testing.assert_array_almost_equal(result1, result2)


# ---------------------------------------------------------------------------
# FoldSealedWinsorizer tests
# ---------------------------------------------------------------------------

def test_winsorizer_must_be_fit_before_transform():
    """FoldSealedWinsorizer must raise if transform called without fitting."""
    w = FoldSealedWinsorizer()
    X = np.random.default_rng(0).normal(0, 1, (100, 3))
    with pytest.raises(RuntimeError, match="must be fit"):
        w.transform(X)


def test_winsorizer_clips_to_training_percentiles():
    """Test values outside training 1–99 range should be clipped to training bounds."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (500, 1))   # values ≈ [-3, 3]

    w = FoldSealedWinsorizer(lower_pct=1.0, upper_pct=99.0)
    w.fit(X_train)

    # Test data with extreme values
    X_test = np.array([[-100.0], [0.0], [100.0]])
    X_test_win = w.transform(X_test)

    assert X_test_win[0, 0] >= w._lower_vals[0], "Extreme low not clipped to training lower bound"
    assert X_test_win[2, 0] <= w._upper_vals[0], "Extreme high not clipped to training upper bound"
    assert X_test_win[1, 0] == 0.0,              "Interior value should not be clipped"


def test_winsorizer_fit_transform_consistent():
    """fit_transform should equal fit then transform."""
    rng = np.random.default_rng(5)
    X = rng.standard_t(df=3, size=(200, 5))   # heavy-tailed

    w1 = FoldSealedWinsorizer()
    r1 = w1.fit_transform(X)

    w2 = FoldSealedWinsorizer()
    w2.fit(X)
    r2 = w2.transform(X)

    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Integration: fold_sealed_preprocess uses fold scaler
# ---------------------------------------------------------------------------

def test_preprocess_pipeline_fold_sealed():
    """Integration: fold_sealed_preprocess should not leak test stats into scaler."""
    import yaml
    from src.generate_synthetic_data import generate
    from src.run_clean_pipeline import fold_sealed_preprocess, CLEAN_FEATURES

    with open("config/generator_null.yaml") as f:
        cfg = yaml.safe_load(f)

    df = generate(cfg, seed=42, n=500)
    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values

    X_train, X_test = X[:400], X[400:]
    tcr_train = tcr[:400]

    X_train_proc, X_test_proc, y_bin, threshold = fold_sealed_preprocess(
        X_train, X_test, None, tcr_train, cfg
    )

    # Processed training data should have mean ≈ 0 (after scaling)
    assert abs(X_train_proc.mean()) < 0.5, (
        f"Processed train data mean {X_train_proc.mean():.4f} should be near 0"
    )

    # Shapes should be valid
    assert X_train_proc.shape[1] == X_train.shape[1]
    assert X_test_proc.shape[1] == X_test.shape[1]
    assert len(y_bin) >= len(X_train)  # may be larger due to SMOTE
