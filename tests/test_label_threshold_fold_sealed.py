"""
test_label_threshold_fold_sealed.py
Assert that Q1 TCR label threshold is computed ONLY from training fold data.

Global threshold computation (from the full dataset before splitting) is
leakage type #6 in our taxonomy and must never appear in the clean pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import fold_sealed_label_threshold
from src.run_clean_pipeline import fold_sealed_preprocess, CLEAN_FEATURES


def test_fold_sealed_threshold_differs_from_global():
    """Threshold computed on a training fold must differ from the global threshold."""
    rng = np.random.default_rng(42)
    n = 1000
    tcr_full = rng.normal(52.2, 18.6, n)

    # Simulate a fold split: 80% train, 20% test
    n_train = int(0.8 * n)
    tcr_train = tcr_full[:n_train]
    tcr_test  = tcr_full[n_train:]

    threshold_global = fold_sealed_label_threshold(tcr_full)
    threshold_fold   = fold_sealed_label_threshold(tcr_train)

    # They should differ (different data, different Q1)
    assert threshold_global != threshold_fold, (
        "Global and fold-based thresholds are identical — "
        "this indicates the function is not respecting fold boundaries."
    )


def test_threshold_computed_from_train_only():
    """Threshold computed from train fold must equal np.percentile(tcr_train, 25)."""
    rng = np.random.default_rng(99)
    tcr_train = rng.normal(52.0, 15.0, 800)

    expected  = float(np.percentile(tcr_train, 25.0))
    computed  = fold_sealed_label_threshold(tcr_train, q=25.0)

    assert abs(computed - expected) < 1e-10, (
        f"fold_sealed_label_threshold returned {computed:.4f}, expected {expected:.4f}"
    )


def test_threshold_not_using_test_data():
    """Confirm that adding test data to the threshold computation changes the result."""
    rng = np.random.default_rng(7)
    tcr_train = rng.normal(52.0, 18.0, 800)
    tcr_test  = rng.normal(52.0, 18.0, 200)

    threshold_train_only = fold_sealed_label_threshold(tcr_train)
    threshold_all_data   = fold_sealed_label_threshold(np.concatenate([tcr_train, tcr_test]))

    assert abs(threshold_train_only - threshold_all_data) > 0.001, (
        "Training-only and full-data thresholds are nearly identical — "
        "test data appears to have no effect (expected small difference)."
    )


def test_fold_sealed_preprocess_uses_training_tcr_for_label():
    """Integration test: fold_sealed_preprocess returns threshold from train TCR."""
    rng = np.random.default_rng(42)
    import yaml
    with open("config/generator_null.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.generate_synthetic_data import generate
    df = generate(cfg, seed=42, n=400)

    X = df[CLEAN_FEATURES].values
    tcr = df["tcr"].values

    n_train = 300
    X_train, X_test = X[:n_train], X[n_train:]
    tcr_train, tcr_test = tcr[:n_train], tcr[n_train:]

    _, _, y_train_bin, threshold_from_fold = fold_sealed_preprocess(
        X_train, X_test, None, tcr_train, cfg
    )

    expected_threshold = float(np.percentile(tcr_train, cfg.get("label_threshold_percentile", 25.0)))
    assert abs(threshold_from_fold - expected_threshold) < 1e-6, (
        f"fold_sealed_preprocess threshold {threshold_from_fold:.4f} != "
        f"expected {expected_threshold:.4f}"
    )

    # Verify label encoding matches the fold threshold
    # After SMOTE the minority class is upsampled to match the majority →
    # balanced output (mean ≈ 0.50).  Accept any reasonable post-SMOTE range.
    assert 0.40 <= y_train_bin.mean() <= 0.60, (
        f"Label prevalence {y_train_bin.mean():.3f} is unreasonable post-SMOTE — "
        "expected ≈ 0.50 after balancing (Q1 threshold ≈ 25% pre-SMOTE)"
    )
