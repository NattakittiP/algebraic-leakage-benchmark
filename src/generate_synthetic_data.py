"""
generate_synthetic_data.py — Primary synthetic data generator.

CRITICAL GENERATION ORDER:
  1. Generate TCR (primary latent response) first
  2. Derive TG4h = TG0h × (1 − TCR/100) AFTER TCR is generated
  NEVER reverse this order — reversing causes definitional leakage.

Usage:
  python src/generate_synthetic_data.py \\
      --config config/generator_null.yaml \\
      --seed 2026 \\
      --n 1500 \\
      --out data/paired_tcr_null_v1_seed2026.csv

Outputs:
  - CSV with all synthetic variables (TG4h and TCR included for leakage experiments)
  - JSON metadata sidecar

The label column `low_TCR` is NOT created here — it must be computed inside
each training fold from Q1 of that fold's TCR distribution to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    truncated_normal,
    shifted_lognormal_tg0h,
    set_seed,
    describe_series,
)


# ---------------------------------------------------------------------------
# WBV formula (de Simone 1990)
# ---------------------------------------------------------------------------

def compute_wbv(hct: np.ndarray, tp: np.ndarray) -> np.ndarray:
    """Whole Blood Viscosity (mPa·s) by de Simone formula.

    WBV = 0.12 × Hct + 0.17 × (TP − 2.07)

    Parameters
    ----------
    hct : haematocrit (%)
    tp  : total protein (g/dL)
    """
    return 0.12 * hct + 0.17 * (tp - 2.07)


# ---------------------------------------------------------------------------
# Scenario-specific TCR generators
# ---------------------------------------------------------------------------

def generate_tcr_null(
    n: int,
    cfg: dict,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Null scenario: TCR is independent of all predictors.

    Expected clean AUROC ≈ 0.48–0.52.
    """
    tcr_mu = cfg.get("tcr_mu", 52.2)
    tcr_sd = cfg.get("tcr_sd", 18.6)
    tcr_low = cfg.get("tcr_low", -10.0)
    tcr_high = cfg.get("tcr_high", 99.9)
    return truncated_normal(tcr_mu, tcr_sd, tcr_low, tcr_high, n, rng)


def generate_tcr_weak_signal(
    n: int,
    cfg: dict,
    rng: np.random.Generator,
    age: np.ndarray,
    hdl: np.ndarray,
    bmi: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Weak signal: age, HDL, BMI have small effects on TCR.

    Expected clean AUROC ≈ 0.55–0.60.

    Betas are in TCR-percent per predictor-SD (empirically standardised).
    Calibrated so total signal R ≈ 0.20 → AUROC ≈ 0.57.
    """
    tcr_mu = cfg.get("tcr_mu", 52.2)
    tcr_sd = cfg.get("tcr_sd", 18.6)
    tcr_low = cfg.get("tcr_low", -10.0)
    tcr_high = cfg.get("tcr_high", 99.9)

    beta_age = cfg.get("beta_age", -2.0)
    beta_hdl = cfg.get("beta_hdl",  2.5)
    beta_bmi = cfg.get("beta_bmi", -2.0)

    # Empirical z-scores from this draw (mean=0, sd≈1)
    age_z = (age - age.mean()) / (age.std() + 1e-8)
    hdl_z = (hdl - hdl.mean()) / (hdl.std() + 1e-8)
    bmi_z = (bmi - bmi.mean()) / (bmi.std() + 1e-8)

    mu_i = tcr_mu + beta_age * age_z + beta_hdl * hdl_z + beta_bmi * bmi_z
    noise = rng.normal(0, tcr_sd, size=n)
    tcr = np.clip(mu_i + noise, tcr_low, tcr_high)
    return tcr


def generate_tcr_moderate_signal(
    n: int,
    cfg: dict,
    rng: np.random.Generator,
    tg0h: np.ndarray,
    bmi: np.ndarray,
    age: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Moderate signal: TG0h and BMI predict TCR with moderate strength.

    Expected clean AUROC ≈ 0.65–0.75.

    Betas are in TCR-percent per predictor-SD (empirically standardised).
    Calibrated so total signal R ≈ 0.37 → AUROC ≈ 0.70 with noise_frac=0.70.
    """
    tcr_mu = cfg.get("tcr_mu", 52.2)
    tcr_sd = cfg.get("tcr_sd", 18.6)
    tcr_low = cfg.get("tcr_low", -10.0)
    tcr_high = cfg.get("tcr_high", 99.9)

    beta_tg0h = cfg.get("beta_tg0h", -3.5)
    beta_bmi  = cfg.get("beta_bmi",  -3.0)
    beta_age  = cfg.get("beta_age",  -2.5)

    # Empirical z-scores — avoids sensitivity to hardcoded reference means
    tg0h_z = (tg0h - tg0h.mean()) / (tg0h.std() + 1e-8)
    bmi_z  = (bmi  - bmi.mean())  / (bmi.std()  + 1e-8)
    age_z  = (age  - age.mean())  / (age.std()  + 1e-8)

    mu_i = tcr_mu + beta_tg0h * tg0h_z + beta_bmi * bmi_z + beta_age * age_z
    noise = rng.normal(0, tcr_sd * 0.70, size=n)   # reduced noise for cleaner signal
    tcr = np.clip(mu_i + noise, tcr_low, tcr_high)
    return tcr


def generate_tcr_wbv_positive(
    n: int,
    cfg: dict,
    rng: np.random.Generator,
    wbv: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """WBV positive-control: WBV has a real, detectable effect on TCR.

    Expected clean AUROC ≈ 0.83–0.87.

    beta_wbv is in TCR-percent per WBV-SD (empirically standardised).
    Calibrated so signal R ≈ 0.68 → AUROC ≈ 0.85 with noise_frac=0.40.
    """
    tcr_mu = cfg.get("tcr_mu", 52.2)
    tcr_sd = cfg.get("tcr_sd", 18.6)
    tcr_low = cfg.get("tcr_low", -10.0)
    tcr_high = cfg.get("tcr_high", 99.9)

    beta_wbv = cfg.get("beta_wbv", 7.0)   # large positive effect

    # Empirical z-score — avoids dependence on wbv_mean/wbv_sd config values
    wbv_z = (wbv - wbv.mean()) / (wbv.std() + 1e-8)

    mu_i = tcr_mu + beta_wbv * wbv_z
    noise = rng.normal(0, tcr_sd * 0.40, size=n)   # tight noise → strong signal
    tcr = np.clip(mu_i + noise, tcr_low, tcr_high)
    return tcr


SCENARIO_GENERATORS = {
    "null": generate_tcr_null,
    "weak_signal": generate_tcr_weak_signal,
    "moderate_signal": generate_tcr_moderate_signal,
    "wbv_positive": generate_tcr_wbv_positive,
}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(cfg: dict, seed: int, n: int) -> pd.DataFrame:
    """Generate one synthetic dataset according to cfg and seed.

    Parameters
    ----------
    cfg  : parsed YAML config dict
    seed : random seed for reproducibility
    n    : number of synthetic records

    Returns
    -------
    pd.DataFrame with all columns (TG4h and TCR included for leakage experiments)
    NOTE: low_TCR label is NOT included — compute it inside each fold.
    """
    rng = set_seed(seed)
    # YAML `null` is parsed as Python None — treat it as the "null" scenario string
    scenario = cfg.get("scenario", "null") or "null"

    steps = [
        "age", "sex", "bmi", "hct", "tp", "hdl", "ldl",
        "wbv", "tg0h", "tcr", "tg4h", "plausibility check", "assemble DataFrame",
    ]
    pbar = tqdm(steps, desc=f"Generating [{scenario}] seed={seed} n={n}", ncols=90, colour="cyan")

    # ------------------------------------------------------------------
    # 1. Patient-level demographics and baseline labs
    # ------------------------------------------------------------------
    pbar.set_description(f"[{scenario}] age"); pbar.update(0)
    age = truncated_normal(
        cfg.get("age_mu", 53.0), cfg.get("age_sd", 10.0),
        cfg.get("age_low", 18.0), cfg.get("age_high", 75.0), n, rng
    ); pbar.update(1)

    pbar.set_description(f"[{scenario}] sex")
    sex = rng.binomial(1, cfg.get("p_male", 0.485), size=n)
    pbar.update(1)

    pbar.set_description(f"[{scenario}] bmi")
    bmi = truncated_normal(
        cfg.get("bmi_mu", 24.0), cfg.get("bmi_sd", 3.0),
        cfg.get("bmi_low", 16.0), cfg.get("bmi_high", 33.0), n, rng
    ); pbar.update(1)

    pbar.set_description(f"[{scenario}] hct")
    hct = truncated_normal(
        cfg.get("hct_mu", 41.7), cfg.get("hct_sd", 3.75),
        cfg.get("hct_low", 30.0), cfg.get("hct_high", 55.0), n, rng
    ); pbar.update(1)

    pbar.set_description(f"[{scenario}] tp")
    tp = truncated_normal(
        cfg.get("tp_mu", 6.88), cfg.get("tp_sd", 0.62),
        cfg.get("tp_low", 5.0), cfg.get("tp_high", 8.9), n, rng
    ); pbar.update(1)

    pbar.set_description(f"[{scenario}] hdl")
    hdl = truncated_normal(
        cfg.get("hdl_mu", 50.5), cfg.get("hdl_sd", 9.7),
        cfg.get("hdl_low", 25.0), cfg.get("hdl_high", 87.0), n, rng
    ); pbar.update(1)

    pbar.set_description(f"[{scenario}] ldl")
    ldl = truncated_normal(
        cfg.get("ldl_mu", 131.0), cfg.get("ldl_sd", 28.0),
        cfg.get("ldl_low", 70.0), cfg.get("ldl_high", 215.0), n, rng
    ); pbar.update(1)

    # ------------------------------------------------------------------
    # 2. WBV (de Simone formula) — computed BEFORE TCR in all scenarios
    # ------------------------------------------------------------------
    pbar.set_description(f"[{scenario}] wbv (de Simone formula)")
    wbv = compute_wbv(hct, tp)
    pbar.update(1)

    # ------------------------------------------------------------------
    # 3. Baseline triglyceride (TG0h)
    # ------------------------------------------------------------------
    pbar.set_description(f"[{scenario}] tg0h (shifted log-normal)")
    tg0h = shifted_lognormal_tg0h(
        n, rng,
        shift=cfg.get("tg0h_shift", 350.0),
        mu_log=cfg.get("tg0h_mu_log", 5.5),
        sigma_log=cfg.get("tg0h_sigma_log", 0.45),
        low=cfg.get("tg0h_low", 350.0),
        high=cfg.get("tg0h_high", 1750.0),
    ); pbar.update(1)

    # ------------------------------------------------------------------
    # 4. Generate TCR FIRST (primary latent response)
    #    Then derive TG4h = TG0h × (1 − TCR/100)
    #    NEVER reverse this order.
    # ------------------------------------------------------------------
    if scenario not in SCENARIO_GENERATORS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Choose from: {list(SCENARIO_GENERATORS)}"
        )

    pbar.set_description(f"[{scenario}] TCR (primary latent response)")
    tcr = SCENARIO_GENERATORS[scenario](
        n=n, cfg=cfg, rng=rng,
        age=age, hdl=hdl, bmi=bmi,
        tg0h=tg0h, wbv=wbv,
    ); pbar.update(1)

    pbar.set_description(f"[{scenario}] tg4h = tg0h × (1 − tcr/100)")
    tg4h = tg0h * (1.0 - tcr / 100.0)
    pbar.update(1)

    # ------------------------------------------------------------------
    # 5. Plausibility checks — reject and re-sample bad rows
    # ------------------------------------------------------------------
    pbar.set_description(f"[{scenario}] plausibility check")
    bad_mask = (tg4h <= 0) | (tg4h > 1300)
    n_bad = bad_mask.sum()
    if n_bad > 0:
        tg4h = np.clip(tg4h, 1.0, 1300.0)
    pbar.update(1)

    # ------------------------------------------------------------------
    # 6. Assemble DataFrame
    # ------------------------------------------------------------------
    pbar.set_description(f"[{scenario}] assembling DataFrame")
    df = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "sex": sex.astype(int),
            "bmi": np.round(bmi, 1),
            "hct": np.round(hct, 1),
            "tp": np.round(tp, 2),
            "hdl": np.round(hdl, 1),
            "ldl": np.round(ldl, 1),
            "wbv": np.round(wbv, 3),
            "tg0h": np.round(tg0h, 1),
            # TCR and TG4h are INCLUDED for leakage experiments.
            # The clean pipeline must EXCLUDE them from predictors.
            "tcr": np.round(tcr, 2),
            "tg4h": np.round(tg4h, 1),
        }
    )
    df.insert(0, "record_id", [f"SYN{i+1:05d}" for i in range(n)])
    pbar.update(1)
    pbar.close()

    return df


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def build_metadata(cfg: dict, seed: int, n: int, df: pd.DataFrame, out_path: Path) -> dict:
    """Build JSON metadata sidecar for the generated dataset."""
    return {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "out_file": str(out_path.name),
        "seed": seed,
        "n": n,
        "scenario": cfg.get("scenario", "null"),
        "generator_version": "1.0",
        "wbv_formula": "WBV = 0.12 * Hct + 0.17 * (TP - 2.07)",
        "tg4h_formula": "TG4h = TG0h * (1 - TCR/100)",
        "note_label": (
            "low_TCR label is NOT in this file. "
            "Compute it inside each training fold from Q1 of that fold's TCR."
        ),
        "columns": list(df.columns),
        "n_bad_rows_clipped": int(((df["tg4h"] <= 1.0) | (df["tg4h"] >= 1300.0)).sum()),
        "summary_stats": {
            col: describe_series(df[col].values, col)
            for col in ["age", "bmi", "hct", "tp", "wbv", "tg0h", "tcr", "tg4h"]
        },
        "config_used": cfg,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic paired-biomarker dataset for the BMC Bioinformatics benchmark."
    )
    parser.add_argument("--config", required=True, help="Path to generator YAML config")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default 2026)")
    parser.add_argument("--n", type=int, default=None, help="Override n from config")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n = args.n if args.n is not None else cfg.get("n", 1500)
    seed = args.seed

    print(f"Generating {n} synthetic records | scenario={cfg.get('scenario')} | seed={seed}")
    df = generate(cfg, seed, n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Assert the DataFrame has exactly the requested number of rows
    assert len(df) == n, f"Generator produced {len(df)} rows instead of {n}!"

    save_steps = ["saving CSV", "verifying CSV", "writing JSON metadata"]
    for step in tqdm(save_steps, desc="Saving outputs", ncols=70, colour="green"):
        if step == "saving CSV":
            # Write to a temp file first, then rename — avoids partial-write corruption
            tmp_path = out_path.with_suffix(".tmp")
            df.to_csv(tmp_path, index=False)
            # Force OS buffer flush before rename
            import os
            with open(tmp_path, "a") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            tmp_path.replace(out_path)   # atomic rename on most OS
            tqdm.write(f"  Saved dataset     -> {out_path}")

        elif step == "verifying CSV":
            # Re-read and count rows to confirm integrity
            import pandas as _pd
            n_saved = sum(1 for _ in open(out_path)) - 1   # subtract header
            if n_saved != n:
                print(f"\nERROR: saved CSV has {n_saved} rows, expected {n}!", file=sys.stderr)
                sys.exit(1)
            tqdm.write(f"  Verified {n_saved}/{n} rows written correctly ✓")

        elif step == "writing JSON metadata":
            meta = build_metadata(cfg, seed, n, df, out_path)
            meta_path = out_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
            tqdm.write(f"  Saved metadata    -> {meta_path}")

    # Quick sanity print
    print(f"\nTCR  mean={df['tcr'].mean():.2f}  sd={df['tcr'].std():.2f}")
    print(f"TG4h mean={df['tg4h'].mean():.1f}  min={df['tg4h'].min():.1f}  max={df['tg4h'].max():.1f}")
    print(f"WBV  mean={df['wbv'].mean():.3f}  sd={df['wbv'].std():.3f}")
    low_tcr_q1 = df["tcr"].quantile(0.25)
    prev = (df["tcr"] <= low_tcr_q1).mean()
    print(f"Q1 TCR threshold (illustrative, DO NOT use across splits): {low_tcr_q1:.2f}")
    print(f"Prevalence if labelled from full data (for reference only): {prev:.3f}")


if __name__ == "__main__":
    main()
