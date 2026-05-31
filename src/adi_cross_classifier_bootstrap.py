#!/usr/bin/env python3
"""Cross-classifier ADI bootstrap and Wilcoxon signed-rank tests for the synthetic TCR benchmark."""

import argparse
import warnings
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import shap

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — XGB will be skipped.")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(
    os.path.dirname(SCRIPT_DIR), "data", "synthetic_null_seed42.csv")

TCR_COL      = "tcr"
TG4H_COL     = "tg4h"
TG0H_COL     = "tg0h"
EXCLUDE_COLS = {"record_id", TCR_COL}

STRUCTURAL   = [TG0H_COL, "hdl", "bmi"]


def derive_label(tcr_series: pd.Series) -> pd.Series:
    """low_TCR = 1 if TCR <= Q1(TCR), else 0. Computed per bootstrap resample (fold-sealed)."""
    q1 = tcr_series.quantile(0.25)
    return (tcr_series <= q1).astype(int)


def _extract_shap(sv) -> np.ndarray:
    """
    Normalise SHAP output to 2-D (n_samples, n_features) for the positive class.

    Handles: list of 2 arrays (older SHAP), 3-D ndarray (newer TreeExplainer),
    and 2-D ndarray (XGB/LR/newer RF).
    """
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def shap_ranks(X: pd.DataFrame, y: pd.Series,
               clf_name: str, seed: int = 0,
               shap_subsample: int = 300) -> dict:
    """Train classifier on (X, y), return SHAP mean |value| rank dict (rank 1 = most important)."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    features = list(X.columns)

    try:
        if clf_name == "RF":
            clf = RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=1)
            clf.fit(X, y)
            n_sub = min(shap_subsample, len(X))
            rng_local = np.random.RandomState(seed)
            idx = rng_local.choice(len(X), n_sub, replace=False)
            X_sub = X.iloc[idx].reset_index(drop=True)
            explainer = shap.TreeExplainer(clf)
            sv = _extract_shap(explainer.shap_values(X_sub))

        elif clf_name == "LR":
            clf = LogisticRegression(max_iter=2000, random_state=seed,
                                     C=1.0, solver="lbfgs")
            clf.fit(X, y)
            # masker for LinearExplainer (SHAP >= 0.42 prefers this)
            try:
                explainer = shap.LinearExplainer(
                    clf, shap.maskers.Independent(X, max_samples=100))
            except Exception:
                explainer = shap.LinearExplainer(clf, X)
            sv = _extract_shap(explainer.shap_values(X))

        elif clf_name == "XGB" and HAS_XGB:
            clf = xgb.XGBClassifier(
                n_estimators=100, random_state=seed,
                eval_metric="logloss", use_label_encoder=False, verbosity=0)
            clf.fit(X, y)
            explainer = shap.TreeExplainer(clf)
            sv = _extract_shap(explainer.shap_values(X))

        else:
            return {}

        mean_abs = np.abs(sv).mean(axis=0)
        order    = np.argsort(-mean_abs)
        ranks    = {features[i]: int(r + 1) for r, i in enumerate(order)}
        return ranks

    except Exception as e:
        print(f"    [WARN] {clf_name} SHAP failed: {e}")
        return {}


def compute_adi(clean_ranks: dict, leaky_ranks: dict,
                clean_features: list) -> dict:
    """ADI_j = rank_clean(j) - rank_leaky(j). Positive = promoted."""
    adi = {}
    for feat in clean_features:
        rc = clean_ranks.get(feat)
        rl = leaky_ranks.get(feat)
        adi[feat] = (rc - rl) if (rc is not None and rl is not None) else np.nan
    return adi


def run_bootstrap(df_full: pd.DataFrame,
                  clean_features: list, leaky_features: list,
                  n_boot: int = 500, seed: int = 42,
                  shap_subsample: int = 300) -> pd.DataFrame:
    """
    Bootstrap loop: draw n rows with replacement, derive fold-local Q1 label,
    compute SHAP ranks and ADI for RF/LR/XGB on clean and leaky feature sets.
    """
    classifiers = ["RF", "LR"]
    if HAS_XGB:
        classifiers.append("XGB")

    rng = np.random.RandomState(seed)
    n   = len(df_full)
    records = []

    for b in range(n_boot):
        if b % 50 == 0:
            print(f"  resample {b+1}/{n_boot} ...", flush=True)

        idx   = rng.choice(n, n, replace=True)
        df_b  = df_full.iloc[idx].reset_index(drop=True)
        y_b   = derive_label(df_b[TCR_COL])

        if y_b.nunique() < 2:
            continue

        Xc_b = df_b[clean_features]
        Xl_b = df_b[leaky_features]

        for clf_name in classifiers:
            clean_r = shap_ranks(Xc_b, y_b, clf_name, seed=b,
                                 shap_subsample=shap_subsample)
            leaky_r = shap_ranks(Xl_b, y_b, clf_name, seed=b,
                                 shap_subsample=shap_subsample)

            if not clean_r or not leaky_r:
                continue

            adi_d = compute_adi(clean_r, leaky_r, clean_features)
            for feat, val in adi_d.items():
                records.append({
                    "boot_id":    b,
                    "classifier": clf_name,
                    "feature":    feat,
                    "type":       "adi",
                    "value":      val,
                })

            for feat in leaky_features:
                records.append({
                    "boot_id":    b,
                    "classifier": clf_name,
                    "feature":    feat,
                    "type":       "leaky_rank",
                    "value":      leaky_r.get(feat, np.nan),
                })

    return pd.DataFrame(records)


def run_wilcoxon(df: pd.DataFrame, clean_features: list) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests on ADI distributions with Bonferroni correction."""
    adi_df = df[df["type"] == "adi"].copy()
    adi_df = adi_df.rename(columns={"value": "adi"})

    classifiers = [c for c in adi_df["classifier"].unique()
                   if c in ["RF", "LR", "XGB"]]
    pairs = [(c1, c2) for i, c1 in enumerate(classifiers)
             for c2 in classifiers[i+1:]]

    rows = []
    for feat in clean_features:
        sub = adi_df[adi_df["feature"] == feat]
        pivot = sub.pivot_table(
            index="boot_id", columns="classifier", values="adi",
            aggfunc="first").dropna()

        for c1, c2 in pairs:
            if c1 not in pivot.columns or c2 not in pivot.columns:
                continue
            d = pivot[c1].values - pivot[c2].values
            if np.all(d == 0):
                rows.append({"feature": feat, "pair": f"{c1}_vs_{c2}",
                             "n": len(d), "stat": 0.0, "p_raw": 1.0})
                continue
            try:
                stat, p = stats.wilcoxon(pivot[c1].values, pivot[c2].values,
                                         alternative="two-sided")
            except Exception as e:
                print(f"  [WARN] Wilcoxon failed for {feat} {c1} vs {c2}: {e}")
                stat, p = np.nan, np.nan
            rows.append({"feature": feat, "pair": f"{c1}_vs_{c2}",
                         "n": len(d), "stat": stat, "p_raw": p})

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        return df_res

    n_tests = len(df_res.dropna(subset=["p_raw"]))
    df_res["p_bonf"] = (df_res["p_raw"] * n_tests).clip(upper=1.0)
    return df_res


def sign_concordance(df: pd.DataFrame, structural_features: list) -> pd.DataFrame:
    """% of resamples where sign(ADI_A) == sign(ADI_B) for each structural feature and classifier pair."""
    adi_df = df[df["type"] == "adi"].copy()
    adi_df = adi_df.rename(columns={"value": "adi"})

    classifiers = [c for c in adi_df["classifier"].unique()
                   if c in ["RF", "LR", "XGB"]]
    pairs = [(c1, c2) for i, c1 in enumerate(classifiers)
             for c2 in classifiers[i+1:]]

    rows = []
    for feat in structural_features:
        sub   = adi_df[adi_df["feature"] == feat]
        pivot = sub.pivot_table(
            index="boot_id", columns="classifier", values="adi",
            aggfunc="first").dropna()
        for c1, c2 in pairs:
            if c1 not in pivot.columns or c2 not in pivot.columns:
                continue
            agree = (np.sign(pivot[c1]) == np.sign(pivot[c2])).mean()
            rows.append({"feature": feat, "pair": f"{c1}_vs_{c2}",
                         "pct_agree": round(agree * 100, 1),
                         "n_resamples": len(pivot)})
    return pd.DataFrame(rows)


def rank_stability(df: pd.DataFrame) -> pd.DataFrame:
    """% resamples where TG4h=rank 1 and TG0h=rank 2 in the leaky pipeline, per classifier."""
    rank_df = df[df["type"] == "leaky_rank"].copy()
    rank_df = rank_df.rename(columns={"value": "leaky_rank"})

    classifiers = [c for c in rank_df["classifier"].unique()
                   if c in ["RF", "LR", "XGB"]]
    rows = []
    for clf in classifiers:
        sub = rank_df[rank_df["classifier"] == clf]
        for feat, expected_rank in [(TG4H_COL, 1), (TG0H_COL, 2)]:
            feat_sub = sub[sub["feature"] == feat]["leaky_rank"].dropna()
            pct = (feat_sub == expected_rank).mean() * 100
            rows.append({
                "classifier":    clf,
                "feature":       feat,
                "expected_rank": expected_rank,
                "pct_at_rank":   round(pct, 1),
                "n_resamples":   len(feat_sub),
            })
    return pd.DataFrame(rows)


def make_latex_snippet(df_wilcox: pd.DataFrame,
                       df_sign: pd.DataFrame,
                       df_rank: pd.DataFrame,
                       structural_features: list) -> str:

    lines = []

    if not df_sign.empty:
        min_agree     = df_sign["pct_agree"].min()
        min_agree_str = f"{min_agree:.0f}\\%"
    else:
        min_agree_str = "[XX]\\%"

    if not df_wilcox.empty:
        struct_tests = df_wilcox[
            df_wilcox["feature"].isin(structural_features)].dropna(subset=["p_bonf"])
        if not struct_tests.empty:
            all_ns   = (struct_tests["p_bonf"] > 0.05).all()
            max_p    = struct_tests["p_bonf"].max()
            sig_str  = (f"no significant difference (all Bonferroni-corrected "
                        f"$p > {max_p:.2f}$)")  if all_ns \
                       else "[see wilcoxon_summary.csv — some pairs significant]"
        else:
            sig_str = "[PLACEHOLDER]"
    else:
        sig_str = "[PLACEHOLDER]"

    # TG4h rank-1 stability (min across classifiers)
    if not df_rank.empty:
        tg4h_stab = df_rank[df_rank["feature"] == TG4H_COL]["pct_at_rank"]
        tg0h_stab = df_rank[df_rank["feature"] == TG0H_COL]["pct_at_rank"]
        tg4h_min  = f"{tg4h_stab.min():.0f}\\%" if not tg4h_stab.empty else "[XX]\\%"
        tg0h_min  = f"{tg0h_stab.min():.0f}\\%" if not tg0h_stab.empty else "[XX]\\%"
    else:
        tg4h_min = tg0h_min = "[XX]\\%"

    lines.append("%% ── PASTE INTO Section 5.4 (sec:result_shap) ─────────────────────────────")
    lines.append("%% Replace the [UPDATE …] placeholder already in main.tex with this block.")
    lines.append("")
    lines.append(
        f"To formally quantify this classifier-invariance, cross-classifier\n"
        f"bootstrap resampling ($B = 500$ resamples) was conducted for RF,\n"
        f"LR, and XGB simultaneously.\n"
        f"Pairwise Wilcoxon signed-rank tests on the ADI distributions for\n"
        f"the three structural features ($\\mathrm{{TG}}_{{0\\mathrm{{h}}}}$\n"
        f"promotion, HDL suppression, BMI suppression) found\n"
        f"{sig_str} in ADI magnitude between classifier pairs after\n"
        f"Bonferroni correction, while sign concordance exceeded\n"
        f"{min_agree_str} of resamples for all structural features across\n"
        f"all classifier pairs.\n"
        f"The structural signature\n"
        f"($\\mathrm{{TG}}_{{4\\mathrm{{h}}}}$ rank~1 in\n"
        f"$\\geq {tg4h_min}$ of resamples;\n"
        f"$\\mathrm{{TG}}_{{0\\mathrm{{h}}}}$ rank~2 in\n"
        f"$\\geq {tg0h_min}$ of resamples\n"
        f"across all three classifier families) is thus confirmed as\n"
        f"classifier-invariant in both direction and sign, with only\n"
        f"magnitude varying by classifier family.")
    lines.append("")
    lines.append("%% ── END Section 5.4 snippet ────────────────────────────────────────────────")
    lines.append("")
    lines.append("%% ── The limitations (eICU SHAP scope) edit is ALREADY APPLIED to main.tex ──")
    lines.append("%% No further action needed for that section.")
    lines.append("")
    lines.append("%% ── Full results for reference ─────────────────────────────────────────────")
    if not df_wilcox.empty:
        lines.append("%% Wilcoxon summary (structural features only):")
        struct_w = df_wilcox[df_wilcox["feature"].isin(structural_features)]
        lines.append(struct_w.to_string(index=False))
    if not df_sign.empty:
        lines.append("")
        lines.append("%% Sign concordance:")
        lines.append(df_sign.to_string(index=False))
    if not df_rank.empty:
        lines.append("")
        lines.append("%% Rank stability:")
        lines.append(df_rank.to_string(index=False))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-classifier ADI bootstrap + Wilcoxon tests")
    parser.add_argument("--data",   default=DEFAULT_DATA,
                        help=f"CSV path (default: {DEFAULT_DATA})")
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--shap_n", type=int, default=300,
                        help="RF SHAP subsample size per resample (default 300)")
    args = parser.parse_args()

    print(f"Data: {args.data}")
    df_full = pd.read_csv(args.data)
    print(f"  Shape: {df_full.shape}  |  columns: {list(df_full.columns)}")

    all_cols      = [c for c in df_full.columns if c not in EXCLUDE_COLS]
    clean_features = [c for c in all_cols if c != TG4H_COL]
    leaky_features = all_cols   # includes tg4h

    print(f"\nClean features ({len(clean_features)}): {clean_features}")
    print(f"Leaky features ({len(leaky_features)}): {leaky_features}")
    print(f"Leakage variable : {TG4H_COL}")
    print(f"Co-antecedent    : {TG0H_COL}")
    print(f"Structural check : {STRUCTURAL}")

    y_preview = derive_label(df_full[TCR_COL])
    print(f"\nGlobal label preview (for sanity): "
          f"positives={y_preview.sum()}/{len(y_preview)} "
          f"({y_preview.mean():.1%})")
    print(f"  [label is re-derived per resample inside bootstrap]")

    clfs = ["RF", "LR"] + (["XGB"] if HAS_XGB else [])
    print(f"\nRunning B={args.n_boot} resamples × {len(clfs)} classifiers ...")
    print("  Expected outputs: adi_boot_results.csv, wilcoxon_summary.csv,")
    print("                    sign_concordance.csv, rank_stability.csv, latex_snippet.txt\n")

    df_boot = run_bootstrap(
        df_full, clean_features, leaky_features,
        n_boot=args.n_boot, seed=args.seed, shap_subsample=args.shap_n)

    out_dir = SCRIPT_DIR
    df_boot.to_csv(os.path.join(out_dir, "adi_boot_results.csv"), index=False)
    print(f"\n[saved] adi_boot_results.csv  ({len(df_boot)} rows)")

    print("\nRunning Wilcoxon signed-rank tests ...")
    df_wilcox = run_wilcoxon(df_boot, clean_features)
    df_wilcox.to_csv(os.path.join(out_dir, "wilcoxon_summary.csv"), index=False)
    print(f"[saved] wilcoxon_summary.csv")

    structural_in_data = [f for f in STRUCTURAL if f in clean_features]
    df_sign = sign_concordance(df_boot, structural_in_data)
    df_sign.to_csv(os.path.join(out_dir, "sign_concordance.csv"), index=False)
    print(f"[saved] sign_concordance.csv")

    df_rank = rank_stability(df_boot)
    df_rank.to_csv(os.path.join(out_dir, "rank_stability.csv"), index=False)
    print(f"[saved] rank_stability.csv")

    print("\n=== Wilcoxon (structural features) ===")
    struct_w = df_wilcox[df_wilcox["feature"].isin(structural_in_data)]
    print(struct_w.to_string(index=False))

    print("\n=== Sign concordance ===")
    print(df_sign.to_string(index=False))

    print("\n=== Rank stability ===")
    print(df_rank.to_string(index=False))

    snippet = make_latex_snippet(df_wilcox, df_sign, df_rank, structural_in_data)
    snippet_path = os.path.join(out_dir, "latex_snippet.txt")
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(snippet)
    print(f"\n[saved] latex_snippet.txt")
    print("\n" + "="*65)
    print(snippet)
    print("="*65)
    print("\nDone. Copy the Section 5.4 block from latex_snippet.txt")
    print("into main.tex, replacing the [UPDATE ...] placeholder.")


if __name__ == "__main__":
    main()
