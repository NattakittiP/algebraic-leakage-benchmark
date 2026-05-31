#!/usr/bin/env python3
"""ICU glucose algebraic leakage audit (v2) — external corroboration of Theorem 2."""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tqdm import tqdm

warnings.filterwarnings("ignore")


CFG = {
    "random_seed":         42,
    "n_cv_splits":         5,
    "primary_window_h":    6,
    "sensitivity_windows": [4, 6, 8],
    "label_quantile":      0.75,        # Q75 → "large glucose drop"
    "insulin_type":        "Short",
    # "BOLUS_INYECTION" is a typo in the raw data (confirmed by audit print).
    "bolus_events":        {"BOLUS_INYECTION", "BOLUS_PUSH"},
}

# Fixed n_estimators — no early stopping (reproducible, no val-set data leakage)
LGBM_PARAMS = {
    "objective":          "binary",
    "metric":             "auc",
    "n_estimators":       300,
    "learning_rate":      0.05,
    "num_leaves":         31,
    "min_child_samples":  30,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "random_state":       42,
    "verbose":            -1,
    "n_jobs":             -1,
}

# REMOVED: LOS_ICU_days  (total stay length — only known after discharge)
# ADDED  : elapsed_icu_days (event_time − ICU admission proxy — known at event time)
CLEAN_FEATURES = [
    "GLC_pre",           # pre-insulin glucose (= GLC_AL in raw data), mg/dL
    "INPUT",             # insulin dose, units
    "EVENT_enc",         # 0 = BOLUS_INYECTION, 1 = BOLUS_PUSH
    "first_ICU_stay",    # 1 if this is patient's first ICU admission
    "elapsed_icu_days",  # days from ICU admission proxy to this insulin event
    "hour_of_day",       # clock hour 0–23 of insulin event (circadian proxy)
    "glcsource_enc",     # 0 = BLOOD, 1 = FINGERSTICK (source of pre-glucose)
]

PIPELINE_DEFS = {
    "clean": {
        "label":       "Clean",
        "description": "Pre-event features only — legitimate baseline",
        "extra":       [],
        "models":      ["lgbm", "lr", "rf"],
    },
    "b_component": {
        "label":       "+G_post (B-component leakage)",
        "description": "Clean + G_post — algebraic leakage per Theorem 2",
        "extra":       ["GLC_post"],
        "models":      ["lgbm", "lr", "rf"],
    },
    "target_leaky": {
        "label":       "+glucose_drop (target leakage)",
        "description": "Clean + glucose_drop = G_pre − G_post — near-oracle via model",
        "extra":       ["glucose_drop"],
        "models":      ["lgbm", "lr", "rf"],
    },
    "oracle": {
        "label":       "Oracle (score = glucose_drop)",
        "description": "Direct algebraic score, no ML model — exact upper bound",
        "extra":       [],
        "models":      ["oracle"],
    },
}


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the mixed glucose–insulin event table.

    TIMER column note [Fix #1]:
      TIMER is a unified event timestamp, 100% complete for all rows.
        - Insulin rows : TIMER ≡ STARTTIME  (same value, confirmed by inspection)
        - Glucose rows : TIMER ≡ GLCTIMER   (same value, confirmed by inspection)
      We use TIMER throughout as the canonical event time for both row types.
      This was correct in v1; this comment documents it explicitly for reviewers.

    Bolus event audit [Fix #6]:
      Unique EVENT values are printed so the bolus_events filter can be verified
      without running a separate exploration notebook.
    """
    print(f"\n[1] Loading data:  {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ["TIMER", "STARTTIME", "ENDTIME", "GLCTIMER", "GLCTIMER_AL"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    print(f"    Total rows : {len(df):>10,}")
    print(f"    Patients   : {df['SUBJECT_ID'].nunique():>10,}")
    print(f"    ICU stays  : {df['ICUSTAY_ID'].nunique():>10,}")

    ins_mask = df["STARTTIME"].notna() & df["INPUT"].notna()
    print("\n    ── Insulin EVENT audit (all unique values in raw data) ──")
    for evt, cnt in df.loc[ins_mask, "EVENT"].value_counts().items():
        tag = "  ← selected" if evt in CFG["bolus_events"] else ""
        print(f"      {evt:35s} {cnt:>8,}{tag}")

    return df


def separate_event_types(df: pd.DataFrame):
    """
    The raw CSV is a mixed event table.
      Insulin rows : STARTTIME + INPUT are filled; GLC columns are null.
      Glucose rows : GLC is filled; insulin columns are null.
    These two types are mutually exclusive in the original curation.
    """
    print("\n[2] Separating event types")
    ins_mask = df["STARTTIME"].notna() & df["INPUT"].notna()
    insulin_df = df[ins_mask].copy()
    glucose_df = df[~ins_mask & df["GLC"].notna()].copy()
    print(f"    Insulin rows : {len(insulin_df):,}")
    print(f"    Glucose rows : {len(glucose_df):,}")
    return insulin_df, glucose_df


def build_short_bolus_cohort(insulin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary cohort: Short-acting insulin, bolus delivery, with aligned
    pre-insulin glucose (GLC_AL non-null).

    Long/Intermediate insulin and infusions are excluded because their
    pharmacodynamic window exceeds 4–8h, making a single paired
    pre/post observation uninterpretable as a unit response.
    """
    print("\n[3] Filtering to Short bolus cohort (primary)")
    mask = (
        insulin_df["INSULINTYPE"].eq(CFG["insulin_type"])
        & insulin_df["EVENT"].isin(CFG["bolus_events"])
        & insulin_df["GLC_AL"].notna()
    )
    cohort = insulin_df[mask].copy()
    print(f"    Short bolus + GLC_pre : {len(cohort):,} events")
    print(f"    Unique patients       : {cohort['SUBJECT_ID'].nunique():,}")
    print(f"    Unique ICU stays      : {cohort['ICUSTAY_ID'].nunique():,}")
    return cohort


def compute_icu_admit_times(raw_df: pd.DataFrame) -> dict:
    """
    Approximate ICU admission time as the earliest TIMER within each ICUSTAY_ID
    across all events (glucose + insulin).

    This proxy is used to compute elapsed_icu_days for each insulin event:
      elapsed_icu_days = (event_TIMER − admission_proxy) / 1 day

    Always computable at the time of the insulin event (it is the past),
    unlike LOS_ICU_days which requires knowing the discharge time.
    """
    return raw_df.groupby("ICUSTAY_ID")["TIMER"].min().to_dict()


def attach_post_glucose(
    cohort: pd.DataFrame,
    glucose_df: pd.DataFrame,
    window_hours: float,
) -> pd.DataFrame:
    """
    For each insulin event, find the FIRST glucose measurement from the same
    patient AND the same ICU stay that occurs strictly after the insulin event
    and within `window_hours`.

    Fixes applied:
      [1] TIMER is the canonical event time for both row types.
      [2] by=["SUBJECT_ID","ICUSTAY_ID"]: prevents cross-stay pairing for
          patients with multiple admissions.

    1-second shift: ~14% of insulin events share an exact timestamp with a
    glucose reading (simultaneous charting). Without the shift, merge_asof
    (direction='forward') would pick up that same-time glucose as G_post.
    Shifting the lookup time by 1s ensures strictly post-event matching.

    merge_asof sort requirement: both left and right must be sorted by the
    key column GLOBALLY (not just within groups). We sort by _TIMER_shifted
    and TIMER_post only (not by [by_col, key_col]).
    """
    window_td = pd.Timedelta(hours=window_hours)

    glc = (
        glucose_df[["SUBJECT_ID", "ICUSTAY_ID", "TIMER", "GLC", "GLCSOURCE"]]
        .copy()
        .rename(columns={
            "TIMER":     "TIMER_post",
            "GLC":       "GLC_post",
            "GLCSOURCE": "GLCSOURCE_post",
        })
        .sort_values("TIMER_post")   # globally sorted by key — required by merge_asof
        .reset_index(drop=True)
    )

    cohort_sorted = cohort.copy()
    cohort_sorted["_TIMER_shifted"] = cohort_sorted["TIMER"] + pd.Timedelta(seconds=1)
    cohort_sorted = (
        cohort_sorted
        .sort_values("_TIMER_shifted")   # globally sorted by key — required by merge_asof
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        cohort_sorted,
        glc,
        left_on="_TIMER_shifted",
        right_on="TIMER_post",
        by=["SUBJECT_ID", "ICUSTAY_ID"],
        direction="forward",
        tolerance=window_td,
    )
    merged.drop(columns=["_TIMER_shifted"], inplace=True)

    paired = merged[merged["GLC_post"].notna()].copy()
    n_paired   = len(paired)
    coverage   = n_paired / len(cohort) * 100
    print(f"    [{window_hours:>2.0f}h window]  "
          f"Paired events : {n_paired:,}  ({coverage:.1f}% coverage)")

    paired["glucose_drop"] = paired["GLC_AL"] - paired["GLC_post"]
    paired.rename(columns={"GLC_AL": "GLC_pre"}, inplace=True)

    return paired.reset_index(drop=True)


def filter_no_intervening_insulin(
    paired_df: pd.DataFrame,
    all_insulin_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Appendix sensitivity analysis [Fix #8]:
    Remove paired events where another insulin event from the same patient+stay
    occurs strictly between the index insulin event and G_post.

    Rationale: In ICU settings, patients may receive multiple boluses in quick
    succession. When another insulin event precedes G_post, the observed
    glucose_drop reflects a combined multi-dose effect, not the index event
    alone. This filter produces a pharmacologically cleaner subset.

    Note: the primary experiment deliberately allows intervening insulin
    because the leakage audit tests algebraic formula leakage, not causal
    attribution of glucose change to a single dose.
    """
    ins_times = (
        all_insulin_df[["SUBJECT_ID", "ICUSTAY_ID", "TIMER"]]
        .rename(columns={"TIMER": "TIMER_other"})
    )

    cross = paired_df[["SUBJECT_ID", "ICUSTAY_ID", "TIMER", "TIMER_post"]].copy()
    cross["_orig_idx"] = cross.index
    cross = cross.merge(ins_times, on=["SUBJECT_ID", "ICUSTAY_ID"], how="left")

    cross["_intervening"] = (
        (cross["TIMER_other"] > cross["TIMER"]) &
        (cross["TIMER_other"] < cross["TIMER_post"])
    )

    has_intervening = cross.groupby("_orig_idx")["_intervening"].any()
    keep_idx = has_intervening[~has_intervening].index

    filtered    = paired_df.loc[keep_idx].copy()
    n_removed   = len(paired_df) - len(filtered)
    pct_removed = n_removed / len(paired_df) * 100

    print(f"    No-intervening filter: removed {n_removed:,} ({pct_removed:.1f}%), "
          f"kept {len(filtered):,} events")
    return filtered


def engineer_features(df: pd.DataFrame, icu_admit_times: dict) -> pd.DataFrame:
    """
    Build all modelling columns from strictly pre-event information.

    elapsed_icu_days replaces LOS_ICU_days [Fix #3]:
      LOS_ICU_days   = total stay length (discharge time − admission time)
                       → FUTURE information, unavailable at event time
      elapsed_icu_days = event time − ICU admission proxy
                       → always computable at the time of the insulin event

    NaN handling [Fix #5]:
      glcsource_enc has NaN where GLCSOURCE_AL is missing (~24% of rows).
      We leave these as np.nan:
        - LightGBM handles np.nan natively as missing value splits
        - LogisticRegression uses SimpleImputer(median) in its pipeline
      Removed: fillna(-999), which caused LightGBM to treat missing as a
      legitimate very-negative value rather than as truly unknown.
    """
    df = df.copy()

    df["EVENT_enc"] = (df["EVENT"] == "BOLUS_PUSH").astype(int)

    df["glcsource_enc"] = np.where(
        df["GLCSOURCE_AL"].isna(), np.nan,
        (df["GLCSOURCE_AL"] == "FINGERSTICK").astype(float)
    )

    df["first_ICU_stay"] = df["first_ICU_stay"].astype(int)

    df["hour_of_day"] = df["TIMER"].dt.hour

    admit_times = df["ICUSTAY_ID"].map(icu_admit_times)
    df["elapsed_icu_days"] = (
        (df["TIMER"] - admit_times).dt.total_seconds() / 86400
    ).clip(lower=0).fillna(0)

    if "GLC_pre" not in df.columns and "GLC_AL" in df.columns:
        df.rename(columns={"GLC_AL": "GLC_pre"}, inplace=True)

    return df


def get_splitter(n_splits: int):
    """StratifiedGroupKFold (sklearn ≥ 1.0) with fallback to GroupKFold."""
    try:
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=CFG["random_seed"]
        )
    except TypeError:
        try:
            return StratifiedGroupKFold(n_splits=n_splits)
        except Exception:
            tqdm.write("    Note: using GroupKFold (StratifiedGroupKFold unavailable)")
            return GroupKFold(n_splits=n_splits)


def make_rf_pipeline() -> SKPipeline:
    """
    Random Forest pipeline:
      SimpleImputer(median) — handles NaN in glcsource_enc
      RandomForestClassifier — scale-invariant; no StandardScaler needed.
      Hyperparameters mirror the synthetic benchmark RF for comparability:
        n_estimators=300, max_features='sqrt', min_samples_leaf=5.
    """
    return SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=CFG["random_seed"],
            n_jobs=-1,
        )),
    ])


def make_lr_pipeline() -> SKPipeline:
    """
    Logistic Regression pipeline [Fix #9]:
      SimpleImputer(median) — handles NaN in glcsource_enc
      StandardScaler        — required for LR convergence and comparability
      LogisticRegression    — learns +G_pre − G_post algebraically for B-component
    [Fix #5]: NaN handled via imputer, not fillna(-999)
    """
    return SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=CFG["random_seed"],
            n_jobs=-1,
        )),
    ])


def run_cv(
    df: pd.DataFrame,
    feature_cols: list,
    groups: np.ndarray,
    model_type: str = "lgbm",
    n_splits: int = 5,
) -> dict:
    """
    Patient-level k-fold cross-validation.

    model_type:
      "lgbm"   : LightGBM, fixed n_estimators=300, np.nan passed natively
      "lr"     : Logistic Regression with imputation + scaling pipeline
      "oracle" : score = glucose_drop directly (no model) [Fix #4]

    Label threshold (Q75) is always computed from the TRAINING fold only.
    No early stopping [Fix #7] — fully reproducible with fixed n_estimators.
    NaN → native handling, not fillna(-999) [Fix #5].
    """
    splitter = get_splitter(n_splits)
    X_dummy  = np.zeros((len(df), 1))   # used only for splitter shape

    if model_type in ("lgbm", "lr", "rf"):
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).values

    global_q75 = df["glucose_drop"].quantile(CFG["label_quantile"])
    y_global   = (df["glucose_drop"] >= global_q75).astype(int).values

    aurocs, praucs = [], []

    fold_bar = tqdm(
        splitter.split(X_dummy, y_global, groups),
        total=n_splits,
        desc=f"      {model_type:6s} folds",
        leave=False,
        ncols=72,
    )

    for fold_i, (train_idx, test_idx) in enumerate(fold_bar):
        drop_train = df["glucose_drop"].iloc[train_idx]
        drop_test  = df["glucose_drop"].iloc[test_idx]
        q75_train  = drop_train.quantile(CFG["label_quantile"])
        y_train    = (drop_train >= q75_train).astype(int).values
        y_test     = (drop_test  >= q75_train).astype(int).values

        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            tqdm.write(f"      Fold {fold_i+1}: degenerate label, skipping")
            continue

        if model_type == "oracle":
            # Direct algebraic score: higher glucose_drop → higher P(large drop)
            y_prob = drop_test.values.astype(float)

        elif model_type == "lgbm":
            # LightGBM handles np.nan natively as a missing-value split [Fix #5]
            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X[train_idx], y_train)
            y_prob = model.predict_proba(X[test_idx])[:, 1]

        elif model_type == "lr":
            pipe = make_lr_pipeline()
            pipe.fit(X[train_idx], y_train)
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]

        elif model_type == "rf":
            pipe = make_rf_pipeline()
            pipe.fit(X[train_idx], y_train)
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]

        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        aurocs.append(float(roc_auc_score(y_test, y_prob)))
        praucs.append(float(average_precision_score(y_test, y_prob)))

    return {
        "auroc_mean":     float(np.mean(aurocs)),
        "auroc_std":      float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
        "prauc_mean":     float(np.mean(praucs)),
        "prauc_std":      float(np.std(praucs, ddof=1)) if len(praucs) > 1 else 0.0,
        "n_folds_used":   len(aurocs),
        "auroc_per_fold": aurocs,
        "prauc_per_fold": praucs,
    }


def run_all_pipelines(
    paired_df: pd.DataFrame,
    window_hours: float,
    icu_admit_times: dict,
    cohort_label: str = "primary",
) -> pd.DataFrame:
    """
    Run all pipeline × model combinations on one paired event table.
    Returns a DataFrame with one row per (pipeline, model).
    """
    df     = engineer_features(paired_df, icu_admit_times)
    groups = df["SUBJECT_ID"].values

    q75      = df["glucose_drop"].quantile(CFG["label_quantile"])
    pos_rate = (df["glucose_drop"] >= q75).mean()
    tqdm.write(
        f"\n    Outcome: mean={df['glucose_drop'].mean():.1f}  "
        f"std={df['glucose_drop'].std():.1f}  "
        f"Q75={q75:.1f} mg/dL  pos_rate={pos_rate*100:.1f}%"
    )

    rows = []
    pipe_bar = tqdm(
        PIPELINE_DEFS.items(),
        total=len(PIPELINE_DEFS),
        desc=f"  [{window_hours}h | {cohort_label}] Pipelines",
        ncols=72,
    )

    for pipe_key, pipe_cfg in pipe_bar:
        feature_cols = CLEAN_FEATURES + pipe_cfg["extra"]
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            tqdm.write(f"    WARNING: skipping [{pipe_key}] — missing: {missing}")
            continue

        for model_type in pipe_cfg["models"]:
            pipe_bar.set_postfix({"pipe": pipe_cfg["label"][:16], "model": model_type})
            tqdm.write(f"\n    ── [{pipe_cfg['label']} | {model_type.upper()}] ──")

            cv = run_cv(
                df,
                feature_cols if model_type != "oracle" else [],
                groups,
                model_type=model_type,
                n_splits=CFG["n_cv_splits"],
            )

            tqdm.write(
                f"       AUROC {cv['auroc_mean']:.4f} ± {cv['auroc_std']:.4f}  |  "
                f"PR-AUC {cv['prauc_mean']:.4f} ± {cv['prauc_std']:.4f}"
            )

            rows.append({
                "window_h":      window_hours,
                "cohort":        cohort_label,
                "pipeline_key":  pipe_key,
                "pipeline":      pipe_cfg["label"],
                "model":         model_type,
                "n_events":      len(df),
                "n_patients":    int(df["SUBJECT_ID"].nunique()),
                "auroc":         round(cv["auroc_mean"], 4),
                "auroc_std":     round(cv["auroc_std"],  4),
                "prauc":         round(cv["prauc_mean"], 4),
                "prauc_std":     round(cv["prauc_std"],  4),
                "n_folds":       cv["n_folds_used"],
                "_auroc_folds":  cv["auroc_per_fold"],
                "_prauc_folds":  cv["prauc_per_fold"],
            })

    result_df = pd.DataFrame(rows)

    for model_type in result_df["model"].unique():
        mask        = result_df["model"] == model_type
        clean_rows  = result_df[mask & (result_df["pipeline_key"] == "clean")]
        if len(clean_rows) == 0:
            continue
        clean_auroc = clean_rows["auroc"].values[0]
        headroom    = 1.0 - clean_auroc
        result_df.loc[mask, "delta_auroc"] = (
            result_df.loc[mask, "auroc"] - clean_auroc
        ).round(4)
        if headroom > 0:
            result_df.loc[mask, "headroom_pct"] = (
                (result_df.loc[mask, "delta_auroc"] / headroom * 100).round(1)
            )

    return result_df


DISPLAY_COLS = [
    "pipeline", "model", "n_events", "n_patients",
    "auroc", "auroc_std", "prauc", "prauc_std",
    "delta_auroc", "headroom_pct",
]


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'═'*82}")
    print(f"  {title}")
    print('═'*82)
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))


def save_results(result_df: pd.DataFrame, out: Path, stem: str) -> None:
    """Save main CSV (drops internal fold arrays) and per-fold CSV."""
    save_cols = [c for c in result_df.columns if not c.startswith("_")]
    result_df[save_cols].to_csv(out / f"{stem}.csv", index=False)

    fold_rows = []
    for _, row in result_df.iterrows():
        for fold_i, (fa, fp) in enumerate(
            zip(row.get("_auroc_folds", []), row.get("_prauc_folds", []))
        ):
            fold_rows.append({
                "window_h": row["window_h"],
                "cohort":   row.get("cohort", "primary"),
                "pipeline": row["pipeline"],
                "model":    row["model"],
                "fold":     fold_i + 1,
                "auroc":    fa,
                "prauc":    fp,
            })
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(out / f"{stem}_per_fold.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="ICU Glucose Algebraic Leakage Audit v2"
    )
    parser.add_argument(
        "--data_path",
        default="glucose_insulin_pair.csv",
        help="Path to Datasets/glucose_insulin_pair.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="results_icu_audit",
        help="Directory for output CSV files",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.data_path)

    insulin_df, glucose_df = separate_event_types(raw)

    cohort = build_short_bolus_cohort(insulin_df)

    icu_admit_times = compute_icu_admit_times(raw)
    print(f"\n    ICU admission proxy computed for {len(icu_admit_times):,} stays")

    print("\n[4] Main experiment — window sensitivity (4h / 6h / 8h)")
    all_results  = []
    paired_cache = {}   # cache paired DataFrames to avoid re-pairing for Step 5

    window_bar = tqdm(CFG["sensitivity_windows"], desc="Windows", ncols=72)
    for window_h in window_bar:
        window_bar.set_postfix({"window": f"{window_h}h"})
        tqdm.write(f"\n{'═'*82}")
        tqdm.write(f"  POST-GLUCOSE WINDOW: {window_h}h")
        tqdm.write('═'*82)

        paired = attach_post_glucose(cohort, glucose_df, window_hours=window_h)
        paired_cache[window_h] = paired

        if len(paired) < 500:
            tqdm.write(f"  Too few events ({len(paired)}), skipping")
            continue

        window_results = run_all_pipelines(paired, window_h, icu_admit_times,
                                           cohort_label="primary")
        all_results.append(window_results)

    final = pd.concat(all_results, ignore_index=True)

    primary_6h = final[final["window_h"] == CFG["primary_window_h"]].copy()
    print_table(
        primary_6h,
        f"MAIN RESULTS  (window={CFG['primary_window_h']}h | label=Q75 | "
        f"5-fold patient-level CV | LightGBM + LR)",
    )
    print_table(
        final[final["model"] == "lgbm"].copy(),
        "SENSITIVITY — window comparison (LightGBM)",
    )
    print_table(
        final[final["model"] == "lr"].copy(),
        "SENSITIVITY — window comparison (Logistic Regression)",
    )

    save_results(primary_6h, out, "results_primary_6h")
    save_results(final,      out, "results_all_windows")

    print(f"\n[5] Sensitivity — no-intervening-insulin "
          f"(primary {CFG['primary_window_h']}h window, Appendix)")

    paired_6h       = paired_cache[CFG["primary_window_h"]]
    paired_no_interv = filter_no_intervening_insulin(paired_6h, insulin_df)

    if len(paired_no_interv) >= 500:
        noint_results = run_all_pipelines(
            paired_no_interv,
            CFG["primary_window_h"],
            icu_admit_times,
            cohort_label="no_intervening",
        )
        print_table(noint_results,
                    "SENSITIVITY — no-intervening-insulin (Appendix)")
        save_results(noint_results, out, "results_sensitivity_no_intervening")
    else:
        print(f"  Too few events after filter ({len(paired_no_interv)}), skipping")

    print(f"\n[6] All outputs saved to  {out}/")
    print("    results_primary_6h.csv                     ← main table (paper)")
    print("    results_primary_6h_per_fold.csv            ← per-fold AUROC (CI)")
    print("    results_all_windows.csv                    ← window sensitivity")
    print("    results_all_windows_per_fold.csv           ← per-fold details")
    print("    results_sensitivity_no_intervening.csv     ← appendix")
    print("    results_sensitivity_no_intervening_per_fold.csv")


if __name__ == "__main__":
    main()
