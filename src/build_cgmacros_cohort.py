"""Build meal-level paired-outcome dataset from raw CGMacros files."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

POST_MEAL_MIN_START = 15
POST_MEAL_MAX_END   = 120
MIN_CGM_READINGS    = 20
MIN_GLUCOSE_RISE_THRESHOLD = -30.0

GUT_HEALTH_COLS = [
    "Gut Lining Health",
    "LPS Biosynthesis Pathways",
    "Biofilm, Chemotaxis, and Virulence Pathways",
    "TMA Production Pathways",
    "Ammonia Production Pathways",
    "Metabolic Fitness",
    "Active Microbial Diversity",
    "Butyrate Production Pathways",
    "Flagellar Assembly Pathways",
    "Putrescine Production Pathways",
    "Uric Acid Production Pathways",
    "Bile Acid Metabolism Pathways",
    "Inflammatory Activity",
    "Gut Microbiome Health",
    "Digestive Efficiency",
    "Protein Fermentation",
    "Gas Production",
    "Methane Gas Production Pathways",
    "Sulfide Gas Production Pathways",
    "Oxalate Metabolism Pathways",
    "Salt Stress Pathways",
    "Microbiome-Induced Stress",
]

GUT_HEALTH_RENAMED = {col: f"gut_{col.lower().replace(' ', '_').replace(',', '').replace('/', '_')}"
                      for col in GUT_HEALTH_COLS}


def normalise_meal_type(raw: str) -> str:
    """Normalise inconsistent meal-type labels to one of 4 canonical values."""
    s = str(raw).strip().lower()
    if "breakfast" in s:
        return "breakfast"
    if "lunch" in s:
        return "lunch"
    if "dinner" in s:
        return "dinner"
    return "snack"


def select_cgm_col(df: pd.DataFrame) -> str:
    """Return the CGM column with more non-null observations."""
    libre_n  = df["Libre GL"].notna().sum()
    dexcom_n = df["Dexcom GL"].notna().sum()
    return "Libre GL" if libre_n >= dexcom_n else "Dexcom GL"


def normalise_subject_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise column names that differ across subject CSVs (METs vs Intensity)."""
    rename_map: dict[str, str] = {}
    if "Intensity" in df.columns and "METs" not in df.columns:
        rename_map["Intensity"] = "METs"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def extract_meal_features(
    subject_id: int,
    df_subject: pd.DataFrame,
    cgm_col: str,
) -> list[dict]:
    """Extract one record per valid meal event for a single subject."""
    meal_rows = df_subject[df_subject["Meal Type"].notna()].copy()
    records: list[dict] = []

    for _, meal in meal_rows.iterrows():
        meal_time = meal["Timestamp"]

        pre_meal_cgm = meal[cgm_col]
        if pd.isna(pre_meal_cgm) or pre_meal_cgm <= 0:
            continue

        window_mask = (
            (df_subject["Timestamp"] > meal_time + pd.Timedelta(minutes=POST_MEAL_MIN_START))
            & (df_subject["Timestamp"] <= meal_time + pd.Timedelta(minutes=POST_MEAL_MAX_END))
        )
        window = df_subject.loc[window_mask, cgm_col].dropna()

        if len(window) < MIN_CGM_READINGS:
            continue

        peak_cgm = float(window.max())
        glucose_rise = peak_cgm - pre_meal_cgm

        if glucose_rise < MIN_GLUCOSE_RISE_THRESHOLD:
            continue

        hour_of_day    = meal_time.hour
        meal_type_norm = normalise_meal_type(meal["Meal Type"])

        records.append({
            "subject_id":   subject_id,
            "meal_time":    meal_time.isoformat(),
            "pre_meal_cgm": float(pre_meal_cgm),    # A (baseline)
            "peak_cgm":     peak_cgm,               # B (leaky feature)
            "glucose_rise": glucose_rise,           # Y = B − A (target)
            "peak_over_pre": peak_cgm / pre_meal_cgm,
            "calories":     float(meal["Calories"]) if pd.notna(meal["Calories"]) else np.nan,
            "carbs":        float(meal["Carbs"])    if pd.notna(meal["Carbs"])    else np.nan,
            "protein":      float(meal["Protein"])  if pd.notna(meal["Protein"])  else np.nan,
            "fat":          float(meal["Fat"])      if pd.notna(meal["Fat"])      else np.nan,
            "fiber":        float(meal["Fiber"])    if pd.notna(meal["Fiber"])    else np.nan,
            "hr":           float(meal["HR"])   if pd.notna(meal["HR"])   else np.nan,
            "mets":         float(meal["METs"]) if pd.notna(meal["METs"]) else np.nan,
            "hour_of_day":  hour_of_day,
            "meal_type":    meal_type_norm,
        })

    return records


def build_cohort(
    data_dir: Path,
    bio_path: Path,
    gut_path:  Path,
) -> pd.DataFrame:
    """Build the full meal-level cohort by merging CGM features, demographics, and gut scores."""
    bio = pd.read_csv(bio_path)
    bio.columns = bio.columns.str.strip()
    bio = bio.rename(columns={
        "subject":                  "subject_id",
        "Age":                      "age",
        "Gender":                   "gender",
        "BMI":                      "bmi",
        "A1c PDL (Lab)":            "hba1c",
        "Fasting GLU - PDL (Lab)":  "fasting_glucose",
        "Insulin":                  "insulin",
        "Triglycerides":            "triglycerides",
        "Cholesterol":              "cholesterol",
        "HDL":                      "hdl",
        "Non HDL":                  "non_hdl",
        "LDL (Cal)":                "ldl",
        "VLDL (Cal)":               "vldl",
        "Cho/HDL Ratio":            "cho_hdl_ratio",
    })
    bio_keep = [
        "subject_id", "age", "gender", "bmi",
        "hba1c", "fasting_glucose", "insulin",
        "triglycerides", "cholesterol", "hdl", "ldl",
    ]
    bio = bio[[c for c in bio_keep if c in bio.columns]]
    bio["gender_binary"] = (bio["gender"].str.strip().str.upper() == "M").astype(int)

    gut = pd.read_csv(gut_path)
    gut.columns = (gut.columns
                   .str.replace("﻿", "", regex=False)
                   .str.strip())
    gut = gut.rename(columns={"subject": "subject_id"})
    gut = gut.rename(columns=GUT_HEALTH_RENAMED)
    gut_safe_cols = list(GUT_HEALTH_RENAMED.values())
    gut_keep = ["subject_id"] + [c for c in gut_safe_cols if c in gut.columns]
    gut = gut[gut_keep]

    all_records: list[dict] = []

    subject_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("CGMacros-")]
    )
    print(f"  Processing {len(subject_dirs)} subject directories …")

    for subj_dir in subject_dirs:
        subj_id = int(subj_dir.name.split("-")[1])
        csv_file = subj_dir / f"{subj_dir.name}.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file, low_memory=False)
        df = normalise_subject_columns(df)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        cgm_col = select_cgm_col(df)
        records = extract_meal_features(subj_id, df, cgm_col)
        all_records.extend(records)

    print(f"  Total meal records extracted: {len(all_records):,}")

    df_meals = pd.DataFrame(all_records)
    df_meals = df_meals.merge(bio, on="subject_id", how="left")
    df_meals = df_meals.merge(gut, on="subject_id", how="left")

    meal_type_map = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3}
    df_meals["meal_type_code"] = df_meals["meal_type"].map(meal_type_map).fillna(3).astype(int)
    df_meals = df_meals.reset_index(drop=True)

    reconstructed = df_meals["peak_cgm"] - df_meals["pre_meal_cgm"]
    oracle_error  = (reconstructed - df_meals["glucose_rise"]).abs()
    assert oracle_error.max() < 1e-6, (
        f"Oracle reconstruction failed! Max error = {oracle_error.max():.8f}"
    )

    return df_meals


def compute_oracle_audit(df: pd.DataFrame) -> dict:
    """Verify glucose_rise can be perfectly reconstructed from peak_cgm and pre_meal_cgm."""
    reconstructed = df["peak_cgm"] - df["pre_meal_cgm"]
    error         = (reconstructed - df["glucose_rise"]).abs()

    oracle_audit = {
        "cohort":               "CGMacros",
        "outcome_formula":      "Y = peak_cgm - pre_meal_cgm",
        "baseline_variable_A":  "pre_meal_cgm",
        "followup_variable_B":  "peak_cgm",
        "n_total_meals":        int(len(df)),
        "n_subjects":           int(df["subject_id"].nunique()),
        "meals_per_subject_mean": round(float(df.groupby("subject_id").size().mean()), 1),
        "meals_per_subject_min":  int(df.groupby("subject_id").size().min()),
        "meals_per_subject_max":  int(df.groupby("subject_id").size().max()),
        "oracle_max_abs_error":   float(error.max()),
        "oracle_mean_abs_error":  float(error.mean()),
        "oracle_agreement_pct":   100.0,
        "A_gt_0_pct":            100.0,
        "B_gt_0_pct":            round(float((df["peak_cgm"] > 0).mean() * 100), 2),
        "glucose_rise_mean":  round(float(df["glucose_rise"].mean()), 2),
        "glucose_rise_sd":    round(float(df["glucose_rise"].std(ddof=1)), 2),
        "glucose_rise_median":round(float(df["glucose_rise"].median()), 2),
        "glucose_rise_q25":   round(float(df["glucose_rise"].quantile(0.25)), 2),
        "glucose_rise_q75":   round(float(df["glucose_rise"].quantile(0.75)), 2),
        "glucose_rise_min":   round(float(df["glucose_rise"].min()), 2),
        "glucose_rise_max":   round(float(df["glucose_rise"].max()), 2),
        "global_q75_threshold_mg_dl": round(float(df["glucose_rise"].quantile(0.75)), 2),
        "label_prevalence_pct":       25.0,
        "pre_meal_cgm_mean":   round(float(df["pre_meal_cgm"].mean()), 2),
        "pre_meal_cgm_sd":     round(float(df["pre_meal_cgm"].std(ddof=1)), 2),
        "peak_cgm_mean":       round(float(df["peak_cgm"].mean()), 2),
        "peak_cgm_sd":         round(float(df["peak_cgm"].std(ddof=1)), 2),
    }

    return oracle_audit


def compute_cohort_summary(df: pd.DataFrame) -> dict:
    """Summary statistics for the subject-level bio characteristics."""
    subj = df.drop_duplicates("subject_id")
    return {
        "n_subjects":           int(subj.shape[0]),
        "n_meal_observations":  int(len(df)),
        "age_mean":    round(float(subj["age"].mean()), 1),
        "age_sd":      round(float(subj["age"].std(ddof=1)), 1),
        "bmi_mean":    round(float(subj["bmi"].mean()), 1),
        "bmi_sd":      round(float(subj["bmi"].std(ddof=1)), 1),
        "female_pct":  round(float((subj["gender"].str.strip().str.upper() == "F").mean() * 100), 1),
        "hba1c_mean":  round(float(subj["hba1c"].mean()), 2),
        "hba1c_sd":    round(float(subj["hba1c"].std(ddof=1)), 2),
        "fasting_glucose_mean":  round(float(subj["fasting_glucose"].mean()), 1),
        "fasting_glucose_sd":    round(float(subj["fasting_glucose"].std(ddof=1)), 1),
        "triglycerides_mean":    round(float(subj["triglycerides"].mean()), 1),
        "hdl_mean":              round(float(subj["hdl"].mean()), 1),
        "ldl_mean":              round(float(subj["ldl"].mean()), 1),
        "meal_type_counts":      df["meal_type"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build CGMacros meal-level cohort for leakage audit."
    )
    parser.add_argument(
        "--data_dir",
        default="CGMacros_dateshifted365/CGMacros_Dataset",
        help="Path to CGMacros_Dataset directory (contains CGMacros-001/, bio.csv, etc.)",
    )
    parser.add_argument(
        "--outdir",
        default="results/tables",
        help="Output directory for CSV and JSON files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bio_path = data_dir / "bio.csv"
    gut_path = data_dir / "gut_health_test.csv"

    print("=" * 65)
    print("  CGMacros Cohort Builder")
    print("  Paired outcome: postprandial glucose excursion Y = B − A")
    print("=" * 65)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not bio_path.exists():
        raise FileNotFoundError(f"bio.csv not found: {bio_path}")
    if not gut_path.exists():
        raise FileNotFoundError(f"gut_health_test.csv not found: {gut_path}")

    print(f"\nBuilding cohort from: {data_dir}")
    df_cohort = build_cohort(data_dir, bio_path, gut_path)

    print(f"\n{'─' * 55}")
    print(f"  Final cohort: {len(df_cohort):,} meal observations "
          f"across {df_cohort['subject_id'].nunique()} subjects")
    print(f"  Meals per subject: "
          f"{df_cohort.groupby('subject_id').size().min()}–"
          f"{df_cohort.groupby('subject_id').size().max()} "
          f"(mean {df_cohort.groupby('subject_id').size().mean():.1f})")
    print(f"  Glucose rise (Y): "
          f"mean={df_cohort['glucose_rise'].mean():.1f}, "
          f"SD={df_cohort['glucose_rise'].std():.1f}, "
          f"range=[{df_cohort['glucose_rise'].min():.1f}, "
          f"{df_cohort['glucose_rise'].max():.1f}] mg/dL")
    print(f"  Global Q75 threshold: {df_cohort['glucose_rise'].quantile(0.75):.1f} mg/dL "
          f"(label prevalence = 25.0%)")
    print(f"{'─' * 55}")

    oracle_audit = compute_oracle_audit(df_cohort)
    print(f"\nOracle reconstruction audit:")
    print(f"  Max absolute error:   {oracle_audit['oracle_max_abs_error']:.8f}")
    print(f"  Label agreement:      100.00% (by algebraic identity Y = B − A)")

    cohort_path = outdir / "cgmacros_meal_cohort.csv"
    df_cohort.to_csv(cohort_path, index=False)
    print(f"\nCohort CSV saved → {cohort_path}")

    audit_path = outdir / "cgmacros_oracle_audit.json"
    with open(audit_path, "w") as f:
        json.dump(oracle_audit, f, indent=2)
    print(f"Oracle audit saved → {audit_path}")

    summary = compute_cohort_summary(df_cohort)
    summary_path = outdir / "cgmacros_cohort_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Cohort summary saved → {summary_path}")

    print("\nDone. Next step:")
    print("  python src/run_external_cgmacros.py --data results/tables/cgmacros_meal_cohort.csv")


if __name__ == "__main__":
    main()
