"""External leakage audit on eICU Collaborative Research Database (Theorem 2 — algebraic leakage)."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.base
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not found — XGB will be skipped.")

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import FoldSealedScaler, FoldSealedWinsorizer

# Numeric features safe to use before any outcome is known
NUMERIC_CLEAN = [
    "age_num",
    "lab_-basos", "lab_-eos", "lab_-lymphs", "lab_-monos", "lab_-polys",
    "lab_ALT_(SGPT)", "lab_AST_(SGOT)", "lab_BUN", "lab_Hct", "lab_Hgb",
    "lab_MCH", "lab_MCHC", "lab_MCV", "lab_MPV", "lab_RBC", "lab_RDW",
    "lab_WBC_x_1000", "lab_albumin", "lab_anion_gap", "lab_bedside_glucose",
    "lab_bicarbonate", "lab_calcium", "lab_chloride", "lab_creatinine",
    "lab_glucose", "lab_magnesium", "lab_platelets_x_1000", "lab_potassium",
    "lab_sodium", "lab_total_protein",
    # Vitals (kept; fold-sealed imputer handles high missingness)
    "vital_cvp", "vital_heartrate", "vital_respiration", "vital_sao2",
    "vital_st1", "vital_st2", "vital_st3",
    "vital_systemicdiastolic", "vital_systemicmean", "vital_systemicsystolic",
]

CATEGORICAL_CLEAN = [
    "gender",
    "ethnicity",
    "unittype",
    "unitstaytype",
    "hospitaladmitsource",
    "unitadmitsource",
]

CLEAN_FEATURES = NUMERIC_CLEAN + CATEGORICAL_CLEAN

LEAKY_LOCATION_NUMERIC: list[str] = []
LEAKY_LOCATION_CATEGORICAL = [
    "hospitaldischargelocation",
    "unitdischargelocation",
]

# label24h = (hospitaldischargestatus == 'Expired') AND (unitdischargeoffset_num <= 1440)
LEAKY_DEFN_NUMERIC = ["unitdischargeoffset_num"]
LEAKY_DEFN_CATEGORICAL = ["hospitaldischargestatus"]

ORACLE_THRESHOLD_24H = 1440   # 24 × 60
ORACLE_THRESHOLD_48H = 2880   # 48 × 60


class FoldSealedPreprocessorICU:
    """
    Full fold-sealed preprocessing for ICU data:
      1. Median imputation for numeric columns (fit on train only)
      2. Most-frequent imputation for categorical columns (fit on train only)
      3. OrdinalEncoder for categorical columns (fit on train only)
      4. Winsorisation 1–99 % for all numeric columns (fit on train only)
      5. StandardScaler (fit on train only)

    The label is NOT computed here — it is pre-defined in the dataset.
    """

    def __init__(self) -> None:
        self._num_imputer = SimpleImputer(strategy="median")
        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._winsorizer = FoldSealedWinsorizer(lower_pct=1.0, upper_pct=99.0)
        self._scaler = FoldSealedScaler()
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._fitted = False

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        num_cols: list[str],
        cat_cols: list[str],
    ) -> np.ndarray:
        self._num_cols = num_cols
        self._cat_cols = cat_cols

        X_num = df_train[num_cols].values.astype(float)
        X_num = self._num_imputer.fit_transform(X_num)

        if cat_cols:
            X_cat = df_train[cat_cols].copy()
            X_cat = X_cat.astype("object")
            X_cat = X_cat.where(pd.notna(X_cat), np.nan)

            X_cat = self._cat_imputer.fit_transform(X_cat)

            X_cat = pd.DataFrame(X_cat, columns=cat_cols).astype(object).values
            X_cat = self._encoder.fit_transform(X_cat)

            X = np.hstack([X_num, X_cat])
        else:
            X = X_num

        X = self._winsorizer.fit_transform(X)
        X = self._scaler.fit_transform(X)
        self._fitted = True
        return X

    def transform(
        self,
        df_test: pd.DataFrame,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Must call fit_transform before transform.")

        X_num = df_test[self._num_cols].values.astype(float)
        X_num = self._num_imputer.transform(X_num)

        if self._cat_cols:
            X_cat = df_test[self._cat_cols].copy()
            X_cat = X_cat.astype("object")
            X_cat = X_cat.where(pd.notna(X_cat), np.nan)

            X_cat = self._cat_imputer.transform(X_cat)

            X_cat = pd.DataFrame(X_cat, columns=self._cat_cols).astype(object).values
            X_cat = self._encoder.transform(X_cat)

            X = np.hstack([X_num, X_cat])
        else:
            X = X_num

        X = self._winsorizer.transform(X)
        X = self._scaler.transform(X)
        return X


def build_models() -> dict:
    """Return {name: unfitted estimator} for all four classifiers."""
    models: dict = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        # LinearSVC wrapped for probability calibration; fast on 150k+ rows
        "SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced",
                      random_state=42),
            cv=3, method="sigmoid",
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, verbosity=0, n_jobs=-1,
        )
    return models


def oracle_auroc(df: pd.DataFrame, label_col: str, offset_threshold: int) -> dict:
    """
    Compute oracle AUROC using the exact definitional rule:
        score = 1  iff  hospitaldischargestatus == 'Expired'
                        AND unitdischargeoffset_num <= offset_threshold

    Because this is a deterministic rule, no cross-validation is needed.
    Returns AUROC and PR-AUC over the entire cohort.
    """
    y_true = df[label_col].values.astype(int)

    expired = (df["hospitaldischargestatus"].fillna("").str.strip() == "Expired")
    within_time = (df["unitdischargeoffset_num"] <= offset_threshold)
    oracle_score = (expired & within_time).astype(int).values

    try:
        auroc = float(roc_auc_score(y_true, oracle_score))
    except ValueError:
        auroc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, oracle_score))
    except ValueError:
        pr_auc = float("nan")

    agreement = float(np.mean(oracle_score == y_true))

    return {
        "pipeline": "oracle",
        "model": "OracleRule",
        "AUROC_mean": round(auroc, 4),
        "AUROC_sd": 0.0,
        "PR_AUC_mean": round(pr_auc, 4),
        "PR_AUC_sd": 0.0,
        "DELTA_AUC": None,   # filled in after clean baseline is known
        "n_seeds": 1,
        "oracle_agreement": round(agreement, 6),
        "note": f"Deterministic rule: Expired AND offset_num <= {offset_threshold}",
    }


def _xgb_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight = n_neg / n_pos for XGBoost."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def run_cv_pipeline(
    df: pd.DataFrame,
    label_col: str,
    num_extra: list[str],
    cat_extra: list[str],
    pipeline_name: str,
    n_folds: int,
    n_seeds: int,
    smote_k: int = 5,
    max_train_n: int | None = None,
) -> list[dict]:
    """
    Run repeated stratified k-fold CV for ONE pipeline definition.

    Parameters
    ----------
    df            : full cohort DataFrame
    label_col     : name of binary target column
    num_extra     : ADDITIONAL numeric features to include (on top of NUMERIC_CLEAN)
    cat_extra     : ADDITIONAL categorical features to include
    pipeline_name : label for output rows
    n_folds       : number of CV folds (5)
    n_seeds       : number of random seeds (30)
    smote_k       : k-neighbours for SMOTE (applied inside training fold only)
    max_train_n   : if not None, subsample training fold to this size for SVM speed

    Returns
    -------
    List of per-seed dicts with AUROC_mean, AUROC_sd (across folds), PR_AUC_mean/sd
    """
    y_all = df[label_col].values.astype(int)

    num_cols = NUMERIC_CLEAN + num_extra
    cat_cols = CATEGORICAL_CLEAN + cat_extra

    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    models = build_models()
    seed_records: list[dict] = []

    seed_pbar = tqdm(
        range(1, n_seeds + 1),
        desc=f"  [{pipeline_name}] seeds",
        ncols=100,
        colour="cyan",
        leave=True,
    )

    for seed in seed_pbar:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        model_fold_aurocs: dict[str, list[float]] = {m: [] for m in models}
        model_fold_praucs: dict[str, list[float]] = {m: [] for m in models}

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(
            np.zeros(len(df)), y_all
        )):
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test  = df.iloc[test_idx].reset_index(drop=True)
            y_train  = y_all[train_idx]
            y_test   = y_all[test_idx]

            if len(np.unique(y_test)) < 2:
                continue

            prep = FoldSealedPreprocessorICU()
            X_train = prep.fit_transform(df_train, num_cols, cat_cols)
            X_test  = prep.transform(df_test)

            n_pos_train = y_train.sum()
            n_neg_train = len(y_train) - n_pos_train
            if n_pos_train >= smote_k + 1 and n_neg_train >= smote_k + 1:
                try:
                    sm = SMOTE(k_neighbors=smote_k, random_state=seed)
                    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
                except Exception:
                    X_train_bal, y_train_bal = X_train, y_train
            else:
                X_train_bal, y_train_bal = X_train, y_train

            # Optional: subsample large training fold (useful for SVM)
            if max_train_n is not None and len(X_train_bal) > max_train_n:
                rng = np.random.default_rng(seed * 1000 + fold_idx)
                idx_sub = rng.choice(len(X_train_bal), max_train_n, replace=False)
                X_train_bal = X_train_bal[idx_sub]
                y_train_bal = y_train_bal[idx_sub]

            for model_name, model in models.items():
                clf = sklearn.base.clone(model)

                if HAS_XGB and model_name == "XGBoost":
                    clf.set_params(
                        scale_pos_weight=_xgb_scale_pos_weight(y_train_bal)
                    )

                try:
                    clf.fit(X_train_bal, y_train_bal)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    auroc  = float(roc_auc_score(y_test, y_prob))
                    prauc  = float(average_precision_score(y_test, y_prob))
                except Exception as exc:
                    tqdm.write(
                        f"    Seed {seed} Fold {fold_idx+1} {model_name}: {exc}"
                    )
                    auroc, prauc = float("nan"), float("nan")

                model_fold_aurocs[model_name].append(auroc)
                model_fold_praucs[model_name].append(prauc)

        for model_name in models:
            aucs = np.array([v for v in model_fold_aurocs[model_name]
                             if not np.isnan(v)])
            prs  = np.array([v for v in model_fold_praucs[model_name]
                             if not np.isnan(v)])
            if len(aucs) == 0:
                continue
            seed_records.append({
                "pipeline":  pipeline_name,
                "model":     model_name,
                "seed":      seed,
                "AUROC_mean_folds": float(np.mean(aucs)),
                "PR_AUC_mean_folds": float(np.mean(prs)) if len(prs) > 0 else float("nan"),
                "n_folds_valid": len(aucs),
            })

        # Show latest AUROC for RF as representative
        rf_this_seed = [
            r["AUROC_mean_folds"] for r in seed_records
            if r["seed"] == seed and r["model"] == "RandomForest"
        ]
        if rf_this_seed:
            seed_pbar.set_postfix(RF_AUROC=f"{rf_this_seed[-1]:.4f}", refresh=True)

    return seed_records


def aggregate_seed_records(
    records: list[dict],
) -> list[dict]:
    """
    Aggregate per-seed records → one row per (pipeline, model).
    Reports mean ± SD of per-seed fold-average AUROC across N_SEEDS.
    """
    import itertools

    pipelines = sorted({r["pipeline"] for r in records})
    models    = sorted({r["model"]    for r in records})
    rows = []

    for pipe, mod in itertools.product(pipelines, models):
        vals = [
            r["AUROC_mean_folds"]
            for r in records
            if r["pipeline"] == pipe and r["model"] == mod
        ]
        pr_vals = [
            r["PR_AUC_mean_folds"]
            for r in records
            if r["pipeline"] == pipe and r["model"] == mod
        ]
        if not vals:
            continue
        arr    = np.array(vals)
        pr_arr = np.array([v for v in pr_vals if not np.isnan(v)])
        rows.append({
            "pipeline":      pipe,
            "model":         mod,
            "AUROC_mean":    round(float(np.mean(arr)), 4),
            "AUROC_sd":      round(float(np.std(arr, ddof=1)), 4),
            "PR_AUC_mean":   round(float(np.mean(pr_arr)), 4) if len(pr_arr) else float("nan"),
            "PR_AUC_sd":     round(float(np.std(pr_arr, ddof=1)), 4) if len(pr_arr) > 1 else float("nan"),
            "DELTA_AUC":     None,   # filled in after clean baseline
            "n_seeds":       len(arr),
        })
    return rows


def attach_delta_auc(rows: list[dict]) -> list[dict]:
    """Fill DELTA_AUC column = AUROC_mean(pipeline) - AUROC_mean('clean')."""
    clean_auc = {
        r["model"]: r["AUROC_mean"]
        for r in rows if r["pipeline"] == "clean"
    }
    for r in rows:
        base = clean_auc.get(r["model"])
        if base is not None and r["pipeline"] != "clean":
            r["DELTA_AUC"] = round(r["AUROC_mean"] - base, 4)
        else:
            r["DELTA_AUC"] = 0.0
    return rows


def run_label(
    df: pd.DataFrame,
    label_col: str,
    offset_threshold: int,
    n_folds: int,
    n_seeds: int,
    outdir: Path,
    label_tag: str,
) -> pd.DataFrame:
    """
    Full evaluation for one label (24h or 48h).

    Returns aggregated results DataFrame.
    """
    print(f"\n{'='*70}")
    print(f"  eICU external cohort — label: {label_col}  ({label_tag})")
    print(f"  N = {len(df):,}  |  positives = {df[label_col].sum():,} "
          f"({df[label_col].mean()*100:.1f} %)")
    print(f"{'='*70}\n")

    print("Computing oracle score …")
    oracle_row = oracle_auroc(df, label_col, offset_threshold)
    print(f"  Oracle AUROC = {oracle_row['AUROC_mean']:.4f}  "
          f"(agreement = {oracle_row['oracle_agreement']:.6f})")

    all_seed_records: list[dict] = []

    print(f"\nPipeline 1/3: clean  (N_SEEDS={n_seeds}, N_FOLDS={n_folds})")
    clean_records = run_cv_pipeline(
        df, label_col,
        num_extra=[], cat_extra=[],
        pipeline_name="clean",
        n_folds=n_folds, n_seeds=n_seeds,
    )
    all_seed_records.extend(clean_records)

    print(f"\nPipeline 2/3: leaky_discharge_location  (adds hospitaldischargelocation, unitdischargelocation)")
    leaky_loc_records = run_cv_pipeline(
        df, label_col,
        num_extra=LEAKY_LOCATION_NUMERIC,
        cat_extra=LEAKY_LOCATION_CATEGORICAL,
        pipeline_name="leaky_discharge_location",
        n_folds=n_folds, n_seeds=n_seeds,
    )
    all_seed_records.extend(leaky_loc_records)

    print(f"\nPipeline 3/3: leaky_definitional  (adds hospitaldischargestatus, unitdischargeoffset_num)")
    leaky_defn_records = run_cv_pipeline(
        df, label_col,
        num_extra=LEAKY_DEFN_NUMERIC,
        cat_extra=LEAKY_DEFN_CATEGORICAL,
        pipeline_name="leaky_definitional",
        n_folds=n_folds, n_seeds=n_seeds,
    )
    all_seed_records.extend(leaky_defn_records)

    agg_rows = aggregate_seed_records(all_seed_records)
    agg_rows = attach_delta_auc(agg_rows)

    oracle_row["DELTA_AUC"] = round(
        oracle_row["AUROC_mean"] - np.mean([
            r["AUROC_mean"] for r in agg_rows if r["pipeline"] == "clean"
        ]),
        4,
    )
    agg_rows.append(oracle_row)

    for r in agg_rows:
        r["cohort"]     = "eICU"
        r["label"]      = label_col
        r["label_tag"]  = label_tag
        r["n_total"]    = len(df)
        r["n_positive"] = int(df[label_col].sum())
        r["prevalence"] = round(float(df[label_col].mean()), 4)

    df_out = pd.DataFrame(agg_rows)

    out_path = outdir / f"external_eicu_{label_tag}_results.csv"
    df_out.to_csv(out_path, index=False)
    tqdm.write(f"\n  Saved → {out_path}")

    seed_df = pd.DataFrame(all_seed_records)
    seed_df["cohort"] = "eICU"
    seed_df["label"]  = label_col
    seed_path = outdir / f"external_eicu_{label_tag}_seed_records.csv"
    seed_df.to_csv(seed_path, index=False)
    tqdm.write(f"  Saved → {seed_path}")

    print(f"\n{'─'*65}")
    print(f"{'Pipeline':<30} {'Model':<22} {'AUROC':>8} {'±SD':>7} {'ΔAUC':>8}")
    print(f"{'─'*65}")
    order = ["clean", "leaky_discharge_location", "leaky_definitional", "oracle"]
    for pipe in order:
        pipe_rows = [r for r in agg_rows if r["pipeline"] == pipe]
        for r in sorted(pipe_rows, key=lambda x: x.get("model", "")):
            delta = r.get("DELTA_AUC")
            delta_str = f"{delta:+.4f}" if delta is not None else "   —  "
            print(f"  {r['pipeline']:<28} {r.get('model',''):<22} "
                  f"{r['AUROC_mean']:>8.4f} {r.get('AUROC_sd', 0.0):>7.4f} "
                  f"{delta_str:>8}")
    print(f"{'─'*65}")

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="External leakage audit on eICU Collaborative Research Database."
    )
    parser.add_argument(
        "--data24",
        default="External Cohort/Dataset/eicu_label24h.csv",
        help="Path to eICU 24h label CSV",
    )
    parser.add_argument(
        "--data48",
        default="External Cohort/Dataset/eicu_label48h.csv",
        help="Path to eICU 48h label CSV",
    )
    parser.add_argument(
        "--outdir",
        default="results/tables",
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--figdir",
        default="results/figures",
        help="Directory for output figures",
    )
    parser.add_argument("--n_seeds", type=int, default=30,
                        help="Number of random seeds for repeated CV (default: 30)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--skip48", action="store_true",
                        help="Skip the 48h label analysis (faster)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("Loading eICU datasets …")
    df24 = pd.read_csv(
        args.data24,
        na_values=["", "nan", "NaN", "NA", "N/A", "Unknown", "Other"],
        keep_default_na=True,
    )
    # Gender: recode to binary (Female=0, Male=1; Other/Unknown → NaN for imputer)
    df24["gender"] = df24["gender"].map({"Female": 0, "Male": 1})

    print(f"  eICU 24h loaded: {len(df24):,} rows")

    results_24 = run_label(
        df24, "label24h", ORACLE_THRESHOLD_24H,
        n_folds=args.n_folds, n_seeds=args.n_seeds,
        outdir=outdir, label_tag="24h",
    )

    if not args.skip48:
        df48 = pd.read_csv(
            args.data48,
            na_values=["", "nan", "NaN", "NA", "N/A", "Unknown", "Other"],
            keep_default_na=True,
        )
        df48["gender"] = df48["gender"].map({"Female": 0, "Male": 1})
        print(f"  eICU 48h loaded: {len(df48):,} rows")

        results_48 = run_label(
            df48, "label48h", ORACLE_THRESHOLD_48H,
            n_folds=args.n_folds, n_seeds=args.n_seeds,
            outdir=outdir, label_tag="48h",
        )

        combined = pd.concat([results_24, results_48], ignore_index=True)
    else:
        combined = results_24

    combined_path = outdir / "external_eicu_combined_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined results saved → {combined_path}")

    try:
        _plot_results(combined, figdir)
    except Exception as e:
        print(f"  Warning: plotting failed ({e}). Results CSV still saved.")

    print("\nDone.")


def _plot_results(df_results: pd.DataFrame, figdir: Path) -> None:
    """
    Produce grouped bar chart: one group per pipeline, bars per model.
    Separate panels for 24h and 48h if both are present.
    Matches the paper's matplotlib style (src/plotting.py).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         150,
    })

    PALETTE = {
        "clean":                    "#2C7BB6",
        "leaky_discharge_location": "#FDAE61",
        "leaky_definitional":       "#D7191C",
        "oracle":                   "#1A1A1A",
    }
    PIPELINE_LABELS = {
        "clean":                    "Clean",
        "leaky_discharge_location": "Leaky A\n(discharge location)",
        "leaky_definitional":       "Leaky B\n(definitional formula)",
        "oracle":                   "Oracle\n(algebraic rule)",
    }
    MODEL_ORDER = ["LogisticRegression", "RandomForest", "SVM", "XGBoost", "OracleRule"]

    label_tags = df_results["label_tag"].unique().tolist()
    n_panels   = len(label_tags)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, tag in zip(axes, label_tags):
        sub = df_results[df_results["label_tag"] == tag].copy()
        pipelines = ["clean", "leaky_discharge_location", "leaky_definitional", "oracle"]
        pipelines = [p for p in pipelines if p in sub["pipeline"].values]
        models    = [m for m in MODEL_ORDER if m in sub["model"].values]

        n_pipe  = len(pipelines)
        n_model = len(models)
        group_w = 0.8
        bar_w   = group_w / n_model

        x_pos = np.arange(n_pipe)

        for mi, model in enumerate(models):
            offsets = (mi - (n_model - 1) / 2) * bar_w
            vals, errs = [], []
            for pipe in pipelines:
                row = sub[(sub["pipeline"] == pipe) & (sub["model"] == model)]
                if row.empty:
                    vals.append(0)
                    errs.append(0)
                else:
                    vals.append(float(row["AUROC_mean"].iloc[0]))
                    errs.append(float(row["AUROC_sd"].fillna(0).iloc[0]))

            bars = ax.bar(
                x_pos + offsets, vals, bar_w * 0.9,
                label=model.replace("LogisticRegression", "LR")
                           .replace("RandomForest", "RF")
                           .replace("XGBoost", "XGB")
                           .replace("OracleRule", "Oracle"),
                color=[PALETTE.get(p, "#888888") for p in pipelines],
                alpha=0.85,
                yerr=errs,
                error_kw=dict(elinewidth=1.2, capsize=3),
            )

        ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.7, label="Chance (0.5)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [PIPELINE_LABELS.get(p, p) for p in pipelines],
            fontsize=9,
        )
        ax.set_ylim(0.40, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_title(
            f"eICU  —  {tag} ICU mortality label\n"
            f"(N={int(sub['n_total'].iloc[0]):,}, "
            f"prevalence={float(sub['prevalence'].iloc[0])*100:.1f}%)",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "External Leakage Audit: eICU Early Mortality Labels\n"
        "Definitional Antecedents Guarantee Near-Perfect Classification",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out = figdir / "external_eicu_auroc_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    out_png = figdir / "external_eicu_auroc_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {out}")
    print(f"  Figure saved → {out_png}")


if __name__ == "__main__":
    main()
