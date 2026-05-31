"""External leakage audit on CGMacros postprandial glucose cohort (Theorem 2 — algebraic leakage)."""

from __future__ import annotations

import argparse
import itertools
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
from sklearn.model_selection import GroupKFold
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

# Numeric features available at or before meal time (CLEAN — no outcome leakage)
NUMERIC_CLEAN_BASE = [
    "pre_meal_cgm",
    "calories", "carbs", "protein", "fat", "fiber",
    "hr", "mets",
    "hour_of_day",
    "age", "bmi", "gender_binary",
    "hba1c", "fasting_glucose", "insulin",
    "triglycerides", "cholesterol", "hdl", "ldl",
]
# Gut health score columns are detected at runtime from the loaded DataFrame
# (all columns starting with "gut_")

CATEGORICAL_CLEAN = [
    "meal_type",
]

LEAKY_PEAK_NUMERIC    = ["peak_cgm"]     # B: algebraic formula component
LEAKY_PEAK_CATEGORICAL: list[str] = []

LEAKY_RISE_NUMERIC    = ["glucose_rise"]  # Y: the outcome directly
LEAKY_RISE_CATEGORICAL: list[str] = []

LABEL_PERCENTILE = 75.0   # Q75 → ~25% label prevalence


class FoldSealedPreprocessorCGMacros:
    """
    Full fold-sealed preprocessing for CGMacros meal data:
      1. Median imputation for numeric columns (fit on train only)
      2. Most-frequent imputation for categorical columns (fit on train only)
      3. OrdinalEncoder for categorical columns (fit on train only)
      4. Winsorisation 1–99 % for all numeric columns (fit on train only)
      5. StandardScaler (fit on train only)

    The binary label is NOT computed here — it is derived inside the CV loop
    from Q75(glucose_rise_train) to maintain fold-sealed protocol.
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
            X_cat = df_train[cat_cols].copy().astype("object")
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

    def transform(self, df_test: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Must call fit_transform before transform.")

        X_num = df_test[self._num_cols].values.astype(float)
        X_num = self._num_imputer.transform(X_num)

        if self._cat_cols:
            X_cat = df_test[self._cat_cols].copy().astype("object")
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
    """Return {name: unfitted estimator} for all classifiers."""
    models: dict = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
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


def _xgb_scale_pos_weight(y_train: np.ndarray) -> float:
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def oracle_auroc_cgmacros(df: pd.DataFrame, n_folds: int) -> dict:
    """
    Compute oracle AUROC using glucose_rise (Y) as the deterministic score.

    Because the binary label is high_responder := 1[Y ≥ Q75(Y_train)], any model
    with direct access to Y achieves AUROC = 1.0000 by construction (perfect ranking).

    Uses a single deterministic GroupKFold pass (no randomisation needed — oracle
    is independent of seed; ML models are evaluated with seed-permuted GroupKFold).
    """
    groups = df["subject_id"].values
    gkf = GroupKFold(n_splits=n_folds)

    fold_aurocs: list[float] = []
    fold_praucs: list[float] = []
    fold_agreements: list[float] = []

    for train_idx, test_idx in gkf.split(df, groups=groups):
        y_rise_train = df["glucose_rise"].iloc[train_idx].values
        y_rise_test  = df["glucose_rise"].iloc[test_idx].values

        threshold = float(np.percentile(y_rise_train, LABEL_PERCENTILE))
        y_test    = (y_rise_test >= threshold).astype(int)

        if len(np.unique(y_test)) < 2:
            continue

        oracle_score = y_rise_test

        try:
            auroc = float(roc_auc_score(y_test, oracle_score))
            prauc = float(average_precision_score(y_test, oracle_score))
        except ValueError:
            auroc, prauc = float("nan"), float("nan")

        oracle_pred = (y_rise_test >= threshold).astype(int)
        agreement   = float(np.mean(oracle_pred == y_test))

        fold_aurocs.append(auroc)
        fold_praucs.append(prauc)
        fold_agreements.append(agreement)

    valid_aurocs = [v for v in fold_aurocs if not np.isnan(v)]
    valid_praucs = [v for v in fold_praucs  if not np.isnan(v)]

    return {
        "pipeline": "oracle",
        "model": "OracleRule",
        "AUROC_mean": round(float(np.mean(valid_aurocs)), 4) if valid_aurocs else float("nan"),
        "AUROC_sd":   round(float(np.std(valid_aurocs, ddof=1)), 4) if len(valid_aurocs) > 1 else 0.0,
        "PR_AUC_mean": round(float(np.mean(valid_praucs)), 4) if valid_praucs else float("nan"),
        "PR_AUC_sd":  0.0,
        "DELTA_AUC": None,
        "n_seeds": 1,
        "oracle_agreement": round(float(np.mean(fold_agreements)), 6),
        "note": (
            "Deterministic oracle: score = glucose_rise (Y); "
            "label = 1[Y >= Q75(Y_train_fold)] → AUROC = 1.0 by construction"
        ),
    }


def run_cv_pipeline(
    df: pd.DataFrame,
    gut_cols: list[str],
    num_extra: list[str],
    cat_extra: list[str],
    pipeline_name: str,
    n_folds: int,
    n_seeds: int,
    smote_k: int = 5,
) -> list[dict]:
    """
    Run repeated GroupKFold CV for ONE pipeline definition.

    GroupKFold split is made genuinely stochastic by permuting the subject-to-group
    mapping with a different RNG seed each iteration. This ensures each seed
    produces a different assignment of subjects to folds (equivalent to repeated CV).

    Parameters
    ----------
    df            : full cohort DataFrame (from cgmacros_meal_cohort.csv)
    gut_cols      : list of gut health columns detected at runtime
    num_extra     : ADDITIONAL numeric features to include (on top of base NUMERIC_CLEAN)
    cat_extra     : ADDITIONAL categorical features to include
    pipeline_name : label for output rows
    n_folds       : number of CV folds (5)
    n_seeds       : number of random seeds / subject-order permutations (30)
    smote_k       : k-neighbours for SMOTE (applied inside training fold only)

    Returns
    -------
    List of per-seed dicts with AUROC_mean_folds, PR_AUC_mean_folds
    """
    subjects = df["subject_id"].values
    all_subjects = np.unique(subjects)

    num_cols_base = NUMERIC_CLEAN_BASE + gut_cols + num_extra
    cat_cols_full  = CATEGORICAL_CLEAN + cat_extra

    num_cols = [c for c in num_cols_base if c in df.columns]
    cat_cols = [c for c in cat_cols_full  if c in df.columns]

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
        rng = np.random.default_rng(seed)

        shuffled_subjects = rng.permutation(all_subjects)
        subj_to_group = {int(s): int(i) for i, s in enumerate(shuffled_subjects)}
        groups_for_split = np.array([subj_to_group[int(s)] for s in subjects])

        gkf = GroupKFold(n_splits=n_folds)

        model_fold_aurocs: dict[str, list[float]] = {m: [] for m in models}
        model_fold_praucs: dict[str, list[float]] = {m: [] for m in models}

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(np.zeros(len(df)), groups=groups_for_split)
        ):
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test  = df.iloc[test_idx].reset_index(drop=True)

            y_rise_train = df_train["glucose_rise"].values
            y_rise_test  = df_test["glucose_rise"].values
            threshold    = float(np.percentile(y_rise_train, LABEL_PERCENTILE))

            y_train = (y_rise_train >= threshold).astype(int)
            y_test  = (y_rise_test  >= threshold).astype(int)

            if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
                continue

            prep = FoldSealedPreprocessorCGMacros()
            X_train = prep.fit_transform(df_train, num_cols, cat_cols)
            X_test  = prep.transform(df_test)

            n_pos_train = int(y_train.sum())
            n_neg_train = int(len(y_train) - n_pos_train)
            actual_k = min(smote_k, n_pos_train - 1, n_neg_train - 1)
            if actual_k >= 1:
                try:
                    sm = SMOTE(k_neighbors=actual_k, random_state=seed)
                    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
                except Exception:
                    X_train_bal, y_train_bal = X_train, y_train
            else:
                X_train_bal, y_train_bal = X_train, y_train

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
                "pipeline":          pipeline_name,
                "model":             model_name,
                "seed":              seed,
                "AUROC_mean_folds":  float(np.mean(aucs)),
                "PR_AUC_mean_folds": float(np.mean(prs)) if len(prs) > 0 else float("nan"),
                "n_folds_valid":     len(aucs),
            })

        # Show RF AUROC as representative live metric
        rf_this_seed = [
            r["AUROC_mean_folds"] for r in seed_records
            if r["seed"] == seed and r["model"] == "RandomForest"
        ]
        if rf_this_seed:
            seed_pbar.set_postfix(RF_AUROC=f"{rf_this_seed[-1]:.4f}", refresh=True)

    return seed_records


def aggregate_seed_records(records: list[dict]) -> list[dict]:
    """Aggregate per-seed records → one row per (pipeline, model)."""
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
            "pipeline":    pipe,
            "model":       mod,
            "AUROC_mean":  round(float(np.mean(arr)),  4),
            "AUROC_sd":    round(float(np.std(arr, ddof=1)), 4),
            "PR_AUC_mean": round(float(np.mean(pr_arr)), 4) if len(pr_arr) else float("nan"),
            "PR_AUC_sd":   round(float(np.std(pr_arr, ddof=1)), 4) if len(pr_arr) > 1 else float("nan"),
            "DELTA_AUC":   None,
            "n_seeds":     len(arr),
        })
    return rows


def attach_delta_auc(rows: list[dict]) -> list[dict]:
    """Fill DELTA_AUC = AUROC_mean(pipeline) − AUROC_mean('clean')."""
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


def _plot_results(df_results: pd.DataFrame, figdir: Path) -> None:
    """
    Grouped bar chart: one group per pipeline, bars per model.
    Error bars = ±1 SD across N_SEEDS.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
    })

    PALETTE = {
        "clean":               "#2C7BB6",
        "leaky_peak_cgm":      "#FDAE61",
        "leaky_glucose_rise":  "#D7191C",
        "oracle":              "#1A1A1A",
    }
    PIPELINE_LABELS = {
        "clean":               "Clean\n(pre-meal only)",
        "leaky_peak_cgm":      "Leaky A\n(peak_cgm = B)",
        "leaky_glucose_rise":  "Leaky B\n(glucose_rise = Y)",
        "oracle":              "Oracle\n(algebraic rule)",
    }
    MODEL_ORDER = ["LogisticRegression", "RandomForest", "SVM", "XGBoost", "OracleRule"]
    PIPELINE_ORDER = ["clean", "leaky_peak_cgm", "leaky_glucose_rise", "oracle"]

    pipelines = [p for p in PIPELINE_ORDER if p in df_results["pipeline"].values]
    models    = [m for m in MODEL_ORDER    if m in df_results["model"].values]

    n_pipe  = len(pipelines)
    n_model = len(models)
    group_w = 0.8
    bar_w   = group_w / n_model

    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos = np.arange(n_pipe)

    for mi, model in enumerate(models):
        offsets = (mi - (n_model - 1) / 2) * bar_w
        vals, errs, colors = [], [], []
        for pipe in pipelines:
            row = df_results[
                (df_results["pipeline"] == pipe) & (df_results["model"] == model)
            ]
            if row.empty:
                vals.append(0)
                errs.append(0)
            else:
                vals.append(float(row["AUROC_mean"].iloc[0]))
                errs.append(float(row["AUROC_sd"].fillna(0).iloc[0]))
            colors.append(PALETTE.get(pipe, "#888888"))

        ax.bar(
            x_pos + offsets, vals, bar_w * 0.88,
            label=model.replace("LogisticRegression", "LR")
                       .replace("RandomForest", "RF")
                       .replace("XGBoost", "XGB")
                       .replace("OracleRule", "Oracle"),
            color=colors,
            alpha=0.85,
            yerr=errs,
            error_kw=dict(elinewidth=1.2, capsize=3),
        )

    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.7, label="Chance (0.5)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [PIPELINE_LABELS.get(p, p) for p in pipelines], fontsize=9
    )
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("AUROC (mean ± SD across 30 seeds)")

    n_total = int(df_results["n_total"].iloc[0]) if "n_total" in df_results.columns else "~1698"
    n_subj  = int(df_results["n_subjects"].iloc[0]) if "n_subjects" in df_results.columns else 45
    ax.set_title(
        f"CGMacros External Leakage Audit — Postprandial Glucose Response\n"
        f"N={n_total:,} meals, {n_subj} subjects | Label: high_responder = 1[ΔGL ≥ Q75(ΔGL_train)]",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    fig.suptitle(
        "Algebraic Leakage (Theorem 2): Including peak_cgm or glucose_rise\n"
        "Inflates AUROC from ~0.60 (clean) to ~1.00 (leaky/oracle)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    out_pdf = figdir / "external_cgmacros_auroc_comparison.pdf"
    out_png = figdir / "external_cgmacros_auroc_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  Figure saved → {out_pdf}")
    tqdm.write(f"  Figure saved → {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "External leakage audit on CGMacros postprandial glucose cohort.\n"
            "Evaluates algebraic leakage (Theorem 2): Y = peak_cgm − pre_meal_cgm."
        )
    )
    parser.add_argument(
        "--data",
        default="results/tables/cgmacros_meal_cohort.csv",
        help="Path to cgmacros_meal_cohort.csv (output of build_cgmacros_cohort.py)",
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
    parser.add_argument(
        "--n_seeds", type=int, default=30,
        help="Number of random seeds / subject-order permutations (default: 30)",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of GroupKFold folds (default: 5)",
    )
    parser.add_argument(
        "--smote_k", type=int, default=5,
        help="SMOTE k-neighbours (default: 5; auto-reduced if minority class is small)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CGMacros External Leakage Audit")
    print("  Paired outcome: Y = peak_cgm − pre_meal_cgm (Theorem 2)")
    print("  CV design: Repeated GroupKFold (subjects as groups)")
    print("=" * 70)

    print(f"\nLoading cohort from: {args.data}")
    df = pd.read_csv(
        args.data,
        na_values=["", "nan", "NaN", "NA", "N/A"],
        keep_default_na=True,
    )
    print(f"  Loaded: {len(df):,} meal observations, {df['subject_id'].nunique()} subjects")
    print(f"  Glucose rise: mean={df['glucose_rise'].mean():.1f}, "
          f"SD={df['glucose_rise'].std():.1f}, "
          f"Q75={df['glucose_rise'].quantile(0.75):.1f} mg/dL")

    gut_cols = sorted([c for c in df.columns if c.startswith("gut_")])
    print(f"  Gut health columns detected: {len(gut_cols)}")

    n_total   = len(df)
    n_subj    = int(df["subject_id"].nunique())
    n_pos_glb = int((df["glucose_rise"] >= df["glucose_rise"].quantile(0.75)).sum())
    prev_glb  = round(n_pos_glb / n_total, 4)

    print(f"\n{'─'*70}")
    print(f"  N meals = {n_total:,}  |  N subjects = {n_subj}")
    print(f"  Global Q75 threshold = {df['glucose_rise'].quantile(0.75):.2f} mg/dL")
    print(f"  Approx label prevalence (global Q75) = {prev_glb*100:.1f}%")
    print(f"  GroupKFold: {args.n_folds} folds × {args.n_seeds} seed permutations")
    print(f"{'─'*70}\n")

    all_seed_records: list[dict] = []

    print("Computing oracle score (glucose_rise → AUROC = 1.0 by construction) …")
    oracle_row = oracle_auroc_cgmacros(df, args.n_folds)
    print(f"  Oracle AUROC = {oracle_row['AUROC_mean']:.4f}  "
          f"(agreement = {oracle_row['oracle_agreement']:.6f})")

    print(f"\nPipeline 1/3: clean  (N_SEEDS={args.n_seeds}, N_FOLDS={args.n_folds})")
    print(f"  Features: pre_meal_cgm + macros + activity + demographics + labs + "
          f"{len(gut_cols)} gut scores")
    clean_records = run_cv_pipeline(
        df, gut_cols,
        num_extra=[], cat_extra=[],
        pipeline_name="clean",
        n_folds=args.n_folds, n_seeds=args.n_seeds,
        smote_k=args.smote_k,
    )
    all_seed_records.extend(clean_records)

    print(f"\nPipeline 2/3: leaky_peak_cgm  (adds peak_cgm = B, the algebraic component)")
    leaky_peak_records = run_cv_pipeline(
        df, gut_cols,
        num_extra=LEAKY_PEAK_NUMERIC,
        cat_extra=LEAKY_PEAK_CATEGORICAL,
        pipeline_name="leaky_peak_cgm",
        n_folds=args.n_folds, n_seeds=args.n_seeds,
        smote_k=args.smote_k,
    )
    all_seed_records.extend(leaky_peak_records)

    print(f"\nPipeline 3/3: leaky_glucose_rise  (adds glucose_rise = Y, the outcome directly)")
    leaky_rise_records = run_cv_pipeline(
        df, gut_cols,
        num_extra=LEAKY_RISE_NUMERIC,
        cat_extra=LEAKY_RISE_CATEGORICAL,
        pipeline_name="leaky_glucose_rise",
        n_folds=args.n_folds, n_seeds=args.n_seeds,
        smote_k=args.smote_k,
    )
    all_seed_records.extend(leaky_rise_records)

    print("\nAggregating results …")
    agg_rows = aggregate_seed_records(all_seed_records)
    agg_rows = attach_delta_auc(agg_rows)

    clean_mean_auroc = np.mean([
        r["AUROC_mean"] for r in agg_rows if r["pipeline"] == "clean"
    ])
    oracle_row["DELTA_AUC"] = round(oracle_row["AUROC_mean"] - clean_mean_auroc, 4)
    agg_rows.append(oracle_row)

    for r in agg_rows:
        r["cohort"]     = "CGMacros"
        r["n_total"]    = n_total
        r["n_subjects"] = n_subj
        r["n_positive_approx"] = n_pos_glb
        r["prevalence"] = prev_glb
        r["label"]      = "high_responder_Q75"
        r["cv_design"]  = f"GroupKFold(k={args.n_folds}) × {args.n_seeds} seeds"
        r["oracle_threshold"] = "Q75(glucose_rise_train) — fold-specific"

    df_out = pd.DataFrame(agg_rows)

    out_path = outdir / "external_cgmacros_results.csv"
    df_out.to_csv(out_path, index=False)
    tqdm.write(f"\n  Saved → {out_path}")

    seed_df = pd.DataFrame(all_seed_records)
    seed_df["cohort"] = "CGMacros"
    seed_df["label"]  = "high_responder_Q75"
    seed_path = outdir / "external_cgmacros_seed_records.csv"
    seed_df.to_csv(seed_path, index=False)
    tqdm.write(f"  Saved → {seed_path}")

    print(f"\n{'─'*70}")
    print(f"{'Pipeline':<25} {'Model':<22} {'AUROC':>8} {'±SD':>7} {'ΔAUC':>8}")
    print(f"{'─'*70}")
    PIPE_ORDER = ["clean", "leaky_peak_cgm", "leaky_glucose_rise", "oracle"]
    for pipe in PIPE_ORDER:
        pipe_rows = [r for r in agg_rows if r["pipeline"] == pipe]
        for r in sorted(pipe_rows, key=lambda x: x.get("model", "")):
            delta = r.get("DELTA_AUC")
            delta_str = f"{delta:+.4f}" if delta is not None else "   —  "
            sd_str = f"{r.get('AUROC_sd', 0.0):.4f}"
            print(f"  {r['pipeline']:<23} {r.get('model',''):<22} "
                  f"{r['AUROC_mean']:>8.4f} {sd_str:>7} {delta_str:>8}")
    print(f"{'─'*70}")

    try:
        _plot_results(df_out, figdir)
    except Exception as e:
        print(f"  Warning: plotting failed ({e}). Results CSV still saved.")

    print("\nDone.")
    print(f"\nNext step:")
    print(f"  python src/run_cgmacros_shap.py "
          f"--data {args.data} --outdir {args.outdir} --figdir {args.figdir}")


if __name__ == "__main__":
    main()
