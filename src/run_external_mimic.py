"""
run_external_mimic.py — External real-world leakage audit on MIMIC-IV all-admissions.

Paper section: §6 External Real-World Leakage Audits — Cohort B

Context
-------
label_mortality = 1  iff  discharge_location == 'DIED'
Agreement with this rule: 99.92 % (11 mismatches in 14,081 rows).

This demonstrates proxy/target leakage: a post-outcome administrative variable
(discharge destination) functionally encodes in-hospital mortality status.
Including discharge_location in the feature set enables a classifier to achieve
AUROC ≈ 1.00 without learning any clinical signal.

Three pipelines are evaluated:
  1. clean    — admission covariates + labs only; discharge_location EXCLUDED
  2. leaky    — clean + discharge_location  (post-outcome proxy variable)
  3. oracle   — deterministic rule: score = 1[discharge_location == 'DIED']

Design matches the synthetic benchmark and the eICU external cohort:
  - Repeated stratified 5-fold CV, N_SEEDS independent random splits
  - Fold-sealed preprocessing: imputation → encoding → scaling → SMOTE inside fold
  - Same four classifiers: LR, RF, SVM (LinearSVC), XGB
  - Primary metric: AUROC; secondary: PR-AUC (important: prevalence = 2.5 %)
  - NOTE: class imbalance (358/14081) — class_weight='balanced' and SMOTE both applied

Usage
-----
    python src/run_external_mimic.py \\
        --data "External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv" \\
        --outdir results/tables \\
        --figdir results/figures \\
        --n_seeds 30 \\
        --n_folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
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

# ─────────────────────────────────────────────────────────────────────────────
# Feature specification
# ─────────────────────────────────────────────────────────────────────────────

# Lab item IDs present in the dataset
# These are MIMIC-IV itemids: all available numeric lab values
LAB_COLS = [
    "lab_50802", "lab_50820", "lab_50821", "lab_50861", "lab_50863",
    "lab_50868", "lab_50878", "lab_50882", "lab_50885", "lab_50893",
    "lab_50902", "lab_50912", "lab_50931", "lab_50960", "lab_50970",
    "lab_50971", "lab_50983", "lab_51006", "lab_51221", "lab_51222",
    "lab_51248", "lab_51249", "lab_51250", "lab_51265", "lab_51274",
    "lab_51275", "lab_51277", "lab_51279", "lab_51301", "lab_52172",
]

# Numeric pre-admission covariates
NUMERIC_CLEAN = ["anchor_age"] + LAB_COLS

# Categorical pre-admission covariates
CATEGORICAL_CLEAN = [
    "gender",            # M / F
    "race",              # 33 categories — ordinal encoded
    "marital_status",
    "insurance",
    "admission_type",
    "admission_location",
]

CLEAN_FEATURES = NUMERIC_CLEAN + CATEGORICAL_CLEAN

# Post-outcome leaky variable
# discharge_location includes 'DIED' which encodes mortality with 99.92 % agreement
LEAKY_CATEGORICAL = ["discharge_location"]
LEAKY_NUMERIC: list[str] = []

# Oracle rule
ORACLE_DIED_VALUE = "DIED"


# ─────────────────────────────────────────────────────────────────────────────
# Fold-sealed preprocessor (mirrors eICU version; self-contained here)
# ─────────────────────────────────────────────────────────────────────────────

class FoldSealedPreprocessorMIMIC:
    """
    Full fold-sealed preprocessing for MIMIC data:
      1. Median imputation for numeric columns (fit on train only)
      2. Most-frequent imputation for categorical columns (fit on train only)
      3. OrdinalEncoder for categorical columns (fit on train only)
      4. Winsorisation 1–99 % for all numeric columns (fit on train only)
      5. StandardScaler (fit on train only)
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
            # Use object dtype so SimpleImputer accepts string values
            X_cat = df_train[cat_cols].astype(object).values
            X_cat = self._cat_imputer.fit_transform(X_cat)
            # Re-wrap as DataFrame so OrdinalEncoder receives labelled columns
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
            X_cat = df_test[self._cat_cols].astype(object).values
            X_cat = self._cat_imputer.transform(X_cat)
            X_cat = pd.DataFrame(X_cat, columns=self._cat_cols).astype(object).values
            X_cat = self._encoder.transform(X_cat)
            X = np.hstack([X_num, X_cat])
        else:
            X = X_num

        X = self._winsorizer.transform(X)
        X = self._scaler.transform(X)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Model factory  (mirrors run_clean_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
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


# ─────────────────────────────────────────────────────────────────────────────
# Oracle AUROC
# ─────────────────────────────────────────────────────────────────────────────

def oracle_auroc(df: pd.DataFrame, label_col: str) -> dict:
    """
    Oracle score = 1[discharge_location == 'DIED'].
    Agreement with label_mortality ≈ 99.92 %.
    """
    y_true = df[label_col].values.astype(int)
    oracle_score = (
        df["discharge_location"].fillna("").str.strip().str.upper() == "DIED"
    ).astype(int).values

    try:
        auroc = float(roc_auc_score(y_true, oracle_score))
    except ValueError:
        auroc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, oracle_score))
    except ValueError:
        pr_auc = float("nan")

    agreement = float(np.mean(oracle_score == y_true))
    tp = int(np.sum((oracle_score == 1) & (y_true == 1)))
    fp = int(np.sum((oracle_score == 1) & (y_true == 0)))
    fn = int(np.sum((oracle_score == 0) & (y_true == 1)))
    tn = int(np.sum((oracle_score == 0) & (y_true == 0)))

    return {
        "pipeline":        "oracle",
        "model":           "OracleRule",
        "AUROC_mean":      round(auroc, 4),
        "AUROC_sd":        0.0,
        "PR_AUC_mean":     round(pr_auc, 4),
        "PR_AUC_sd":       0.0,
        "DELTA_AUC":       None,
        "n_seeds":         1,
        "oracle_agreement":round(agreement, 6),
        "oracle_tp":       tp,
        "oracle_fp":       fp,
        "oracle_fn":       fn,
        "oracle_tn":       tn,
        "note":            "Deterministic rule: discharge_location == 'DIED'",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core CV evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv_pipeline(
    df: pd.DataFrame,
    label_col: str,
    num_extra: list[str],
    cat_extra: list[str],
    pipeline_name: str,
    n_folds: int,
    n_seeds: int,
    smote_k: int = 5,
) -> list[dict]:
    """
    Repeated stratified k-fold CV for one pipeline.

    SMOTE is applied inside each training fold (never before splitting).
    Fold-sealed preprocessing ensures no data leakage from preprocessing.
    """
    y_all = df[label_col].values.astype(int)

    num_cols = [c for c in NUMERIC_CLEAN + num_extra  if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_CLEAN + cat_extra if c in df.columns]

    models = build_models()
    seed_records: list[dict] = []

    seed_pbar = tqdm(
        range(1, n_seeds + 1),
        desc=f"  [{pipeline_name}] seeds",
        ncols=100,
        colour="green",
        leave=True,
    )

    for seed in seed_pbar:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        model_fold_aurocs: dict[str, list[float]] = {m: [] for m in models}
        model_fold_praucs: dict[str, list[float]] = {m: [] for m in models}

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(np.zeros(len(df)), y_all)
        ):
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test  = df.iloc[test_idx].reset_index(drop=True)
            y_train  = y_all[train_idx]
            y_test   = y_all[test_idx]

            if len(np.unique(y_test)) < 2:
                continue

            # ── Fold-sealed preprocessing ─────────────────────────────────
            prep = FoldSealedPreprocessorMIMIC()
            X_train = prep.fit_transform(df_train, num_cols, cat_cols)
            X_test  = prep.transform(df_test)

            # ── SMOTE inside training fold ────────────────────────────────
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            if n_pos >= smote_k + 1 and n_neg >= smote_k + 1:
                try:
                    sm = SMOTE(k_neighbors=smote_k, random_state=seed)
                    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
                except Exception:
                    X_train_bal, y_train_bal = X_train, y_train
            else:
                X_train_bal, y_train_bal = X_train, y_train

            # ── Fit & evaluate ────────────────────────────────────────────
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

        # Aggregate across folds
        for model_name in models:
            aucs = np.array([v for v in model_fold_aurocs[model_name]
                             if not np.isnan(v)])
            prs  = np.array([v for v in model_fold_praucs[model_name]
                             if not np.isnan(v)])
            if len(aucs) == 0:
                continue
            seed_records.append({
                "pipeline":         pipeline_name,
                "model":            model_name,
                "seed":             seed,
                "AUROC_mean_folds": float(np.mean(aucs)),
                "PR_AUC_mean_folds":float(np.mean(prs)) if len(prs) else float("nan"),
                "n_folds_valid":    len(aucs),
            })

        # Show RF progress
        rf_this = [r["AUROC_mean_folds"] for r in seed_records
                   if r["seed"] == seed and r["model"] == "RandomForest"]
        if rf_this:
            seed_pbar.set_postfix(RF_AUROC=f"{rf_this[-1]:.4f}", refresh=True)

    return seed_records


def aggregate_seed_records(records: list[dict]) -> list[dict]:
    import itertools
    pipelines = sorted({r["pipeline"] for r in records})
    models    = sorted({r["model"]    for r in records})
    rows = []
    for pipe, mod in itertools.product(pipelines, models):
        vals = [r["AUROC_mean_folds"] for r in records
                if r["pipeline"] == pipe and r["model"] == mod]
        pr_vals = [r["PR_AUC_mean_folds"] for r in records
                   if r["pipeline"] == pipe and r["model"] == mod]
        if not vals:
            continue
        arr    = np.array(vals)
        pr_arr = np.array([v for v in pr_vals if not np.isnan(v)])
        rows.append({
            "pipeline":    pipe,
            "model":       mod,
            "AUROC_mean":  round(float(np.mean(arr)), 4),
            "AUROC_sd":    round(float(np.std(arr, ddof=1)), 4),
            "PR_AUC_mean": round(float(np.mean(pr_arr)), 4) if len(pr_arr) else float("nan"),
            "PR_AUC_sd":   round(float(np.std(pr_arr, ddof=1)), 4) if len(pr_arr) > 1 else float("nan"),
            "DELTA_AUC":   None,
            "n_seeds":     len(arr),
        })
    return rows


def attach_delta_auc(rows: list[dict]) -> list[dict]:
    clean_auc = {r["model"]: r["AUROC_mean"] for r in rows if r["pipeline"] == "clean"}
    for r in rows:
        base = clean_auc.get(r["model"])
        if base is not None and r["pipeline"] != "clean":
            r["DELTA_AUC"] = round(r["AUROC_mean"] - base, 4)
        else:
            r["DELTA_AUC"] = 0.0
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="External leakage audit on MIMIC-IV all-admissions mortality."
    )
    parser.add_argument(
        "--data",
        default="External Cohort/Dataset/full_analytic_dataset_mortality_all_admissions.csv",
        help="Path to MIMIC all-admissions CSV",
    )
    parser.add_argument("--outdir",  default="results/tables",  help="Output CSV directory")
    parser.add_argument("--figdir",  default="results/figures", help="Output figure directory")
    parser.add_argument("--n_seeds", type=int, default=30,
                        help="Number of random seeds for repeated CV (default: 30)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("Loading MIMIC dataset …")
    df = pd.read_csv(
        args.data,
        na_values=["", "nan", "NaN", "NA", "N/A"],
        keep_default_na=True,
    )
    print(f"  MIMIC loaded: {len(df):,} rows")
    print(f"  label_mortality: {int(df['label_mortality'].sum())} positives "
          f"({df['label_mortality'].mean()*100:.1f} %)")

    label_col = "label_mortality"

    # ── Oracle ────────────────────────────────────────────────────────────
    print("\nComputing oracle score …")
    oracle_row = oracle_auroc(df, label_col)
    print(f"  Oracle AUROC      = {oracle_row['AUROC_mean']:.4f}")
    print(f"  Oracle PR-AUC     = {oracle_row['PR_AUC_mean']:.4f}")
    print(f"  Oracle agreement  = {oracle_row['oracle_agreement']:.6f}")
    print(f"  TP={oracle_row['oracle_tp']}  FP={oracle_row['oracle_fp']}  "
          f"FN={oracle_row['oracle_fn']}  TN={oracle_row['oracle_tn']}")

    all_records: list[dict] = []

    # ── Pipeline 1: clean ─────────────────────────────────────────────────
    print(f"\nPipeline 1/2: clean  (N_SEEDS={args.n_seeds}, N_FOLDS={args.n_folds})")
    print(f"  Features: {len(NUMERIC_CLEAN)} numeric + {len(CATEGORICAL_CLEAN)} categorical")
    clean_records = run_cv_pipeline(
        df, label_col,
        num_extra=[], cat_extra=[],
        pipeline_name="clean",
        n_folds=args.n_folds, n_seeds=args.n_seeds,
    )
    all_records.extend(clean_records)

    # ── Pipeline 2: leaky (discharge_location) ────────────────────────────
    print(f"\nPipeline 2/2: leaky  (adds discharge_location)")
    print(f"  discharge_location == 'DIED' encodes mortality with 99.92 % agreement")
    leaky_records = run_cv_pipeline(
        df, label_col,
        num_extra=LEAKY_NUMERIC,
        cat_extra=LEAKY_CATEGORICAL,
        pipeline_name="leaky_discharge_location",
        n_folds=args.n_folds, n_seeds=args.n_seeds,
    )
    all_records.extend(leaky_records)

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg_rows = aggregate_seed_records(all_records)
    agg_rows = attach_delta_auc(agg_rows)

    # Append oracle
    oracle_row["DELTA_AUC"] = round(
        oracle_row["AUROC_mean"] - float(np.mean([
            r["AUROC_mean"] for r in agg_rows if r["pipeline"] == "clean"
        ])),
        4,
    )
    agg_rows.append(oracle_row)

    for r in agg_rows:
        r["cohort"]     = "MIMIC"
        r["label"]      = label_col
        r["label_tag"]  = "in_hospital_mortality"
        r["n_total"]    = len(df)
        r["n_positive"] = int(df[label_col].sum())
        r["prevalence"] = round(float(df[label_col].mean()), 4)

    df_out = pd.DataFrame(agg_rows)

    out_path = outdir / "external_mimic_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    seed_df = pd.DataFrame(all_records)
    seed_df["cohort"] = "MIMIC"
    seed_df["label"]  = label_col
    seed_path = outdir / "external_mimic_seed_records.csv"
    seed_df.to_csv(seed_path, index=False)
    print(f"Seed records saved → {seed_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Pipeline':<35} {'Model':<22} {'AUROC':>8} {'±SD':>7} {'ΔAUC':>8}")
    print(f"{'─'*65}")
    for pipe in ["clean", "leaky_discharge_location", "oracle"]:
        pipe_rows = [r for r in agg_rows if r["pipeline"] == pipe]
        for r in sorted(pipe_rows, key=lambda x: x.get("model", "")):
            delta = r.get("DELTA_AUC")
            delta_str = f"{delta:+.4f}" if delta is not None else "   —  "
            print(f"  {r['pipeline']:<33} {r.get('model',''):<22} "
                  f"{r['AUROC_mean']:>8.4f} {r.get('AUROC_sd', 0.0):>7.4f} "
                  f"{delta_str:>8}")
    print(f"{'─'*65}")

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        _plot_results(df_out, figdir)
    except Exception as e:
        print(f"  Warning: plotting failed ({e}). Results CSV still saved.")

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_results(df_results: pd.DataFrame, figdir: Path) -> None:
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
        "clean":                    "#2C7BB6",
        "leaky_discharge_location": "#D7191C",
        "oracle":                   "#1A1A1A",
    }
    PIPELINE_LABELS = {
        "clean":                    "Clean\n(admission vars + labs)",
        "leaky_discharge_location": "Leaky\n(+ discharge_location)",
        "oracle":                   "Oracle\n(discharge_location == 'DIED')",
    }
    MODEL_ORDER = ["LogisticRegression", "RandomForest", "SVM", "XGBoost", "OracleRule"]
    MODEL_LABELS = {
        "LogisticRegression": "LR",
        "RandomForest":       "RF",
        "SVM":                "SVM",
        "XGBoost":            "XGB",
        "OracleRule":         "Oracle",
    }

    pipelines = ["clean", "leaky_discharge_location", "oracle"]
    pipelines = [p for p in pipelines if p in df_results["pipeline"].values]
    models    = [m for m in MODEL_ORDER if m in df_results["model"].values]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: AUROC grouped bar ─────────────────────────────────────────
    ax = axes[0]
    n_pipe  = len(pipelines)
    n_model = len(models)
    group_w = 0.8
    bar_w   = group_w / n_model
    x_pos   = np.arange(n_pipe)

    for mi, model in enumerate(models):
        offsets = (mi - (n_model - 1) / 2) * bar_w
        vals, errs, colours = [], [], []
        for pipe in pipelines:
            row = df_results[
                (df_results["pipeline"] == pipe) &
                (df_results["model"] == model)
            ]
            vals.append(float(row["AUROC_mean"].iloc[0]) if not row.empty else 0.0)
            errs.append(float(row["AUROC_sd"].fillna(0).iloc[0]) if not row.empty else 0.0)
            colours.append(PALETTE.get(pipe, "#888888"))

        ax.bar(
            x_pos + offsets, vals, bar_w * 0.9,
            label=MODEL_LABELS.get(model, model),
            color=colours, alpha=0.85,
            yerr=errs,
            error_kw=dict(elinewidth=1.2, capsize=3),
        )

    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines], fontsize=9)
    ax.set_ylim(0.40, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC by pipeline", fontsize=10)
    ax.legend(title="Model", fontsize=8, loc="lower right")

    # ── Panel B: ΔAUC bar (leaky vs clean) ───────────────────────────────
    ax2 = axes[1]
    leaky_rows = df_results[df_results["pipeline"] == "leaky_discharge_location"].copy()
    models_b   = [m for m in MODEL_ORDER if m in leaky_rows["model"].values]

    bar_colours = ["#D7191C"] * len(models_b)
    delta_vals  = []
    delta_errs  = []
    for model in models_b:
        row = leaky_rows[leaky_rows["model"] == model]
        delta_vals.append(float(row["DELTA_AUC"].iloc[0]) if not row.empty else 0.0)
        delta_errs.append(float(row["AUROC_sd"].fillna(0).iloc[0]) if not row.empty else 0.0)

    x_b = np.arange(len(models_b))
    bars = ax2.bar(
        x_b, delta_vals, 0.55,
        color=bar_colours, alpha=0.85,
        yerr=delta_errs,
        error_kw=dict(elinewidth=1.2, capsize=4),
    )
    ax2.axhline(0, color="gray", lw=1, ls="--", alpha=0.6)
    ax2.set_xticks(x_b)
    ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_b], fontsize=10)
    ax2.set_ylabel("ΔAUC (leaky − clean)")
    ax2.set_title("AUC inflation from discharge_location", fontsize=10)

    # Annotate bars
    for bar, val in zip(bars, delta_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:+.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle(
        "External Leakage Audit: MIMIC-IV In-Hospital Mortality\n"
        "Post-Outcome Discharge Location Encodes the Label (99.92 % Agreement)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    for ext in (".pdf", ".png"):
        out = figdir / f"external_mimic_auroc_comparison{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
