"""run_cgmacros_subject_uncertainty.py — Subject-level LOSO and bootstrap uncertainty analysis for CGMacros."""

from __future__ import annotations

import argparse
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.base
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

NUMERIC_CLEAN_BASE = [
    "pre_meal_cgm",
    "calories", "carbs", "protein", "fat", "fiber",
    "hr", "mets",
    "hour_of_day",
    "age", "bmi", "gender_binary",
    "hba1c", "fasting_glucose", "insulin",
    "triglycerides", "cholesterol", "hdl", "ldl",
]
CATEGORICAL_CLEAN = ["meal_type"]
LEAKY_PEAK_NUMERIC = ["peak_cgm"]

LABEL_PERCENTILE = 75.0
SMOTE_K          = 5
MIN_TEST_POS     = 1


class FoldSealedPrep:
    def __init__(self):
        self._num_imp = SimpleImputer(strategy="median")
        self._cat_imp = SimpleImputer(strategy="most_frequent")
        self._enc     = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._wins    = FoldSealedWinsorizer(lower_pct=1.0, upper_pct=99.0)
        self._scaler  = FoldSealedScaler()
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []

    def fit_transform(self, df_tr: pd.DataFrame,
                      num_cols: list[str], cat_cols: list[str]) -> np.ndarray:
        self._num_cols, self._cat_cols = num_cols, cat_cols
        X = df_tr[num_cols].values.astype(float)
        X = self._num_imp.fit_transform(X)
        if cat_cols:
            Xc = df_tr[cat_cols].copy().astype("object")
            Xc = Xc.where(pd.notna(Xc), np.nan)
            Xc = self._cat_imp.fit_transform(Xc)
            Xc = pd.DataFrame(Xc, columns=cat_cols).astype(object).values
            Xc = self._enc.fit_transform(Xc)
            X  = np.hstack([X, Xc])
        X = self._wins.fit_transform(X)
        X = self._scaler.fit_transform(X)
        return X

    def transform(self, df_te: pd.DataFrame) -> np.ndarray:
        X = df_te[self._num_cols].values.astype(float)
        X = self._num_imp.transform(X)
        if self._cat_cols:
            Xc = df_te[self._cat_cols].copy().astype("object")
            Xc = Xc.where(pd.notna(Xc), np.nan)
            Xc = self._cat_imp.transform(Xc)
            Xc = pd.DataFrame(Xc, columns=self._cat_cols).astype(object).values
            Xc = self._enc.transform(Xc)
            X  = np.hstack([X, Xc])
        X = self._wins.transform(X)
        X = self._scaler.transform(X)
        return X


def build_models(loso_mode: bool = False) -> dict:
    """loso_mode=True excludes SVM (too slow for 45 single-subject evaluations)."""
    models: dict = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, verbosity=0, n_jobs=-1,
        )
    if not loso_mode:
        models["SVM"] = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42),
            cv=3, method="sigmoid",
        )
    return models


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def _fit_predict(clf, X_tr, y_tr, X_te, seed=42):
    """Clone, fit, return predict_proba[:,1]. Returns None on failure."""
    clf = sklearn.base.clone(clf)
    try:
        if HAS_XGB and isinstance(clf, XGBClassifier):
            clf.set_params(scale_pos_weight=_scale_pos_weight(y_tr))
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    except Exception as e:
        return None


def _smote_train(X_tr, y_tr, seed):
    """Apply SMOTE inside training fold; fall back to original on failure."""
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    k = min(SMOTE_K, n_pos - 1, n_neg - 1)
    if k < 1:
        return X_tr, y_tr
    try:
        sm = SMOTE(k_neighbors=k, random_state=seed)
        return sm.fit_resample(X_tr, y_tr)
    except Exception:
        return X_tr, y_tr


def get_feature_cols(df: pd.DataFrame, extra_num: list[str]) -> tuple[list, list]:
    gut_cols = [c for c in df.columns if c.startswith("gut_")]
    num_cols = [c for c in NUMERIC_CLEAN_BASE + gut_cols + extra_num if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_CLEAN if c in df.columns]
    return num_cols, cat_cols


def run_loso(
    df: pd.DataFrame,
    pipeline_name: str,
    extra_num: list[str],
    models: dict,
) -> list[dict]:
    """Train on all other subjects' meals, test on held-out subject. Fold-sealed label threshold."""
    subjects = np.unique(df["subject_id"].values)
    num_cols, cat_cols = get_feature_cols(df, extra_num)
    records: list[dict] = []

    for subj in tqdm(subjects, desc=f"  LOSO [{pipeline_name}]", ncols=100):
        train_mask = df["subject_id"] != subj
        df_train   = df[train_mask].reset_index(drop=True)
        df_test    = df[~train_mask].reset_index(drop=True)

        threshold = float(np.percentile(df_train["glucose_rise"].values, LABEL_PERCENTILE))
        y_train   = (df_train["glucose_rise"].values >= threshold).astype(int)
        y_test    = (df_test["glucose_rise"].values  >= threshold).astype(int)

        if y_test.sum() < MIN_TEST_POS or (len(y_test) - y_test.sum()) < 1:
            continue

        prep    = FoldSealedPrep()
        X_train = prep.fit_transform(df_train, num_cols, cat_cols)
        X_test  = prep.transform(df_test)

        X_tr_bal, y_tr_bal = _smote_train(X_train, y_train, seed=42)

        for model_name, clf in models.items():
            probs = _fit_predict(clf, X_tr_bal, y_tr_bal, X_test)
            if probs is None or len(np.unique(y_test)) < 2:
                auroc = float("nan")
            else:
                try:
                    auroc = float(roc_auc_score(y_test, probs))
                except Exception:
                    auroc = float("nan")

            records.append({
                "pipeline":  pipeline_name,
                "model":     model_name,
                "subject_id": int(subj),
                "n_test_meals": len(df_test),
                "n_test_pos":   int(y_test.sum()),
                "threshold_mg_dl": round(threshold, 4),
                "AUROC":     round(auroc, 6) if not np.isnan(auroc) else float("nan"),
            })

    return records


def summarise_loso(records: list[dict]) -> list[dict]:
    """Aggregate LOSO records → mean, SD, 95% exact CI per (pipeline, model)."""
    df = pd.DataFrame(records).dropna(subset=["AUROC"])
    rows = []
    for (pipe, mod), grp in df.groupby(["pipeline", "model"]):
        aurocs = grp["AUROC"].values
        n      = len(aurocs)
        mean_  = float(np.mean(aurocs))
        sd_    = float(np.std(aurocs, ddof=1)) if n > 1 else 0.0
        # Exact 95% CI from percentiles (no distribution assumption)
        ci_lo  = float(np.percentile(aurocs, 2.5))
        ci_hi  = float(np.percentile(aurocs, 97.5))
        rows.append({
            "pipeline": pipe,
            "model":    mod,
            "n_subjects": n,
            "AUROC_mean": round(mean_, 4),
            "AUROC_sd":   round(sd_, 4),
            "CI_95_lo":   round(ci_lo, 4),
            "CI_95_hi":   round(ci_hi, 4),
            "CI_width":   round(ci_hi - ci_lo, 4),
        })
    return rows


def run_subject_bootstrap(
    df: pd.DataFrame,
    pipeline_name: str,
    extra_num: list[str],
    models: dict,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Subject-level OOB bootstrap CI. Each iteration samples subjects with replacement."""
    subjects   = np.unique(df["subject_id"].values)
    N_subjects = len(subjects)
    num_cols, cat_cols = get_feature_cols(df, extra_num)

    rng     = np.random.default_rng(seed)
    records: list[dict] = []

    for b in tqdm(range(n_bootstrap),
                  desc=f"  Bootstrap [{pipeline_name}]", ncols=100):
        drawn = rng.choice(subjects, size=N_subjects, replace=True)
        drawn_set = set(drawn.tolist())
        oob_set   = set(subjects.tolist()) - drawn_set

        if len(oob_set) < 5:
            continue

        train_parts = []
        for s in drawn:
            train_parts.append(df[df["subject_id"] == s])
        df_train = pd.concat(train_parts, ignore_index=True)
        df_test  = df[df["subject_id"].isin(oob_set)].reset_index(drop=True)

        threshold = float(np.percentile(df_train["glucose_rise"].values, LABEL_PERCENTILE))
        y_train   = (df_train["glucose_rise"].values >= threshold).astype(int)
        y_test    = (df_test["glucose_rise"].values  >= threshold).astype(int)

        if y_test.sum() < MIN_TEST_POS or (len(y_test) - y_test.sum()) < 1:
            continue
        if y_train.sum() < 1 or (len(y_train) - y_train.sum()) < 1:
            continue

        prep    = FoldSealedPrep()
        X_train = prep.fit_transform(df_train, num_cols, cat_cols)
        X_test  = prep.transform(df_test)

        X_tr_bal, y_tr_bal = _smote_train(X_train, y_train, seed=int(b))

        for model_name, clf in models.items():
            probs = _fit_predict(clf, X_tr_bal, y_tr_bal, X_test)
            if probs is None or len(np.unique(y_test)) < 2:
                auroc = float("nan")
            else:
                try:
                    auroc = float(roc_auc_score(y_test, probs))
                except Exception:
                    auroc = float("nan")

            records.append({
                "pipeline":         pipeline_name,
                "model":            model_name,
                "bootstrap_iter":   b,
                "n_oob_subjects":   len(oob_set),
                "n_oob_meals":      len(df_test),
                "threshold_mg_dl":  round(threshold, 4),
                "AUROC":            round(auroc, 6) if not np.isnan(auroc) else float("nan"),
            })

    return records


def summarise_bootstrap(records: list[dict]) -> list[dict]:
    """Aggregate bootstrap records → 95% CI per (pipeline, model)."""
    df = pd.DataFrame(records).dropna(subset=["AUROC"])
    rows = []
    for (pipe, mod), grp in df.groupby(["pipeline", "model"]):
        aurocs = grp["AUROC"].values
        n_valid = len(aurocs)
        rows.append({
            "pipeline":       pipe,
            "model":          mod,
            "n_valid_iters":  n_valid,
            "AUROC_mean":     round(float(np.mean(aurocs)), 4),
            "AUROC_sd":       round(float(np.std(aurocs, ddof=1)), 4),
            "CI_95_lo":       round(float(np.percentile(aurocs, 2.5)), 4),
            "CI_95_hi":       round(float(np.percentile(aurocs, 97.5)), 4),
            "CI_width":       round(float(np.percentile(aurocs, 97.5)
                                          - np.percentile(aurocs, 2.5)), 4),
        })
    return rows


MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_LABELS = {
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "XGBoost":            "XGB",
}
PIPE_COLORS = {
    "clean":           "#2E86AB",
    "leaky_peak_cgm":  "#E84855",
}
PIPE_LABELS = {
    "clean":           "Clean",
    "leaky_peak_cgm":  "Leaky (peak\_cgm)",
}


def make_figure(
    loso_summary: list[dict],
    boot_summary: list[dict],
    loso_records: list[dict],
    figdir: Path,
) -> None:
    """Two-panel figure: LOSO strip plot (left) and bootstrap 95% CI intervals (right)."""
    loso_df  = pd.DataFrame(loso_records).dropna(subset=["AUROC"])
    loso_sum = pd.DataFrame(loso_summary)
    boot_sum = pd.DataFrame(boot_summary)

    models_present = [m for m in MODEL_ORDER if m in loso_df["model"].unique()]
    pipelines = ["clean", "leaky_peak_cgm"]

    fig = plt.figure(figsize=(13, 5.5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    n_mod  = len(models_present)
    n_pipe = len(pipelines)
    group_width = 0.7
    bar_width   = group_width / n_pipe
    offsets     = np.linspace(-(group_width - bar_width) / 2,
                              (group_width - bar_width) / 2, n_pipe)

    for pi, pipe in enumerate(pipelines):
        color = PIPE_COLORS[pipe]
        for mi, mod in enumerate(models_present):
            x0 = mi + offsets[pi]
            sub = loso_df[(loso_df["pipeline"] == pipe) & (loso_df["model"] == mod)]["AUROC"].values
            jitter = np.random.default_rng(0).uniform(-0.06, 0.06, size=len(sub))
            ax1.scatter(x0 + jitter, sub, s=14, color=color, alpha=0.4, zorder=2)
            row = loso_sum[(loso_sum["pipeline"] == pipe) & (loso_sum["model"] == mod)]
            if not row.empty:
                mean_ = float(row["AUROC_mean"].iloc[0])
                ci_lo = float(row["CI_95_lo"].iloc[0])
                ci_hi = float(row["CI_95_hi"].iloc[0])
                ax1.plot([x0, x0], [ci_lo, ci_hi], color=color, lw=2.2, zorder=3)
                ax1.scatter(x0, mean_, s=60, color=color, zorder=4,
                            label=PIPE_LABELS[pipe] if mi == 0 else "_nolegend_")

    ax1.set_xticks(range(n_mod))
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present], fontsize=11)
    ax1.set_ylabel("AUROC (LOSO)", fontsize=11)
    ax1.set_title("Leave-One-Subject-Out\n(N = 45 subjects)", fontsize=11)
    ax1.set_ylim(0.30, 1.05)
    ax1.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax1.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax1.spines[["top", "right"]].set_visible(False)

    for pi, pipe in enumerate(pipelines):
        color  = PIPE_COLORS[pipe]
        for mi, mod in enumerate(models_present):
            x0  = mi + offsets[pi]
            row = boot_sum[(boot_sum["pipeline"] == pipe) & (boot_sum["model"] == mod)]
            if row.empty:
                continue
            mean_ = float(row["AUROC_mean"].iloc[0])
            ci_lo = float(row["CI_95_lo"].iloc[0])
            ci_hi = float(row["CI_95_hi"].iloc[0])
            ax2.plot([x0, x0], [ci_lo, ci_hi], color=color, lw=2.2, zorder=3,
                     solid_capstyle="round")
            ax2.scatter(x0, mean_, s=60, color=color, zorder=4,
                        label=PIPE_LABELS[pipe] if mi == 0 else "_nolegend_")
            ax2.bar(x0, ci_hi - ci_lo, bottom=ci_lo, width=bar_width * 0.55,
                    color=color, alpha=0.18, zorder=1)

    ax2.set_xticks(range(n_mod))
    ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present], fontsize=11)
    ax2.set_ylabel("AUROC (Subject Bootstrap)", fontsize=11)
    ax2.set_title("Subject-Level Bootstrap\n95% CI (B = OOB replicates)", fontsize=11)
    ax2.set_ylim(0.30, 1.05)
    ax2.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "CGMacros Subject-Level Uncertainty Analysis (N = 45 subjects)\n"
        "Clean vs. Leaky (peak\_cgm) pipelines",
        fontsize=11, y=1.01,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fp = figdir / f"cgmacros_subject_uncertainty.{ext}"
        fig.savefig(fp, bbox_inches="tight", dpi=200)
        print(f"  Saved: {fp}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Subject-level uncertainty analysis for CGMacros (LOSO + Bootstrap)"
    )
    p.add_argument("--data",        default="results/tables/cgmacros_meal_cohort.csv",
                   help="Path to CGMacros meal cohort CSV")
    p.add_argument("--outdir",      default="results/tables",
                   help="Output directory for CSVs")
    p.add_argument("--figdir",      default="results/figures",
                   help="Output directory for figures")
    p.add_argument("--n_bootstrap", type=int, default=500,
                   help="Number of subject-bootstrap replicates (default 500)")
    p.add_argument("--seed",        type=int, default=42,
                   help="Master random seed for bootstrap")
    p.add_argument("--skip_loso",   action="store_true",
                   help="Skip LOSO (only run bootstrap)")
    p.add_argument("--skip_bootstrap", action="store_true",
                   help="Skip bootstrap (only run LOSO)")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir);  outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir);  figdir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"  Loaded: {len(df):,} rows, {df['subject_id'].nunique()} subjects")

    pipelines = {
        "clean":          [],
        "leaky_peak_cgm": LEAKY_PEAK_NUMERIC,
    }

    loso_models = build_models(loso_mode=True)
    boot_models = build_models(loso_mode=True)

    all_loso_records: list[dict] = []
    if not args.skip_loso:
        print("\n── Leave-One-Subject-Out (LOSO) ─────────────────────────────────")
        for pipe_name, extra_num in pipelines.items():
            recs = run_loso(df, pipe_name, extra_num, loso_models)
            all_loso_records.extend(recs)

        loso_df = pd.DataFrame(all_loso_records)
        loso_df.to_csv(outdir / "cgmacros_loso_results.csv", index=False)
        print(f"\n  Saved: {outdir}/cgmacros_loso_results.csv ({len(loso_df)} rows)")

        loso_summary = summarise_loso(all_loso_records)
        loso_sum_df  = pd.DataFrame(loso_summary)
        loso_sum_df.to_csv(outdir / "cgmacros_loso_summary.csv", index=False)
        print(f"  Saved: {outdir}/cgmacros_loso_summary.csv")

        print("\n  LOSO Summary:")
        print(loso_sum_df.to_string(index=False))

    all_boot_records: list[dict] = []
    if not args.skip_bootstrap:
        print(f"\n── Subject Bootstrap (B={args.n_bootstrap}) ─────────────────────────")
        for pipe_name, extra_num in pipelines.items():
            recs = run_subject_bootstrap(
                df, pipe_name, extra_num, boot_models,
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
            all_boot_records.extend(recs)

        boot_df = pd.DataFrame(all_boot_records)
        boot_df.to_csv(outdir / "cgmacros_bootstrap_ci.csv", index=False)
        print(f"\n  Saved: {outdir}/cgmacros_bootstrap_ci.csv ({len(boot_df)} rows)")

        boot_summary = summarise_bootstrap(all_boot_records)
        boot_sum_df  = pd.DataFrame(boot_summary)
        boot_sum_df.to_csv(outdir / "cgmacros_bootstrap_ci_summary.csv", index=False)
        print(f"  Saved: {outdir}/cgmacros_bootstrap_ci_summary.csv")

        print("\n  Bootstrap CI Summary:")
        print(boot_sum_df.to_string(index=False))

    if not args.skip_loso and not args.skip_bootstrap:
        print("\n── Generating figure ────────────────────────────────────────────")
        loso_summary_list = loso_summary        # type: ignore[possibly-undefined]
        boot_summary_list = boot_summary        # type: ignore[possibly-undefined]
        make_figure(loso_summary_list, boot_summary_list, all_loso_records, figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
