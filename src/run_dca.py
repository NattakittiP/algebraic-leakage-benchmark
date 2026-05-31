"""Decision Curve Analysis (DCA) for clean vs leaky eICU pipelines."""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import FoldSealedScaler, FoldSealedWinsorizer

NUMERIC_CLEAN = [
    "age_num",
    "lab_-basos", "lab_-eos", "lab_-lymphs", "lab_-monos", "lab_-polys",
    "lab_ALT_(SGPT)", "lab_AST_(SGOT)", "lab_BUN", "lab_Hct", "lab_Hgb",
    "lab_MCH", "lab_MCHC", "lab_MCV", "lab_MPV", "lab_RBC", "lab_RDW",
    "lab_WBC_x_1000", "lab_albumin", "lab_anion_gap", "lab_bedside_glucose",
    "lab_bicarbonate", "lab_calcium", "lab_chloride", "lab_creatinine",
    "lab_glucose", "lab_magnesium", "lab_platelets_x_1000", "lab_potassium",
    "lab_sodium", "lab_total_protein",
    "vital_cvp", "vital_heartrate", "vital_respiration", "vital_sao2",
    "vital_st1", "vital_st2", "vital_st3",
    "vital_systemicdiastolic", "vital_systemicmean", "vital_systemicsystolic",
]

CATEGORICAL_CLEAN = [
    "gender", "ethnicity", "unittype", "unitstaytype",
    "hospitaladmitsource", "unitadmitsource",
]

# Algebraic antecedents of the label formula
NUMERIC_LEAKY_EXTRA    = ["unitdischargeoffset_num"]
CATEGORICAL_LEAKY_EXTRA = ["hospitaldischargestatus"]


class FoldSealedPreprocessorICU:
    def __init__(self):
        self._num_imputer = SimpleImputer(strategy="median")
        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        self._encoder     = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)
        self._winsorizer  = FoldSealedWinsorizer(lower_pct=1.0, upper_pct=99.0)
        self._scaler      = FoldSealedScaler()
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._fitted = False

    def fit_transform(self, df_train, num_cols, cat_cols):
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

    def transform(self, df_test):
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


def get_oof_probabilities(
    df: pd.DataFrame,
    label_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    clf_name: str,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Collect out-of-fold predicted probabilities using stratified k-fold CV with fold-sealed preprocessing."""
    y_all = df[label_col].values.astype(int)
    cv    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_probs = np.full(len(df), np.nan)

    if clf_name == "LR":
        base_clf = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=seed)
    elif clf_name == "RF":
        base_clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    print(f"    Collecting OOF probs [{clf_name}] …", flush=True)
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(df)), y_all), 1
    ):
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_te = df.iloc[test_idx].reset_index(drop=True)
        y_tr  = y_all[train_idx]

        prep = FoldSealedPreprocessorICU()
        X_tr = prep.fit_transform(df_tr, num_cols, cat_cols)
        X_te = prep.transform(df_te)

        clf = sklearn.base.clone(base_clf)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        oof_probs[test_idx] = probs

        print(f"      fold {fold_idx}/{n_folds} done", flush=True)

    return oof_probs


def net_benefit(y_true: np.ndarray, probs: np.ndarray,
                thresholds: np.ndarray) -> np.ndarray:
    """Compute net benefit at each threshold probability: NB(pt) = TP/N - FP/N * pt/(1-pt)."""
    N = len(y_true)
    nb = np.zeros(len(thresholds))
    for i, pt in enumerate(thresholds):
        predicted_pos = probs >= pt
        tp = np.sum((predicted_pos == 1) & (y_true == 1))
        fp = np.sum((predicted_pos == 1) & (y_true == 0))
        nb[i] = tp / N - fp / N * (pt / (1.0 - pt))
    return nb


def treat_all_nb(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit of 'treat all' strategy."""
    N   = len(y_true)
    pos = y_true.sum()
    nb  = np.zeros(len(thresholds))
    for i, pt in enumerate(thresholds):
        tp = pos
        fp = N - pos
        nb[i] = tp / N - fp / N * (pt / (1.0 - pt))
    return nb


def plot_dca(df_nb: pd.DataFrame, prevalence: float, figdir: Path) -> None:
    """Plot DCA curves: panel A full range 0.01–0.50, panel B clinical zoom 0.01–0.10."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi":      150,
    })

    COLORS = {
        "Treat All":            "#888888",
        "Treat None":           "#cccccc",
        "Clean LR":             "#2166AC",
        "Clean RF":             "#4DAC26",
        "Leaky LR":             "#D7191C",
        "Leaky RF":             "#E66101",
    }
    STYLES = {
        "Treat All":  dict(ls="--",  lw=1.5, alpha=0.8),
        "Treat None": dict(ls=":",   lw=1.2, alpha=0.6),
        "Clean LR":   dict(ls="-",   lw=2.0),
        "Clean RF":   dict(ls="-.",  lw=2.0),
        "Leaky LR":   dict(ls="-",   lw=2.0),
        "Leaky RF":   dict(ls="-.",  lw=2.0),
    }

    thresholds = df_nb["threshold"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (ax, xlim) in enumerate(zip(axes, [(0.01, 0.50), (0.01, 0.10)])):
        mask = (thresholds >= xlim[0]) & (thresholds <= xlim[1])

        for col in df_nb.columns:
            if col == "threshold":
                continue
            nb_vals = df_nb.loc[mask, col].values
            thrs    = thresholds[mask]
            kw = dict(**STYLES.get(col, {}))
            ax.plot(thrs, nb_vals,
                    label=col,
                    color=COLORS.get(col, "#333333"),
                    **kw)

        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.axvline(prevalence, color="gray", lw=1.0, ls=":", alpha=0.7,
                   label=f"Prevalence ({prevalence*100:.1f}%)")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit" if ax_idx == 0 else "")
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=-0.005)

        title_suffix = "Full range" if ax_idx == 0 else "Clinical zoom (0–10%)"
        ax.set_title(f"DCA — eICU 24h mortality\n{title_suffix}", fontsize=10)

        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Decision Curve Analysis: Clean vs Leaky Pipeline\n"
        "Leaky pipeline shows spuriously elevated net benefit "
        "driven by algebraic identity, not clinical signal",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    for ext, dpi in [("pdf", None), ("png", 300)]:
        out = figdir / f"dca_comparison.{ext}"
        kw  = {"bbox_inches": "tight"}
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(out, **kw)
        print(f"  Figure saved → {out}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="DCA: clean vs leaky pipeline on eICU 24h mortality label."
    )
    parser.add_argument("--data24",
                        default="External Cohort/Dataset/eicu_label24h.csv")
    parser.add_argument("--outdir", default="results/tables")
    parser.add_argument("--figdir", default="results/figures")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("Loading eICU 24h dataset …")
    df = pd.read_csv(
        args.data24,
        na_values=["", "nan", "NaN", "NA", "N/A", "Unknown", "Other"],
        keep_default_na=True,
    )
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    label_col    = "label24h"
    y_all        = df[label_col].values.astype(int)
    prevalence   = float(y_all.mean())
    N            = len(df)

    print(f"  N = {N:,}  |  positives = {y_all.sum():,} ({prevalence*100:.2f}%)")

    results: dict[str, np.ndarray] = {}

    for pipeline_name, num_extra, cat_extra in [
        ("Clean",  [],                         []),
        ("Leaky",  NUMERIC_LEAKY_EXTRA,         CATEGORICAL_LEAKY_EXTRA),
    ]:
        num_cols = [c for c in NUMERIC_CLEAN + num_extra if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_CLEAN + cat_extra if c in df.columns]

        print(f"\n[{pipeline_name}] n_features = {len(num_cols)+len(cat_cols)}")

        for clf_name in ["LR", "RF"]:
            key = f"{pipeline_name} {clf_name}"
            print(f"\nPipeline: {key}")
            probs = get_oof_probabilities(
                df, label_col, num_cols, cat_cols,
                clf_name, n_folds=args.n_folds, seed=args.seed,
            )
            results[key] = probs

    thresholds = np.linspace(0.01, 0.50, 490)

    print("\nComputing DCA curves …")
    rows = {"threshold": thresholds}
    rows["Treat All"]  = treat_all_nb(y_all, thresholds)
    rows["Treat None"] = np.zeros(len(thresholds))

    for key, probs in results.items():
        valid = ~np.isnan(probs)
        nb = net_benefit(y_all[valid], probs[valid], thresholds)
        rows[key] = nb

    df_nb = pd.DataFrame(rows)

    out_csv = outdir / "dca_results.csv"
    df_nb.to_csv(out_csv, index=False)
    print(f"\nDCA results → {out_csv}")

    print("\n=== DCA summary at key thresholds ===")
    key_thresholds = [0.02, 0.03, 0.05, 0.10, 0.20]
    print(f"{'Threshold':>12s}", end="")
    for col in df_nb.columns[1:]:
        print(f"  {col:>12s}", end="")
    print()
    for pt in key_thresholds:
        idx = np.argmin(np.abs(thresholds - pt))
        print(f"{pt:>12.2f}", end="")
        for col in df_nb.columns[1:]:
            print(f"  {df_nb[col].iloc[idx]:>12.5f}", end="")
        print()

    print("\nGenerating figure …")
    try:
        plot_dca(df_nb, prevalence, figdir)
    except Exception as e:
        print(f"  Warning: plotting failed ({e}). CSV still saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()
