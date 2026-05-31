"""plot_external_cohort.py — Combined publication-ready figure for external cohort audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PALETTE = {
    "clean":                    "#2C7BB6",
    "leaky_discharge_location": "#D7191C",
    "leaky_definitional":       "#D7191C",
    "oracle":                   "#1A1A1A",
}
PIPELINE_LABELS = {
    "clean":                    "Clean",
    "leaky_discharge_location": "Leaky\n(definitional\ncomponents)",
    "leaky_definitional":       "Leaky\n(definitional\ncomponents)",
    "oracle":                   "Oracle\n(algebraic rule)",
}
MODEL_ORDER  = ["LogisticRegression", "RandomForest", "SVM", "XGBoost", "OracleRule"]
MODEL_LABELS = {
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "SVM":                "SVM",
    "XGBoost":            "XGB",
    "OracleRule":         "Oracle",
}
HATCH_CYCLE = ["", "//", "xx", ".."]

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


def _filter_pipeline_order(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    present = df["pipeline"].unique().tolist()
    return [p for p in candidates if p in present]


def _draw_auroc_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    pipeline_order: list[str],
    title: str,
    show_legend: bool = True,
) -> None:
    """Draw a grouped bar chart of AUROC by pipeline on the given Axes."""
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    n_pipe  = len(pipeline_order)
    n_model = len(models)
    bar_w   = 0.75 / n_model
    x_pos   = np.arange(n_pipe)

    for mi, model in enumerate(models):
        offsets = (mi - (n_model - 1) / 2) * bar_w
        vals, errs, colours = [], [], []
        for pipe in pipeline_order:
            row = df[(df["pipeline"] == pipe) & (df["model"] == model)]
            vals.append(float(row["AUROC_mean"].iloc[0])
                        if not row.empty else 0.0)
            errs.append(float(row["AUROC_sd"].fillna(0).iloc[0])
                        if not row.empty else 0.0)
            colours.append(PALETTE.get(pipe, "#888888"))

        ax.bar(
            x_pos + offsets, vals, bar_w * 0.88,
            label=MODEL_LABELS.get(model, model),
            color=colours, alpha=0.85,
            hatch=HATCH_CYCLE[mi % len(HATCH_CYCLE)],
            yerr=errs,
            error_kw=dict(elinewidth=1.1, capsize=3),
        )

    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.55)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [PIPELINE_LABELS.get(p, p) for p in pipeline_order],
        fontsize=8.5,
    )
    ax.set_ylim(0.40, 1.08)
    ax.set_ylabel("AUROC")
    ax.set_title(title, fontsize=9.5, pad=6)

    if show_legend:
        ax.legend(title="Model", fontsize=7.5, loc="lower right", framealpha=0.7)


def _draw_delta_inset(
    ax: plt.Axes,
    df: pd.DataFrame,
    leaky_pipeline: str,
    title: str = "ΔAUC (leaky − clean)",
) -> None:
    """Bar chart of ΔAUC for each model, annotated with values."""
    leaky_rows = df[df["pipeline"] == leaky_pipeline].copy()
    models     = [m for m in MODEL_ORDER if m in leaky_rows["model"].values]

    x_b = np.arange(len(models))
    delta_vals = []
    delta_errs = []
    for model in models:
        row = leaky_rows[leaky_rows["model"] == model]
        delta_vals.append(
            float(row["DELTA_AUC"].iloc[0]) if not row.empty else 0.0
        )
        delta_errs.append(
            float(row["AUROC_sd"].fillna(0).iloc[0]) if not row.empty else 0.0
        )

    bars = ax.bar(
        x_b, delta_vals, 0.55,
        color=PALETTE.get(leaky_pipeline, "#D7191C"),
        alpha=0.85,
        yerr=delta_errs,
        error_kw=dict(elinewidth=1.2, capsize=4),
    )
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.55)
    ax.set_xticks(x_b)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=9)
    ax.set_ylabel(title)
    for bar, val in zip(bars, delta_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:+.3f}",
            ha="center", va="bottom", fontsize=8.5,
        )


def build_figure(
    df_eicu24: pd.DataFrame | None,
    df_eicu48: pd.DataFrame | None,
    df_mimic:  pd.DataFrame | None,
    figdir:    Path,
) -> None:
    """
    Produce a 2-row figure:
      Row 1: AUROC panels for each available cohort
      Row 2: ΔAUC panels for each available cohort

    Saves as external_cohort_combined.pdf + .png
    """
    datasets = []
    if df_eicu24 is not None:
        datasets.append(("eICU 24h label\n(ICU death within 24h)", df_eicu24, "leaky_definitional", "24h"))
    if df_eicu48 is not None:
        datasets.append(("eICU 48h label\n(ICU death within 48h)", df_eicu48, "leaky_definitional", "48h"))
    if df_mimic is not None:
        datasets.append(("MIMIC-IV\nIn-hospital mortality", df_mimic, "leaky_discharge_location", "mimic"))

    if not datasets:
        print("No datasets found — nothing to plot.")
        return

    n_cols  = len(datasets)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    eicu_pipe_order = [
        "clean",
        "leaky_discharge_location",
        "leaky_definitional",
        "oracle",
    ]
    mimic_pipe_order = ["clean", "leaky_discharge_location", "oracle"]

    for col, (title, df, leaky_pipe, tag) in enumerate(datasets):
        pipe_order = (
            mimic_pipe_order if tag == "mimic" else eicu_pipe_order
        )
        pipe_order = _filter_pipeline_order(df, pipe_order)

        n_total    = int(df["n_total"].iloc[0])
        n_pos      = int(df["n_positive"].iloc[0])
        prevalence = float(df["prevalence"].iloc[0]) * 100

        panel_title = (
            f"{title}\n"
            f"N={n_total:,}  |  positives={n_pos:,}  ({prevalence:.1f}%)"
        )
        _draw_auroc_panel(
            axes[0, col], df, pipe_order,
            title=panel_title,
            show_legend=(col == n_cols - 1),
        )

        leaky_available = leaky_pipe if leaky_pipe in df["pipeline"].values else None
        if leaky_available:
            _draw_delta_inset(
                axes[1, col], df,
                leaky_pipeline=leaky_available,
                title="ΔAUC (leaky − clean)",
            )
        else:
            axes[1, col].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#2C7BB6", label="Clean pipeline"),
        mpatches.Patch(color="#FDAE61", label="Leaky A: discharge location (proxy)"),
        mpatches.Patch(color="#D7191C", label="Leaky B / Definitional antecedents"),
        mpatches.Patch(color="#1A1A1A", label="Oracle (algebraic rule)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "External Real-World Leakage Audits\n"
        "Definitional Antecedents Reproduce Near-Perfect Classification in Public ICU Databases",
        fontsize=11, y=1.02,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in (".pdf", ".png"):
        out = figdir / f"external_cohort_combined{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")

    plt.close(fig)


def build_prauc_supplement(
    df_eicu24: pd.DataFrame | None,
    df_mimic:  pd.DataFrame | None,
    figdir: Path,
) -> None:
    """PR-AUC comparison figure (supplementary) — important for imbalanced labels."""
    datasets = []
    if df_eicu24 is not None:
        datasets.append(("eICU 24h (prevalence 2.8%)", df_eicu24))
    if df_mimic is not None:
        datasets.append(("MIMIC mortality (prevalence 2.5%)", df_mimic))

    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    pipe_colours = {
        "clean":                    "#2C7BB6",
        "leaky_discharge_location": "#D7191C",
        "leaky_definitional":       "#D7191C",
        "oracle":                   "#1A1A1A",
    }

    for ax, (title, df) in zip(axes, datasets):
        for _, row in df.iterrows():
            pipe = row.get("pipeline", "")
            mod  = row.get("model", "")
            if mod == "OracleRule":
                continue
            auroc = float(row.get("AUROC_mean", 0))
            prauc = float(row.get("PR_AUC_mean", 0))
            if np.isnan(prauc):
                continue
            colour = pipe_colours.get(pipe, "#888888")
            ax.scatter(auroc, prauc, c=colour, s=60, alpha=0.85,
                       label=f"{pipe}" if mod == "LogisticRegression" else "_nolegend_")
            ax.annotate(
                MODEL_LABELS.get(mod, mod),
                (auroc, prauc),
                textcoords="offset points", xytext=(4, 2),
                fontsize=7.5, color=colour,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("AUROC")
        ax.set_ylabel("PR-AUC")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5)

    fig.suptitle("AUROC vs PR-AUC: External Cohorts", fontsize=10)
    plt.tight_layout()

    for ext in (".pdf", ".png"):
        out = figdir / f"external_cohort_prauc{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined figure for external cohort leakage audits."
    )
    parser.add_argument(
        "--eicu24",
        default="results/tables/external_eicu_24h_results.csv",
        help="eICU 24h results CSV",
    )
    parser.add_argument(
        "--eicu48",
        default="results/tables/external_eicu_48h_results.csv",
        help="eICU 48h results CSV (optional)",
    )
    parser.add_argument(
        "--mimic",
        default="results/tables/external_mimic_results.csv",
        help="MIMIC results CSV",
    )
    parser.add_argument(
        "--figdir",
        default="results/figures",
        help="Output figure directory",
    )
    args = parser.parse_args()

    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    def _load(path: str) -> pd.DataFrame | None:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] not found: {path}")
            return None
        df = pd.read_csv(p)
        print(f"  Loaded {len(df)} rows from {path}")
        return df

    print("Loading result CSVs …")
    df24   = _load(args.eicu24)
    df48   = _load(args.eicu48)
    dfmimic = _load(args.mimic)

    print("\nBuilding combined figure …")
    build_figure(df24, df48, dfmimic, figdir)

    print("Building PR-AUC supplement …")
    build_prauc_supplement(df24, dfmimic, figdir)

    print("Done.")


if __name__ == "__main__":
    main()
