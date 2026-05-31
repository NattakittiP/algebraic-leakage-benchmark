"""run_external_cohort.py — Runner for both external cohort audits (eICU + MIMIC)."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

EICU_DATA_24 = ROOT / "External Cohort" / "Dataset" / "eicu_label24h.csv"
EICU_DATA_48 = ROOT / "External Cohort" / "Dataset" / "eicu_label48h.csv"
MIMIC_DATA   = ROOT / "External Cohort" / "Dataset" / "full_analytic_dataset_mortality_all_admissions.csv"

OUTDIR  = ROOT / "results" / "tables"
FIGDIR  = ROOT / "results" / "figures"
SRC     = ROOT / "src"


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 72)
    print("  $ " + " ".join(str(c) for c in cmd))
    print("=" * 72 + "\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Command exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",       action="store_true", help="5 seeds (smoke test)")
    parser.add_argument("--skip_eicu48", action="store_true", help="Skip eICU 48h label")
    parser.add_argument("--eicu_only",   action="store_true", help="Run eICU only")
    parser.add_argument("--mimic_only",  action="store_true", help="Run MIMIC only")
    parser.add_argument("--n_seeds",     type=int, default=None)
    parser.add_argument("--n_folds",     type=int, default=5)
    args = parser.parse_args()

    n_seeds = args.n_seeds if args.n_seeds else (5 if args.quick else 30)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    if not args.mimic_only:
        cmd_eicu = [
            sys.executable,
            str(SRC / "run_external_eicu.py"),
            "--data24", str(EICU_DATA_24),
            "--data48", str(EICU_DATA_48),
            "--outdir",  str(OUTDIR),
            "--figdir",  str(FIGDIR),
            "--n_seeds", str(n_seeds),
            "--n_folds", str(args.n_folds),
        ]
        if args.skip_eicu48 or args.quick:
            cmd_eicu.append("--skip48")
        run(cmd_eicu)

    if not args.eicu_only:
        cmd_mimic = [
            sys.executable,
            str(SRC / "run_external_mimic.py"),
            "--data",    str(MIMIC_DATA),
            "--outdir",  str(OUTDIR),
            "--figdir",  str(FIGDIR),
            "--n_seeds", str(n_seeds),
            "--n_folds", str(args.n_folds),
        ]
        run(cmd_mimic)

    cmd_plot = [
        sys.executable,
        str(SRC / "plot_external_cohort.py"),
        "--eicu24", str(OUTDIR / "external_eicu_24h_results.csv"),
        "--eicu48", str(OUTDIR / "external_eicu_48h_results.csv"),
        "--mimic",  str(OUTDIR / "external_mimic_results.csv"),
        "--figdir", str(FIGDIR),
    ]
    run(cmd_plot)

    print("\n✓  All done.  Results in:")
    print(f"     Tables  → {OUTDIR}")
    print(f"     Figures → {FIGDIR}")


if __name__ == "__main__":
    main()
