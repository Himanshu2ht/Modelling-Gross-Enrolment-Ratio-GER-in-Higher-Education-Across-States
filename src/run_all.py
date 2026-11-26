r"""
Run the complete pipeline in order.

Usage
-----
    .\.py\python.exe src\run_all.py
    .\.py\python.exe src\run_all.py --skip-download
    .\.py\python.exe src\run_all.py --skip-download --skip-extract

The default re-downloads the two official sources (requires internet), rebuilds
the analysis files, re-estimates all models, and regenerates every table and
figure.  Use the skip flags when the raw files are already present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("download_data.py", "Download and checksum the official sources"),
    ("extract_aishe.py", "Extract AISHE tables and validate against the PDF"),
    ("build_dataset.py", "Merge the OGD target with AISHE predictors"),
    ("eda.py", "Descriptive tables and figures 01-04"),
    ("checks.py", "Log-transform comparison (Table 12) and missingness (Table 13)"),
    ("modeling.py", "Model comparison, inference, diagnostics, figures 05-11"),
    ("projection.py", "2035 scenarios, required density, figures 12-15"),
]


def run(script: str) -> None:
    print(f"\n{'=' * 72}\nRunning {script}\n{'=' * 72}", flush=True)
    subprocess.run([PY, str(ROOT / "src" / script)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true",
                        help="use the existing files in data/raw")
    parser.add_argument("--skip-extract", action="store_true",
                        help="use the existing files in data/processed")
    args = parser.parse_args()

    for script, description in STEPS:
        if script == "download_data.py" and args.skip_download:
            print(f"Skipping {script} ({description})")
            continue
        if script == "extract_aishe.py" and args.skip_extract:
            print(f"Skipping {script} ({description})")
            continue
        run(script)

    print("\nPipeline complete. See outputs/ and report/.")


if __name__ == "__main__":
    main()
