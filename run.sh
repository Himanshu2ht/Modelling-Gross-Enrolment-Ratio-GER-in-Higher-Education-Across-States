#!/usr/bin/env bash
# Convenience runner for Linux / macOS.
# Usage:
#   ./run.sh                     # full pipeline (downloads official sources)
#   ./run.sh --skip-download     # use existing data/raw files
#   ./run.sh --skip-download --skip-extract
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
    echo "Creating virtual environment in .venv ..."
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import pandas, numpy, sklearn, statsmodels, scipy, matplotlib, seaborn, pypdf, pdfplumber, openpyxl" >/dev/null 2>&1; then
    echo "Installing dependencies from requirements.txt ..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

exec python src/run_all.py "$@"
