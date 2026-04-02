#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HOST_PYTHON="${PYTHON_BIN:-}"
if [[ -z "${HOST_PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    HOST_PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    HOST_PYTHON="python"
  else
    echo "No python interpreter found. Install python3 and retry."
    exit 1
  fi
fi

VENV_DIR="${ROOT_DIR}/.wasm-build-venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating build virtual environment at ${VENV_DIR}..."
  "${HOST_PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

echo "Building local simdec wheel..."
"${PYTHON_BIN}" -m pip install --upgrade pip build panel matplotlib seaborn scipy SALib
"${PYTHON_BIN}" -m build --wheel .

SIMDEC_WHEEL="$(ls dist/simdec-*.whl | head -n 1)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "Converting Panel apps to Pyodide worker output..."
"${PYTHON_BIN}" -m panel convert panel/simdec_app.py panel/sampling.py \
  --to pyodide-worker \
  --out dist/pyodide \
  --requirements "${SIMDEC_WHEEL}" numpy pandas matplotlib seaborn scipy SALib \
  --resources panel/data/stress.csv

echo "Copying custom index page and static assets..."
# 1. Create the _static folder in the output directory
mkdir -p dist/pyodide/_static

# 2. Copy all your images and thumbnails from docs/_static (if they exist)
if [ -d "docs/_static" ]; then
  cp -r docs/_static/* dist/pyodide/_static/
fi

# 3. Copy your beautiful custom HTML file to act as the homepage OVERWRITING anything else
cp panel/index.html dist/pyodide/index.html

echo "WASM site generated at dist/pyodide"
