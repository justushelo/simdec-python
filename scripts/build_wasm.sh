#!/usr/bin/env bash
set -euo pipefail

# 1. Setup paths and environment
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Determine which Python interpreter to use
HOST_PYTHON="${PYTHON_BIN:-}"
if [[ -z "${HOST_PYTHON}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        HOST_PYTHON="python3"
    elif command -v python >/dev/null 2>&1; then
        HOST_PYTHON="python"
    else
        echo "Error: No python interpreter found. Install python3 and retry."
        exit 1
    fi
fi

# Create build virtual environment
VENV_DIR="${ROOT_DIR}/.wasm-build-venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating build virtual environment at ${VENV_DIR}..."
    "${HOST_PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

# Install build dependencies and build the wheel
echo "Installing build tools and generating local wheel..."
"${PYTHON_BIN}" -m pip install --upgrade pip build panel matplotlib seaborn scipy SALib

# Clean old builds to avoid picking up the wrong wheel
rm -rf dist/*.whl
"${PYTHON_BIN}" -m build --wheel .

# Identify the generated wheel file
# This prevents the "unbound variable" error by checking if the file exists
SIMDEC_WHEEL_PATH=$(ls dist/*.whl | head -n 1 || echo "")

if [[ -z "${SIMDEC_WHEEL_PATH}" ]]; then
    echo "Error: No wheel file found in dist/. Build failed."
    exit 1
fi

WHEEL_FILENAME=$(basename "${SIMDEC_WHEEL_PATH}")

# Prepare output directory
OUT_DIR="dist/pyodide"
mkdir -p "${OUT_DIR}/_static"

# IMPORTANT: Copy the wheel into the output directory so it's accessible via HTTP
cp "${SIMDEC_WHEEL_PATH}" "${OUT_DIR}/"

# Convert Panel apps to Pyodide worker
echo "Converting Panel apps to Pyodide worker output..."
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# Use the full path for the requirements so the converter can find the file
"${PYTHON_BIN}" -m panel convert panel/simdec_app.py panel/sampling.py \
    --to pyodide-worker \
    --out "${OUT_DIR}" \
    --requirements "${SIMDEC_WHEEL_PATH}" numpy pandas matplotlib seaborn scipy SALib \
    --resources panel/data/stress.csv

# Copy custom index page and static assets
echo "Copying custom index page and static assets..."

# Copy images/thumbnails from docs/_static if they exist
if [ -d "docs/_static" ]; then
    cp -r docs/_static/* "${OUT_DIR}/_static/"
fi

# Overwrite default index.html with your custom homepage
if [ -f "panel/index.html" ]; then
    cp panel/index.html "${OUT_DIR}/index.html"
else
    echo "Warning: panel/index.html not found. Using default Panel index."
fi

echo "---"
echo "WASM site successfully generated at ${OUT_DIR}"
