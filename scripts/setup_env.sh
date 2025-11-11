#!/usr/bin/env bash
set -euo pipefail

# Configuration
VENV_NAME="venv_Tg_predictor"
PYTHON_BIN="python3"  # Assumes Python 3.12 is the default python3
KERNEL_NAME="venv_Tg_predictor"
KERNEL_DISPLAY_NAME="Python (venv_Tg_predictor)"

echo "[1/6] Creating virtual environment: $VENV_NAME"
if [ -d "$VENV_NAME" ]; then
  echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
else
  $PYTHON_BIN -m venv "$VENV_NAME"
  echo "Created virtual environment with $("$PYTHON_BIN" --version)"
fi

echo "[2/6] Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_NAME/bin/activate"

echo "[3/6] Upgrading core packaging tools"
pip install setuptools wheel

echo "[4/6] Installing pip-tools"
pip install --upgrade pip-tools

if [ -f requirements.in ]; then
  echo "[5/6] Compiling requirements.in -> requirements.txt"
  pip-compile requirements.in --resolver=backtracking
else
  echo "requirements.in not found. Exiting." >&2
  exit 1
fi

echo "[6/6] Installing dependencies from requirements.txt"
pip install -r requirements.txt

echo "Registering Jupyter kernel '$KERNEL_NAME'"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo "Environment setup complete. Activate later with: source $VENV_NAME/bin/activate"