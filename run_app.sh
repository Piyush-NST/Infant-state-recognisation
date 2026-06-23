#!/bin/zsh
set -euo pipefail

ROOT_DIR="/Users/piyush/Infant-state-recognisation"
APP_DIR="$ROOT_DIR/infant-state-recognition"
TF_PYTHON="$ROOT_DIR/.venv-tf/bin/python"
FALLBACK_PYTHON="$ROOT_DIR/.venv/bin/python"

cd "$APP_DIR"

if [[ -x "$TF_PYTHON" ]]; then
  echo "Starting app with TensorFlow environment (.venv-tf)..."
  exec "$TF_PYTHON" app/app.py
elif [[ -x "$FALLBACK_PYTHON" ]]; then
  echo "Starting app with fallback environment (.venv)..."
  exec "$FALLBACK_PYTHON" app/app.py
else
  echo "No virtual environment found. Please create one first."
  exit 1
fi
