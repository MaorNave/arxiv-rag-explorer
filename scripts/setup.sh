#!/usr/bin/env bash
# One-shot setup for macOS and Linux.
#
#   ./scripts/setup.sh          # interactive
#   ./scripts/setup.sh --yes    # install Ollama without asking (if it is missing)
#
# 1. checks Python >= 3.10, creates .venv and installs requirements.txt
# 2. makes sure Ollama is installed and running (offers to install it)
# 3. downloads the Qwen chat model, the embedding model and the reranker
set -euo pipefail
cd "$(dirname "$0")/.."

ASSUME_YES=false
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && ASSUME_YES=true

confirm() {
  $ASSUME_YES && return 0
  read -r -p "$1 [y/N] " answer
  [[ "$answer" =~ ^[Yy] ]]
}

# --- 1. Python environment ----------------------------------------------------
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.10+ is required (https://www.python.org/downloads/)." >&2
  exit 1
fi
"$PYTHON" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python 3.10+ is required, found {sys.version.split()[0]}")
EOF

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment (.venv)"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> Installing Python dependencies"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

# --- 2. Ollama --------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Ollama is not installed (it serves the local Qwen model)."
  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1 && confirm "Install it with Homebrew (brew install ollama)?"; then
        brew install ollama
      else
        echo "Download the macOS app from https://ollama.com/download, start it, then re-run this script."
        exit 1
      fi
      ;;
    Linux)
      if confirm "Install it with the official script (curl -fsSL https://ollama.com/install.sh | sh)?"; then
        curl -fsSL https://ollama.com/install.sh | sh
      else
        echo "See https://ollama.com/download/linux, then re-run this script."
        exit 1
      fi
      ;;
    *)
      echo "Install Ollama from https://ollama.com/download, then re-run this script."
      exit 1
      ;;
  esac
fi

OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
if ! curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
  echo "==> Starting Ollama in the background (log: ${TMPDIR:-/tmp}/ollama.log)"
  nohup ollama serve >"${TMPDIR:-/tmp}/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
fi

# --- 3. Models ------------------------------------------------------------------
echo "==> Downloading models (first run: ~3.5 GB for Qwen 3.5 4B + embeddings + reranker)"
python -m rag_app.setup_models

echo
echo "All set! Start the app with:"
echo "    source .venv/bin/activate && python main.py"
echo "then open http://127.0.0.1:8080"
