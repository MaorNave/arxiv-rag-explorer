#!/usr/bin/env bash
# Local, free image generation with FLUX.2 [klein] through Ollama (macOS, Apple Silicon).
#
# Ollama removed its experimental image generation in 0.32.6, and its release notes say to keep
# using 0.32.5 for it. This script runs Ollama 0.32.5 as a SECOND server, dedicated to images.
# Your regular Ollama (the one serving Qwen) is not touched, downgraded or slowed down.
#
#   ./scripts/ollama_imagegen.sh          # first run: install + start; later runs: start
#   ./scripts/ollama_imagegen.sh status
#   ./scripts/ollama_imagegen.sh stop
#   ./scripts/ollama_imagegen.sh uninstall   # removes ~/.ollama-imagegen (your models are kept)
#
# The app detects the server automatically (no .env change needed) and uses it first for
# illustrations, once the answer is complete. Measured on an M1 Pro with 16 GB: about 38 s per
# 1024x576 image with a peak of ~7.6 GB. The first run downloads Ollama 0.32.5 (146 MB) and reuses
# the x/flux2-klein:4b files of your regular Ollama when present (copy-on-write clones, no extra
# disk space); otherwise it downloads the model (5.7 GB).
set -euo pipefail

VERSION="${OLLAMA_IMAGEGEN_VERSION:-0.32.5}"
HOME_DIR="${OLLAMA_IMAGEGEN_HOME:-$HOME/.ollama-imagegen}"
PORT="${OLLAMA_IMAGEGEN_PORT:-11435}"
MODEL="${OLLAMA_IMAGE_MODEL:-x/flux2-klein:4b}"
MAIN_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

BIN="$HOME_DIR/bin/ollama"
MODELS="$HOME_DIR/models"
PIDFILE="$HOME_DIR/serve.pid"
LOG="$HOME_DIR/serve.log"
URL="http://127.0.0.1:$PORT"
NAME="${MODEL%%:*}"
TAG="${MODEL##*:}"
[[ "$MODEL" == *:* ]] || TAG="latest"
MANIFEST="manifests/registry.ollama.ai/$NAME/$TAG"

is_up() { curl -fsS --max-time 2 "$URL/api/version" >/dev/null 2>&1; }

stop_server() {
  if [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Stopped the image server on port $PORT"
  elif is_up; then
    kill "$(lsof -tiTCP:"$PORT" -sTCP:LISTEN)" && echo "Stopped the image server on port $PORT"
  else
    echo "The image server is not running"
  fi
  rm -f "$PIDFILE"
}

case "${1:-start}" in
  start) ;;
  stop) stop_server; exit 0 ;;
  status)
    if is_up; then echo "Running at $URL (Ollama $(curl -s "$URL/api/version" | sed 's/.*"version":"\([^"]*\)".*/\1/'))"
    else echo "Not running"; fi
    exit 0 ;;
  uninstall) stop_server; rm -rf "$HOME_DIR"; echo "Removed $HOME_DIR"; exit 0 ;;
  *) echo "usage: $0 [start|status|stop|uninstall]" >&2; exit 1 ;;
esac

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper supports macOS only. On other systems the app uses the free image APIs" >&2
  echo "(see README: Bonus: image generation)." >&2
  exit 1
fi
if is_up; then
  echo "The image server is already running at $URL"
  exit 0
fi

# 1. Ollama 0.32.5, installed next to (not over) your regular Ollama
if [[ ! -x "$BIN" ]]; then
  echo "==> Downloading Ollama $VERSION into $HOME_DIR (146 MB)"
  mkdir -p "$HOME_DIR/bin"
  curl -fL --progress-bar "https://github.com/ollama/ollama/releases/download/v$VERSION/ollama-darwin.tgz" \
    | tar -xz -C "$HOME_DIR/bin"
fi

# 2. Reuse the image model from your regular Ollama when it is there (APFS clones: no extra disk)
if [[ ! -f "$MODELS/$MANIFEST" && -f "$MAIN_MODELS/$MANIFEST" ]] && command -v python3 >/dev/null 2>&1; then
  echo "==> Reusing $MODEL from $MAIN_MODELS (copy-on-write clones)"
  python3 - "$MAIN_MODELS" "$MODELS" "$MANIFEST" <<'PY'
import json, os, shutil, subprocess, sys

src, dst, manifest = sys.argv[1:4]
data = json.load(open(os.path.join(src, manifest)))
digests = [data["config"]["digest"]] + [layer["digest"] for layer in data["layers"]]
os.makedirs(os.path.join(dst, "blobs"), exist_ok=True)
for digest in dict.fromkeys(digests):
    name = digest.replace(":", "-")
    target = os.path.join(dst, "blobs", name)
    if not os.path.exists(target):
        source = os.path.join(src, "blobs", name)
        if subprocess.run(["cp", "-c", source, target]).returncode != 0:
            shutil.copy2(source, target)  # not APFS: plain copy
os.makedirs(os.path.dirname(os.path.join(dst, manifest)), exist_ok=True)
shutil.copy2(os.path.join(src, manifest), os.path.join(dst, manifest))
print(f"    {len(set(digests))} files")
PY
fi

# 3. Start the dedicated server (its own model folder; pruning off so nothing is ever deleted)
mkdir -p "$MODELS"
echo "==> Starting Ollama $VERSION for images at $URL (log: $LOG)"
OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_MODELS="$MODELS" OLLAMA_NOPRUNE=1 \
  nohup "$BIN" serve >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
for _ in $(seq 1 30); do is_up && break; sleep 1; done
is_up || { echo "The server did not start; see $LOG" >&2; exit 1; }

# 4. Download the model only if it could not be reused
if [[ ! -f "$MODELS/$MANIFEST" ]]; then
  echo "==> Downloading $MODEL (5.7 GB)"
  OLLAMA_HOST="127.0.0.1:$PORT" "$BIN" pull "$MODEL"
fi

echo
echo "Ready: local FLUX.2 klein images at $URL."
echo "The app picks it up automatically within 30 s (tick 'Generate image' in the UI)."
echo "Stop it with: ./scripts/ollama_imagegen.sh stop"
