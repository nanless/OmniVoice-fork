#!/bin/bash
# Download voxblink2_samresnet100_ft weights and symlink to model/avg_model.pt
#
# Official source (WeNet):
#   https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet100_ft.zip
#
# GitHub Release 亦指向上述链接，权重不入库（>100MB GitHub 限制）。
set -euo pipefail

cd "$(dirname "$0")"
MODEL_DIR="$(pwd)/model"
CACHE_DIR="$MODEL_DIR/.cache/voxblink2_samresnet100_ft"
LINK="$MODEL_DIR/avg_model.pt"
WENET_URL="https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet100_ft.zip"
ZIP_NAME="voxblink2_samresnet100_ft.zip"

find_weights() {
  find "$CACHE_DIR" -name avg_model.pt -type f 2>/dev/null | head -1
}

ensure_symlink() {
  local weights="$1"
  ln -sfn "$weights" "$LINK"
  echo "Linked: $LINK -> $weights"
  ls -lh "$LINK"
}

if [[ -f "$LINK" && ! -L "$LINK" ]]; then
  echo "Removing existing regular file: $LINK"
  rm -f "$LINK"
fi

if existing="$(find_weights)"; then
  ensure_symlink "$existing"
  exit 0
fi

mkdir -p "$CACHE_DIR"
ZIP="${1:-$CACHE_DIR/$ZIP_NAME}"

if [[ ! -f "$ZIP" ]]; then
  echo "Weights not found. Download from WeNet (browser):"
  echo "  $WENET_URL"
  echo ""
  echo "Save the zip, then re-run:"
  echo "  bash download_model.sh /path/to/$ZIP_NAME"
  exit 1
fi

echo "Extracting $ZIP -> $CACHE_DIR"
unzip -oq "$ZIP" -d "$CACHE_DIR"

weights="$(find_weights)"
if [[ -z "$weights" ]]; then
  echo "ERROR: avg_model.pt not found after unzip. Check zip contents." >&2
  exit 1
fi

ensure_symlink "$weights"
