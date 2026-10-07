#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -f "models/$f" ] || curl -fL -o "models/$f" "$base/$f"
done
python -m vo.agent download-files
