#!/usr/bin/env bash
# Run the demo against a live gateway. Configure credentials first (see ../.env.example).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"

exec "${root}/bin/llm-extract" \
  --input "$here" \
  --output "${here}/out" \
  --ocr always \
  --format both \
  "$@"
