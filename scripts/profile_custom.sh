#!/usr/bin/env bash
set -euo pipefail
CHAIR_PROMPT='Look at the chairs in this image.
Are all visible chairs pushed in and neatly tucked under the table?
Return exactly one valid JSON object and no markdown or explanation. The first character must be "{" and the last character must be "}".
Schema: {"answer":"yes"} or {"answer":"no"}'
args=(
  custom
  --model "${MODEL:-Qwen/Qwen3.5-0.8B}"
  --prompt "${PROMPT:-$CHAIR_PROMPT}"
  --image-size "${IMAGE_SIZE:-256}"
)
if [[ -n "${IMAGE:-}" ]]; then
  args+=(--image "$IMAGE")
fi
qaihub-qwen-profile "${args[@]}"
