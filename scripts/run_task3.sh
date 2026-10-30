#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MATH_ROLLOUTS:?Set MATH_ROLLOUTS to the supplied LLM folder}"
: "${MATHVISTA_ROLLOUTS:?Set MATHVISTA_ROLLOUTS to the supplied VLM folder}"
: "${MATHVISTA_METADATA:?Set MATHVISTA_METADATA to the released MathVista question JSON}"
: "${ROLLOUT_README:?Set ROLLOUT_README to the original supplied README.MD}"
: "${LIVE_REVISION:?Set LIVE_REVISION to the exact Qwen2.5-3B SHA in that README}"
pa3 task3 offline --math-dir "$MATH_ROLLOUTS" --mathvista-dir "$MATHVISTA_ROLLOUTS" \
  --metadata "$MATHVISTA_METADATA" --readme "$ROLLOUT_README" --schema configs/rollout_schema.json
pa3 task3 live --readme "$ROLLOUT_README" --revision "$LIVE_REVISION"
pa3 evidence
