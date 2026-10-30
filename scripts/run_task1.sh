#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${QUALITATIVE_PROMPTS:?Set QUALITATIVE_PROMPTS to the supplied, schema-normalized first_plot_questions.yaml}"
pa3 task1 prepare
pa3 task1 evaluate --condition baseline
for method in full l2sp lora4 lora16 oplora16; do
  pa3 task1 train --method "$method"
  if [[ "$method" == full ]]; then pa3 task1 evaluate --condition full_mid; fi
  pa3 task1 evaluate --condition "$method"
done
pa3 task1 diagnostics
pa3 task1 qualitative --prompts "$QUALITATIVE_PROMPTS"
pa3 evidence
