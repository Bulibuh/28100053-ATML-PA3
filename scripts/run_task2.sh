#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${COCO_ANNOTATIONS:?Set COCO_ANNOTATIONS to the folder with the ten annotation JSON files}"
# Task 1 prepare resolves the immutable starting Qwen checkpoint, shared by Task 2.
[[ -f data/prepared/model.json ]] || pa3 task1 prepare
pa3 task2 prepare --annotations "$COCO_ANNOTATIONS"
pa3 task2 cache --download
pa3 task2 train --visual-tokens 196
pa3 task2 evaluate --visual-tokens 196
pa3 evidence
