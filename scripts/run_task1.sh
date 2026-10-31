#!/usr/bin/env bash
# Run from the repository root using the isolated environment's Python.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PA3_PYTHON:-$PWD/.venv-colab/bin/python}"
ARTIFACTS="${PA3_ARTIFACT_ROOT:-$PWD}"
"$PYTHON_BIN" -m atml_pa3.cli doctor --require-cuda
"$PYTHON_BIN" -m atml_pa3.cli task1 prepare
condition() {
    local name="$1"
    if [[ ! -f "$ARTIFACTS/evidence/task1/evaluation/$name/scores.json" ]]; then
        "$PYTHON_BIN" -m atml_pa3.cli task1 evaluate --condition "$name" --batch-size "${PA3_EVAL_BATCH_SIZE:-2}"
    fi
    if [[ ! -f "$ARTIFACTS/evidence/task1/qualitative/$name.json" ]]; then
        "$PYTHON_BIN" -m atml_pa3.cli task1 qualitative --condition "$name"
    fi
}
condition baseline
for method in full l2sp lora4 lora16 oplora16; do
    final="$ARTIFACTS/checkpoints/task1/$method/update_1250.pt"
    if [[ ! -f "$ARTIFACTS/evidence/task1/$method/resources.json" ]]; then
        latest=""
        for update in 312 625 937 1250; do
            file="$ARTIFACTS/checkpoints/task1/$method/update_$update.pt"
            if [[ -f "$file" ]]; then latest="$file"; fi
        done
        if [[ -n "$latest" ]]; then
            "$PYTHON_BIN" -m atml_pa3.cli task1 train --method "$method" --resume "$latest"
        else
            "$PYTHON_BIN" -m atml_pa3.cli task1 train --method "$method"
        fi
    fi
    if [[ ! -f "$final" ]]; then echo "Missing $final" >&2; exit 1; fi
    if [[ "$method" == full ]]; then condition full_mid; fi
    condition "$method"
done
"$PYTHON_BIN" -m atml_pa3.cli task1 diagnostics
"$PYTHON_BIN" -m atml_pa3.cli task1 summarize
