# ATML PA3 — Task 1

Implementation of the **Large Language Model Fine-Tuning** task in the supplied ATML-PA3 manual. This checkout contains Task 1 only. Experiments have **not** been run to completion: the checked-in validation evidence is from small offline tests, not assignment results.

Open `notebooks/Task1_Colab.ipynb` in Google Colab. The notebook includes installation, persistent storage, baseline evaluation, five training runs, checkpoint evaluation, diagnostics, and evidence export. A100 40 GB or better is recommended for the prescribed FP16 full fine-tuning recipe. Budget **at least 80–100 GB** for persistent files: full-model recovery checkpoints include FP32 parameters and AdamW states. A free 15 GB Drive account is insufficient for all recovery checkpoints. Runtime duration depends on hardware; no completion time is promised.

## Colab setup

Use a fresh Python 3.11–3.13 Colab GPU runtime. Clone this implementation after publishing it, or upload/extract the supplied `task1-colab.zip`. Work from the repository root:

```python
# Colab Python cell: installs into a separate environment, leaving the kernel alone.
import subprocess, sys
subprocess.run([sys.executable, "scripts/setup_colab.py"], check=True)
```

The installer creates `.venv-colab`, installs official `torch==2.9.1+cu128` wheels, installs the fully pinned `requirements-lock.txt`, checks dependencies, runs a tiny Qwen3 CUDA/AMP update, and runs the offline tests. It supports system Python installations without `pip` or `ensurepip`, bootstrapping pip from its TLS- and SHA-256-verified PyPI wheel. Do not use `--cpu` for assignment training. A CUDA/driver failure is reported before experiments; select a fresh Colab GPU runtime rather than installing system drivers in the notebook.

**Every subsequent command must use `.venv-colab/bin/python`, or activate that environment in every separate `%%bash` cell.** `!pip install` and `%pip` change the notebook kernel and do not install into this environment. No notebook-kernel restart is needed. No torchvision, torchaudio, bitsandbytes, flash-attn, or custom CUDA extension is required by this text-only implementation. The isolated environment prevents accidental imports of incompatible Colab vision/audio packages.

Use a persistent artifact directory in **every** execution cell, for example:

```bash
export PA3_ARTIFACT_ROOT=/content/drive/MyDrive/ATML_PA3_task1
export HF_HOME=/content/hf-cache
```

Mount Drive first with `google.colab.drive.mount('/content/drive')`. Model/dataset caches may remain on local disk; `PA3_ARTIFACT_ROOT` preserves selections, immutable revision manifests, SVD bases, checkpoints, and outputs. Keep this value unchanged across resumed sessions. Store the venv on Colab's local disk, not Drive; recreate it after a runtime reset. For command examples below, use a `%%bash` cell and export these variables first.

```bash
.venv-colab/bin/python -m atml_pa3.cli task1 prepare
# All experiments, with checkpoint resumption and reuse of completed conditions:
bash scripts/run_task1.sh
```

The full pipeline is long. Prefer the individual commands below when working across sessions. If interrupted before the first recovery checkpoint, archive/remove only that incomplete run's log directory before starting the method again. The program refuses to silently overwrite a training log. Later interruptions can resume from updates 312, 625, 937, or 1250; incomplete work after the chosen checkpoint is discarded from the active log and preserved in a backup log.

## Individual commands

```bash
PY=.venv-colab/bin/python
$PY -m atml_pa3.cli doctor --require-cuda
$PY -m atml_pa3.cli task1 prepare
$PY -m atml_pa3.cli task1 evaluate --condition baseline
$PY -m atml_pa3.cli task1 qualitative --condition baseline

$PY -m atml_pa3.cli task1 train --method full
$PY -m atml_pa3.cli task1 evaluate --condition full_mid
$PY -m atml_pa3.cli task1 evaluate --condition full

# Each run starts independently from the same immutable original checkpoint.
for method in l2sp lora4 lora16 oplora16; do
  $PY -m atml_pa3.cli task1 train --method "$method"
  $PY -m atml_pa3.cli task1 evaluate --condition "$method"
done

$PY -m atml_pa3.cli task1 diagnostics
$PY -m atml_pa3.cli task1 qualitative
$PY -m atml_pa3.cli task1 summarize
$PY -m atml_pa3.cli task1 audit
```

Both `task1 evaluate` and `task1 qualitative` accept `--batch-size`. For A100, you can request 32 for either command; the qualitative set has only 24 prompts, so it forms one batch. This larger batch has not been tested on an actual A100; reduce to 16/8/2 if memory is insufficient. Qualitative generation uses left padding and trims each response at its own EOS. Greedy outputs can differ near numerical ties when batch shapes change. Existing qualitative artifacts are reused, and their original batch size is preserved in the per-condition file.

`--batch-size 1` lowers **evaluation** memory if necessary, without changing the training recipe. Example recovery:

```bash
.venv-colab/bin/python -m atml_pa3.cli task1 train --method full \
  --resume "$PA3_ARTIFACT_ROOT/checkpoints/task1/full/update_625.pt"
```

Only load trusted, locally produced `.pt` recovery files. Recovery files contain optimizer and Python/NumPy RNG objects, so they use PyTorch's full checkpoint loader. Never load arbitrary downloaded recovery files.

## Protocol implemented

- **Data:** first 40,000 eligible `meta-math/MetaMathQA` examples in source order; no shuffling, answer truncation, or replay. Empty examples, tokenized conversations exceeding 2,048 tokens, and query/original-question matches to GSM8K test questions are excluded. Normalization is NFKC, case-folding, punctuation-to-spaces, and whitespace collapse. Source indices, rejection counts, fingerprints, checksums, and immutable model/dataset revisions are saved.
- **Formatting:** Qwen3-0.6B non-thinking chat prefix (`enable_thinking=False`) for both SFT and generation. The supplied response and `<|im_end|>` are the only prediction targets; prompts, the empty non-thinking block, and padding are masked. Loss is divided by the total valid shifted response tokens in each 32-example effective batch, rather than averaging microbatch means.
- **Training:** seed 6304; one epoch; 1,250 successful updates; microbatch 2 × accumulation 16; AdamW `(0.9, 0.999)`, epsilon `1e-8`; weight decay 0.01 except biases and tensors with fewer than two dimensions; cosine schedule with 38 warmup updates; unscaled global gradient clipping at 1.0; gradient checkpointing; FP16 AMP and dynamic loss scaling, no quantization. Overflow steps are logged and retried on the same batch/RNG without advancing the data or scheduler.
- **Methods:** Full FT and L2-SP use FP32 parameters and states at LR `2e-5`; L2-SP uses `lambda=1e-3` times the **unaveraged** half-squared distance over unique parameters, including biases/norms, counting tied parameters once. Adapters use a frozen FP16 backbone, FP32 adapter parameters/states, LR `2e-4`, dropout 0.05, random A, zero B, no trainable bias, and all seven prescribed projection targets in every layer. LoRA `(r,alpha)=(4,8)` and `(16,32)`; OPLoRA uses `(16,32)`, protection rank 128, identical initial unprojected factors to LoRA-16, exact FP32 starting-weight SVDs, and both input/output factor projections. Projected factors are exported.
- **Recovery:** updates 312/625/937/1250, corresponding to 9,984/20,000/29,984/40,000 examples. States include master/adapter weights, optimizer, scheduler, scaler, Python/NumPy/Torch/CUDA RNG, data position, data hash, and model revision. Resume rejects mismatched data/model/methods.
- **Evaluation:** shared baseline, Full FT midpoint, all five final methods. Pinned `lm_eval==0.4.9` task definitions and complete prescribed splits; zero-shot. GSM8K uses `gsm8k_cot`, non-thinking chat, greedy generation capped at 1,024 new tokens, and stock stopping/filtering. HellaSwag, BoolQ, ARC-Easy use stock completion prompts and candidate likelihoods without chat templates. FP16 evaluation with dropout disabled; adapter merge logits checked before evaluation. Raw/normalized accuracies, stock GSM8K filters/extracted answers, token traces, and truncation flags are retained. Task source hashes are checked between runs.
- **Diagnostics:** layers 0, 13, 27 `down_proj`; FP32 starting weights/bases and update subtraction; rho, eta, and per-probe R(z). The 32 GSM8K and 32 HellaSwag probes are sampled before results are inspected and frozen for every condition. The original model's inputs at the final unpadded prompt token are cached once and reused, without answer continuations. Zero updates have undefined rho (`null`). Full-model displacement is logged for Full FT midpoint/final and L2-SP final.
- **Qualitative:** the exact supplied 24-prompt YAML, one greedy non-thinking response capped at 512 tokens per condition, all 168 responses aligned by question and format. The YAML's external judge configuration and 100-sample setting are data from its source and are **not** executed. A CSV provides the 42 required first/last-question response annotations for you to complete. Existing human annotations are preserved on rerun.

## Outputs

Under `$PA3_ARTIFACT_ROOT` (repository root by default):

| Path | Contents |
| --- | --- |
| `data/prepared/` | Immutable model/dataset manifests, tokenized train selection, fixed probes |
| `data/cache/op_basis/` | FP32 singular bases, computed once |
| `checkpoints/task1/<method>/` | Four resumable checkpoints per method |
| `evidence/task1/<method>/` | Recipe/hardware, per-update losses/LR/penalty, overflows, resources, checkpoint sizes/displacements, exported factors |
| `evidence/task1/evaluation/<condition>/` | Scores, complete per-example outputs, merge check, harness config and counts |
| `evidence/task1/diagnostics/` | Cached baseline inputs and per-layer/per-probe metrics, summary CSV |
| `evidence/task1/qualitative/` | Per-condition responses, all 168 aligned responses, 42-row human annotation CSV |
| `evidence/task1/` | Learning/regularization/benchmark plots, primary score and percentage-point delta tables, resource/displacement tables, audit |

`task1 summarize` only plots/tables measured results; absent runs remain absent. `task1 audit` marks missing experiments and unfilled human annotations. Do not interpret a passing implementation test as experimental evidence. Raw datasets, caches, environments, and checkpoints are ignored by Git; commit small machine-readable result files and configurations after running your experiments.

## Local tests and limitations

```bash
python scripts/setup_colab.py --cpu
.venv-cpu/bin/python -m pytest -q
```

The offline suite checks response masking, token-weighted accumulation gradients, both OPLoRA constraints, matching initialization and rank counts, merge equivalence, checkpoint recovery (including tied weights and RNG), overflow recovery, real tiny-Qwen adapter updates, stock harness likelihood/generation/filter handling, and exact qualitative prompt counts. Separate Python 3.12 and 3.13 installations passed dependency checks and numerical tests during implementation. GPU training, full benchmark splits, actual model/dataset downloads, A100 memory use, and completion in Colab remain unverified on the development machine (no CUDA device and Hugging Face access denied by its network policy). The installer deliberately checks CUDA in Colab before any assignment run. No code can guarantee compatibility with future Colab runtime changes.

## Attribution and your report

See [ATTRIBUTIONS.md](ATTRIBUTIONS.md). The manual permits LLM coding assistance; understand and verify every submitted line. Complete your own experiments, the 42 annotations, interpretation, and PDF report. This repository supplies code and plotting utilities, not report prose or invented results.
