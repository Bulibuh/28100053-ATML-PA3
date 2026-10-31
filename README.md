# ATML PA3: reproducible experiment repository

Code for all three tasks in the supplied Fall 2026 assignment. The default recipe targets a **single Colab A100 40 GB** while preserving the assignment's fixed training budgets, batches, FP16 computation, and evaluation protocols.

**Experiment status:** implementations and CPU integration tests are provided; the full GPU experiments have **not** been run. There are no invented benchmark numbers, training curves, model outputs, or qualitative findings. `evidence/status.json` tracks missing experiment artifacts. The course rollout package and its README/model revision are still external inputs. The public prompt YAML and MathVista metadata linked inside the PDF are supported directly and can be fetched with the checksum-verified helper below.

- [Colab command guide (PDF)](docs/ATML_PA3_Colab_Commands.pdf)
- [Same guide as editable Markdown](docs/COLAB_COMMANDS.md)
- [Colab notebook](notebooks/ATML_PA3_Colab.ipynb)
- [Evidence map](evidence/README.md)
- [External inputs and canonical schemas](data/external/README.md)
- [Attribution and implementation notes](ATTRIBUTIONS.md)

The command PDF is operational documentation, **not** the graded report. You must write the report, interpret results, inspect labels, and complete qualitative annotations yourself. Coding assistance was used to generate this repository; review and understand it before submission.

## Quick start

The pinned setup supports Python 3.11–3.13. In Colab, select an A100 runtime and open this repository directory. If using the delivered ZIP, follow the PDF's source-upload cell.

```bash
python scripts/setup_colab.py
source .venv-colab/bin/activate
pa3 doctor --require-cuda
```

The installer creates an isolated `.venv-colab` with the official matching CUDA 12.8 pair **torch 2.9.1 / torchvision 0.24.1**. It runs dependency checks, torchvision's compiled NMS operator, Qwen/SigLIP/harness imports, an actual FP16 CUDA operation, and pytest. It leaves Colab's shared packages untouched, avoiding shared-environment dependency conflicts and mismatched torchvision binaries. The first installation requires several GB of disk for the separate CUDA stack.

In a `%%bash` cell use `python`, **not `!python`**. Activate `.venv-colab` in each new shell cell; shell activation does not persist between notebook cells. The updated notebook selects the same virtualenv for every subprocess. Do not run the previous global-install or torchvision `--no-deps` workaround cells.

For local CPU implementation checks (not GPU validation), use `python scripts/setup_colab.py --cpu`, then `source .venv-cpu/bin/activate`. Training still requires CUDA. No quantization, BF16 substitution, or batch-size enlargement is used.

```bash
# Small public inputs linked in the assignment PDF.
python scripts/download_public_inputs.py
# Task 1: the original linked YAML is read directly, preserving JSON system messages.
export QUALITATIVE_PROMPTS="$PWD/data/external/first_plot_questions.yaml"
bash scripts/run_task1.sh

# Task 2: official annotations, selected-image downloads, FP16 feature caching.
python scripts/download_annotations.py
export COCO_ANNOTATIONS="$PWD/data/external/coco/annotations"
bash scripts/run_task2.sh

# Task 3: supply the assignment's actual files; adapt the schema from its README.
export MATH_ROLLOUTS=/content/ATML-PA3-Data/LLM
export MATHVISTA_ROLLOUTS=/content/ATML-PA3-Data/VLM
export MATHVISTA_METADATA="$PWD/data/external/mathvista_testmini.json"
export ROLLOUT_README=/content/ATML-PA3-Data/README.MD
export LIVE_REVISION=REPLACE_WITH_THE_40_CHARACTER_SHA_IN_THE_SUPPLIED_README
bash scripts/run_task3.sh

pa3 evidence
pa3 audit
```

These paths are examples, not claims about the missing rollout archive's actual structure. Pass the real locations. The command guide gives each individual command, checkpoint resumption, Drive backups, optional 49-token training, and artifact export.

## Repository layout

```text
configs/                 fixed A100 recipe and adaptable rollout column mapping
src/atml_pa3/
  common.py              losses, reproducibility, checkpoints, schedules
  adapters.py            LoRA/OPLoRA factor projections, FP32 SVD, merging
  task1.py               ordered MetaMathQA filtering and five training runs
  task1_eval.py          harness benchmarks, probes, diagnostics, qualitative output
  vlm_data.py            image-disjoint COCO/VQAv2 splits and feature caching
  task2.py               connector, visual tokens, curriculum, paired evaluations
  answers.py             VQA normalization, math equivalence, extraction
  task3.py               offline scoring, consensus, bootstrap, adaptive policies
  live.py                fixed-revision 50-question FP16 inference comparison
  evidence.py            tables, figures, missing-artifact audit
scripts/                 task runners, official annotation downloads, PDF builder
notebooks/               Colab runner
checkpoints/             large local resumable states (Git-ignored)
data/prepared/           local selected/tokenized data (Git-ignored)
data/cache/              local features and singular bases (Git-ignored)
evidence/task{1,2,3}/     measured outputs when experiments are actually run
 evidence/validation/    actual CPU test results; never assignment experiment results
```

## Protocol coverage

| Task | Implementation and saved evidence |
| --- | --- |
| 1.1–1.2 Full FT / L2-SP | Independent starts; fixed ordered 40k examples; 1,250 effective batches; global response-token loss; fixed starting reference counted once per unique parameter; unaveraged penalty; displacement and overhead logs |
| 1.3–1.4 LoRA / OPLoRA | Seven projection types per layer; rank 4/16; matched rank-16 initialization; FP32 exact singular bases; factorwise projections; projected export and merge checks |
| 1.5 Benchmarks | Pinned lm-eval task definitions; seven conditions; complete splits with count assertions; zero-shot; non-thinking greedy GSM8K; stock completion likelihood for retention tasks; raw samples/extractions/truncation |
| 1.6 Diagnostics | Layers 0/13/27; common starting bases; rho/eta; fixed 32+32 probes; baseline last-prompt-token inputs; FP32 local-update corrections; plot/table builder |
| 1.7 Qualitative | 24 supplied prompts × 7 conditions; greedy cap 512; 42-row first/last-question annotation CSV for human completion |
| 2.1 Interface / data | 196 SigLIP patch features, normalization once, cached FP16; calibrated two-layer connector; masked visual/prompt tokens; explicit positions; image-disjoint dev/test and split-qualified training isolation |
| 2.2–2.4 Curriculum | 2,250/1,500/750 updates; effective batch 16; example-weighted response loss; stage-specific rates/warmup; paired stochastic replay draws; checkpoint every 250; dev before training/every 500/endpoints |
| 2.5–2.6 Grounding | Correct/swap/mismatched/no-image; original references; local VQA soft score; per-type results; pair accuracy, identical/empty rates, reference NLL/time; paired 4,000-resample intervals; fixed-rule example sheets |
| 2.7 Optional | Independent 49-token curriculum using spatial 2×2 average pooling, same data/budgets |
| 3.1–3.3 Offline | One-to-one table joins; token alignment; fixed 200/800 splits; ten nested rollout orders; all selection/RWC rules; prefix-relative VisConf; failure decomposition; signal/quartile/covariance diagnostics; tokenwise fusion |
| 3.4 Live | Fixed supplied revision; batch 1 FP16; longer/three alternatives/shared-initial refinement; balanced last-box extraction; seeds/prompts/tokens/truncation/time/memory; revision transitions |
| 3.5–3.6 Adaptive | Four/eight/sixteen checks and 32 fallback; agreement/quality grid tuned only on dev; frozen settings saved before held-out evaluation; paired question bootstrap; stopping distributions and retrospective failure records |

## A100 settings and practical constraints

- FP16 autocast + dynamic scaling, FP32 trainable parameters and AdamW states, no quantization. Full FT masters and L2-SP reference stay FP32 on GPU. This avoids repeated reference transfers; reference storage is logged.
- Native PyTorch SDPA uses supported CUDA attention kernels. No custom FlashAttention build is required. Gradient checkpointing uses non-reentrant mode so visual gradients propagate through frozen Qwen.
- Task 1 keeps microbatch **2**, accumulation **16**, length **2,048**. Task 2 keeps microbatch **8**, accumulation **2**. Longer/fatter batches would change the required recipe and are not used.
- SigLIP caches images in batches of 64 and leaves the training GPU before Qwen starts. Selected images are downloaded with eight workers; CPU feature caching avoids repeat deserialization during curriculum replay.
- Exact SVD for OPLoRA is an up-front cost. Bases are cached by immutable model revision. Checkpoint files include full optimizer state; Full FT recovery files can each be several GB. Keep tens of GB free and back up checkpoints to Drive. A100 VRAM is not persistent disk storage.
- Full experiments may span multiple Colab sessions. No runtime or accuracy estimate is claimed without a GPU measurement. Recovery is supported; overflowed steps are logged separately and retried on the same effective batch and dropout RNG before advancing the successful-update counter; persistent nonfinite loss fails explicitly.

## Reproducibility and limitations

- `task1 prepare` resolves the requested Qwen revision to a commit SHA and stores it in `data/prepared/model.json`. Reuse that file and the prepared data across all methods. Seeds, data IDs/fingerprints/checksums, configuration, resources, and raw results are saved.
- `configs/a100.yaml` records the assignment recipe. These are protocol constants, not a hyperparameter-search interface: several constants are intentionally enforced in code. The 49-token optional variant has a separate CLI option. Signal quartiles use stable within-question ranks (eight of 32 candidates at each end); undefined constant-signal correlations are recorded as missing, with valid-rollout counts.
- External inputs are deliberately not replaced by synthetic data. The native linked qualitative YAML is parsed directly, including its JSON system prompts; upstream judge instructions and 100-sample settings are not used. Adapt rollout column names from the supplied README; do not guess whether attention fields are totals or means.
- MathVista metadata must be the released question metadata, with choice lists, answer type, precision, and question type. All extracted-answer groups are formed without gold; gold only grades selections and retrospective analyses. Empty extractions remain in denominators, votes, and costs.
- Natural VQA pairs use consensus support of at least 8/10 for both images. Annotation-derived counts include annotated instances (including crowd annotations); inspect the selected labels independently of predictions, as COCO annotations can omit objects.
- The official VQA punctuation/digit/article/contraction normalization data are attributed in `ATTRIBUTIONS.md`. Task 3 math-verify version and input checksums are logged.
- Audit reports artifact presence, not scientific correctness. Inspect actual example counts, split IDs, full metrics, skipped updates, annotations, and deviations before submission. An interrupted run or blank human annotation does not become complete merely because an output file exists.
- CPU tests use small, clearly synthetic fixtures and a randomly initialized tiny Qwen. They verify implementations and API integration, not A100 memory fit or model quality. The real models/datasets and full Colab run are not validated here.

## Publication

The source repository is https://github.com/Bulibuh/28100053-ATML-PA3. Clone it using the Colab guide to reproduce the experiments. Keep raw datasets, feature caches, and large checkpoints out of Git. Include the required repository URL in your own report.
