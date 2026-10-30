# Required external inputs

The uploaded PDF contains the protocol, not the datasets or prompt files. Do not substitute generated rollouts or invented prompts.

## Task 1

`pa3 task1 prepare` downloads `Qwen/Qwen3-0.6B`, `meta-math/MetaMathQA`, GSM8K, and the harness's probe datasets from Hugging Face. It requires Internet access and adequate cache space. Training targets all five methods at the prescribed 40,000 examples; there is no shortened run presented as a full assignment result.

The PDF links the original prompts at https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml and MathVista metadata at https://github.com/lupantech/MathVista/blob/main/data/testmini.json. Fetch their inspected, checksum-pinned versions with:

```bash
python scripts/download_public_inputs.py
```

The files go under `data/external/`. The native prompt YAML is supported directly: its `id`, one `paraphrases` entry, and optional `system` message are retained; JSON system messages stay system messages. Its judge prompts, API model names, and 100-sample settings are not executed. The assignment uses exactly one greedy response per prompt/condition. The reader also accepts canonical records with `question_id`, `format` (`plain`/`json`/`template`), `prompt`, and optional `system`.

Content hashes in `configs/public_inputs.json` pin the versions inspected during implementation. Upstream changes cause a checksum error rather than silently changing the experiment. Preserve the original file, its attribution, and prompt wording/order.

## Task 2

`python scripts/download_annotations.py` uses published COCO/VQA HTTPS archive paths. Alternatively download from https://cocodataset.org/#download and https://visualqa.org/download.html, extract, and collect these ten JSON files in one folder:

```text
captions_train2014.json
captions_val2014.json
instances_train2014.json
instances_val2014.json
v2_OpenEnded_mscoco_train2014_questions.json
v2_OpenEnded_mscoco_val2014_questions.json
v2_mscoco_train2014_annotations.json
v2_mscoco_val2014_annotations.json
v2_mscoco_train2014_complementary_pairs.json
v2_mscoco_val2014_complementary_pairs.json
```

Complementary-pair JSON supports a list of `[question_id, complementary_question_id]`, or objects with those two named fields, optionally under `pairs`/`complementary_pairs`. If the released structure differs, adapt the reader explicitly after inspecting it.

`pa3 task2 cache --download` downloads only selected JPEGs to `data/external/coco/images/{train,val}2014/COCO_...jpg`. It caches 196×768 FP16 features with deterministic preprocessing. Supply locally downloaded COCO images via `--images` to avoid downloading. Keep the default directory if you want `pa3 evidence` to render image sheets automatically. URLs can change; failed downloads must be resolved from official sources, with TLS verification retained.

## Task 3

The PDF links the course rollout folder at https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23?usp=sharing. Supply its original `ATML-PA3-Data` files and `README.MD`. Released MathVista metadata is fetched by the public-input helper. The course Drive host was blocked by this cloud machine's outbound policy; access from Colab still needs to be checked.

In Colab, download the folder with `gdown --folder` (install `gdown==5.2.0`), or use your authorized Drive/browser access. If the folder is not publicly downloadable, use the course-provided access; no credential is embedded in the repository. Two model folders must contain:

- `generations.parquet`
- `probability_metrics.parquet`
- VLM additionally: `attention_metrics.parquet`

The reader requires a one-to-one join on `(dataset, sample_id, rollout_id)`, 1,000 questions and 32 rollouts per question. Edit **only the right-hand column names** in `configs/rollout_schema.json` to match the actual release:

| Canonical field | Meaning |
| --- | --- |
| token_ids | Generated token ID list excluding termination, aligned with probability/attention entries |
| token_count | Retained generated-token count, excluding termination |
| extracted_answer | Supplied extracted answer, possibly missing |
| gold_answer | Reference answer; never passed into a selector or stopping controller |
| problem | Original problem text; required for live MATH prompts |
| logp | Per-token sampled-token log probabilities of raw distributions |
| kl_u_p | Per-token KL(U || model distribution) |
| image_attention | Per-token image-group attention totals |
| prompt_attention | Per-token prompt-text-group attention totals |
| generated_attention | Per-token generated-text-group attention totals |

Attention must be group totals, not independently normalized group averages. List lengths must equal `token_count`. Invalid joins or alignment fail loudly.

The schema expects `problem` for live MATH inputs. If the VLM export omits this unused text field, its released MathVista question is read from the supplied metadata by `sample_id`. If live MATH problem text is in a metadata table rather than generations, join it from the release. Do not invent or reconstruct problem statements from answers.

Metadata accepts a JSON dictionary keyed by string question/sample IDs or a list with `pid`/`sample_id`. Each MathVista record must contain `question_type`, `answer_type`, `choices` for multiple choice, and `precision` for floats. Use the released values.

The live command requires the exact 40-character Qwen2.5-3B-Instruct revision in the original README. It checks that the SHA occurs in that README and resolves to the requested checkpoint. This value cannot be inferred from the assignment PDF.

## Network destinations

Colab normally provides Internet access. A restricted cloud environment additionally needs Hugging Face model/dataset/CDN hosts (including redirects), `images.cocodataset.org`, `s3.amazonaws.com`, and whichever official host serves the supplied course archive. Package installation uses PyPI and Python wheel hosts. No API keys are required by the public-model workflow; do not put credentials in source, command guides, or evidence.
