# Evidence index

`status.json` is an explicit missing-artifact audit. **No full assignment experiments were executed in this GPU-free session.** Only `validation/` contains measured local test evidence. Running the commands fills the following folders with actual results; unrun experiments stay missing.

| Required evidence | Output |
| --- | --- |
| T1 ordered selection, leakage exclusions, probe IDs | `task1/data.json` |
| T1 losses/rates/penalties/skipped steps/config | `task1/{full,l2sp,lora4,lora16,oplora16}/{train.jsonl,config.json}` |
| T1 time/memory/parameters/reference/basis/export costs | each method's `resources.json`, `checkpoints.jsonl`; combined `task1/resources.csv` |
| T1 seven-condition/four-benchmark table and baseline deltas | `task1/benchmark_comparison.csv`, `primary_benchmarks.csv` |
| T1 all benchmark samples/extractions/truncation | `task1/evaluation/CONDITION/*_samples.jsonl` |
| T1 adapter equivalence verification | adapter condition's `merge_check.json` |
| T1 losses and regularization/displacement | `task1/learning_curves.png`, `regularization.png`, `displacements.csv` |
| T1 weight/activation geometry | `task1/diagnostics/{metrics.json,summary.csv,baseline_activations.pt}` |
| T1 168 responses and human 42-response rubric | `task1/qualitative/{responses.json,annotations_to_complete.csv}` |
| T2 splits, pairs, all references, mismatched-image IDs | `task2/splits.json` |
| T2 encoder preprocessing/cache costs | `task2/feature_cache.json` |
| T2 training, stochastic mixture counts, dev raw NLL, calibration | `task2/tokens_196/{train.jsonl,development.jsonl,calibration.json,config.json}` |
| T2 stage resources and completion | `task2/tokens_196/{resources.jsonl,completed.json}` |
| T2 stage curves and swap penalties | `task2/tokens_196/{learning_curve.png,development_curves.png,development_metrics.csv}` |
| T2 per-example generations/references/scores/NLL/time | `task2/tokens_196/final/predictions.json` |
| T2 condition/type/pair results and paired CIs | `task2/tokens_196/final/{summary.json,results.csv,paired_contrasts.json}` |
| T2 aligned visual examples, fixed selection rule | `task2/tokens_196/final/{qualitative_examples.json,PAIR_ID.png}` |
| T2 optional compression | separate identical structure under `tokens_49/` |
| T3 split IDs, ten nested candidate orders, provenance | `task3/offline/{math,mathvista}/{splits.json,orders.json,provenance.json}` |
| T3 accuracy/oracle/cost/failure data and figures | each offline dataset's `per_question_fixed.parquet`, `fixed_summary.csv`, `accuracy_cost_curves.png`, `failure_decomposition.csv` |
| T3 paired question-level differences/intervals | each offline dataset's `paired_differences.json` |
| T3 signal associations, confidence/vision fusion | each offline dataset's `signals.json`; MathVista `signal_trajectories.png` |
| T3 fixed-rule aligned examples | `illustrative_trajectories.json`, `selection_disagreements.json`, `premature_stop_examples_*.json` |
| T3 frozen thresholds and held-out stopping behavior | `selected_policies.json`, `development_grid.json`, `adaptive_*.parquet`, `adaptive_summary.json` |
| T3 all live prompts/responses/tokens/seeds/time/memory | `task3/live/generations.json` |
| T3 three strategies and revision transitions | `task3/live/{per_question.json,summary.csv,revision_transitions.json,qualitative_examples.json}` |

## Human evidence

Inspect selected COCO labels independently of model predictions. Complete the 42-row Task 1 rubric with format compliance, coherence/relevance, repetition, mathematical intrusion, inappropriate content, excerpts, and your notes. Write your own interpretation and 8-page report. Machine-generated numbers and plots are inputs to your analysis, not a substitute for it.

## Artifact sizes

Raw fixed-budget and adaptive records are Parquet to keep storage practical; they are intentionally Git-ignored by default because they can be large. Keep them in your Drive evidence archive. To publish manageable required evidence, explicitly `git add -f` only the Parquet files you have reviewed, or document an accessible external archive. Do not add raw course datasets or model weights to Git. `*.pt` states are likewise local/Drive artifacts.

## Validation versus experiments

`validation/pytest.xml` and `validation/summary.json` describe CPU unit/integration tests with synthetic fixtures and a tiny randomly initialized Qwen. They do not demonstrate that the training/evaluation experiments ran, nor that reported model performance improved. The test runner's real exit code is preserved.
