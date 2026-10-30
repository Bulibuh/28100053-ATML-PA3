# Attribution and provenance

The task implementations were generated with OpenAI Codex coding assistance for the user's requested repository. They were not copied from an existing assignment solution. The user should review, understand, and correct them as needed. The command guide is operational documentation; no graded research report or experimental interpretation is supplied.

## Materially reused data / conventions

- `answers.py` reuses the **contraction mapping data** from the official VQA evaluator, and follows its punctuation, numeric-word, and article normalization conventions. Source: https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py (consulted 30 October 2026). Scoring uses the leave-one-annotator-out equation in the assignment, applied consistently to normalized answers. The upstream mapping's capitalization is preserved rather than silently adding contractions.
- The LoRA, L2-SP, OPLoRA, visual connector, curriculum, RWC, VisConf, and adaptive-controller formulas/settings follow the supplied `ATML-PA3.pdf`. Cite the specified papers in your own report, including Hu et al. (LoRA), Li et al. (L2-SP), Xiong and Xie (OPLoRA), Liu et al. (visual instruction tuning), and Goyal et al. (complementary pairs).

- The qualitative prompts are the exact file linked by the assignment from the emergent-misalignment repository: https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml . The public-input helper fetches it without rewriting prompts or system instructions; the experiment does not invoke its external judge models. The helper also fetches the assignment-linked MathVista testmini metadata. Their content hashes are stored in `configs/public_inputs.json`.

## Libraries and datasets

- PyTorch: https://pytorch.org/
- Hugging Face Transformers / Datasets / Hub: https://github.com/huggingface/transformers and https://github.com/huggingface/datasets
- EleutherAI lm-evaluation-harness, pinned release 0.4.9; stock task prompts, filters, likelihood scoring, and metrics: https://github.com/EleutherAI/lm-evaluation-harness
- math-verify, pinned release 0.8.0, used rather than string-only equivalence for MATH: https://github.com/huggingface/Math-Verify
- Qwen3-0.6B, Qwen2.5-3B-Instruct: https://huggingface.co/Qwen
- SigLIP: https://huggingface.co/google/siglip-base-patch16-224
- MetaMathQA: https://huggingface.co/datasets/meta-math/MetaMathQA
- GSM8K / HellaSwag / BoolQ / ARC-Easy: the data sources and task definitions referenced by the pinned harness.
- MS COCO 2014 and VQAv2: https://cocodataset.org/ and https://visualqa.org/
- MathVista: https://mathvista.github.io/ ; original course rollout exports must be provided separately.
- NumPy, pandas, SciPy, PyArrow, Pillow, Matplotlib, ReportLab, pytest: standard APIs used for analysis, storage, figures, the command PDF, and verification.

Model, dataset, and tool licenses remain their respective owners' terms. Cite and attribute the actual versions/data you use. No course dataset, pretrained weight, or raw large checkpoint is distributed in this source repository.
