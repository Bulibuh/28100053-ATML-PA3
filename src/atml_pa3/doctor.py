"""Offline functional dependency checks, including a tiny random Qwen3 model."""
import importlib.metadata
import json
import torch
from .common import hardware, ARTIFACT_ROOT, write_json, causal_loss


def run(args):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from lm_eval.models.huggingface import HFLM
    import datasets, accelerate, peft, pyarrow, pandas, matplotlib, yaml
    device = "cuda" if args.require_cuda else "cpu"
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime first.")
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attention_dropout=0.0, tie_word_embeddings=True,
    )).to(device)
    inputs = torch.randint(0, 64, (2, 12), device=device)
    if args.require_cuda:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = causal_loss(model(inputs).logits, inputs)
        scaler.scale(loss / 24).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
    else:
        loss, _ = causal_loss(model(inputs).logits, inputs)
        loss.backward()
    from .task1_eval import load_qualitative_prompts
    from .common import ROOT
    prompts, questions = load_qualitative_prompts(ROOT / "configs/first_plot_questions.yaml")
    result = dict(hardware=hardware(), tiny_qwen_forward_backward=True,
                  cuda_amp_step=bool(args.require_cuda), qualitative_prompts=len(prompts),
                  packages={k: importlib.metadata.version(k) for k in
                            ["transformers", "datasets", "lm_eval", "peft", "numpy", "pyarrow"]})
    write_json(ARTIFACT_ROOT / "evidence/validation/doctor.json", result)
    print(json.dumps(result, indent=2))
