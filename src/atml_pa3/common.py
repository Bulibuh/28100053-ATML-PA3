from __future__ import annotations
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

SEED = 6304
ROOT = Path(__file__).resolve().parents[2]


def config(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs/a100.yaml").read_text())


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # FP16 required by assignment; no BF16 substitution or quantization.
    torch.backends.cuda.matmul.allow_tf32 = False


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training/generation requires a CUDA runtime. Select Colab A100 GPU."
        )
    return torch.device("cuda")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=json_default) + "\n"
    )
    tmp.replace(path)


def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def append_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, default=json_default) + "\n")


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, scaler, **progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save FP32 masters for full FT; adapter-only state for frozen-backbone runs.
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {
        n: t.detach().cpu()
        for n, t in model.state_dict().items()
        if n in names or n.endswith("scale_factor")
    }
    torch.save(
        dict(
            model=state,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            rng=rng_state(),
            **progress,
        ),
        str(path) + ".tmp",
    )
    Path(str(path) + ".tmp").replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    # Only load checkpoints produced locally by this repository.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ck["model"], strict=False)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys)
    for name, obj in [
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ]:
        if obj is not None:
            obj.load_state_dict(ck[name])
    if optimizer is not None:
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to("cuda")
    restore_rng(ck["rng"])
    return ck


def schedule(optimizer, total, warmup, floor=0.0):
    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup - 1, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def optimizer_groups(model, lr, decay=0.01):
    yes, no = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no if p.ndim < 2 or n.endswith("bias") else yes).append(p)
    return [
        dict(params=yes, lr=lr, weight_decay=decay),
        dict(params=no, lr=lr, weight_decay=0.0),
    ]


def supervised_tokens(tokenizer, question, answer, cap=None):
    messages = [dict(role="user", content=question)]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    complete = tokenizer.apply_chat_template(
        messages + [dict(role="assistant", content=answer)],
        tokenize=True,
        enable_thinking=False,
    )
    if complete[: len(prompt)] != prompt:
        raise ValueError(
            "Chat prompt is not a prefix of full example; inspect tokenizer template."
        )
    # Include im_end; exclude the trailing template newline, which is not response content.
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop = len(prompt) + complete[len(prompt) :].index(end) + 1
    ids = complete[:stop]
    if cap and len(ids) > cap:
        return None
    return dict(input_ids=ids, labels=[-100] * len(prompt) + ids[len(prompt) :])


def collate(rows, pad):
    size = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), size), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        labels[i, :n] = torch.tensor(r["labels"])
        mask[i, :n] = 1
    return dict(input_ids=ids, labels=labels, attention_mask=mask)


def causal_loss(logits, labels, per_example=False):
    target = labels[:, 1:]
    valid = target.ne(-100)
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(target)
    if not valid.sum():
        raise ValueError("No supervised response tokens")
    if per_example:
        return (loss.sum(1) / valid.sum(1).clamp_min(1)).mean()
    return loss.sum(), valid.sum()


def device_batch(batch):
    return {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}


def hardware():
    import platform

    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    )


def pair_bootstrap(values, resamples=4000, seed=SEED):
    a = np.asarray(values, dtype=float)
    if not len(a):
        raise ValueError("Cannot bootstrap an empty set")
    rng = np.random.default_rng(seed)
    means = np.array(
        [a[rng.integers(len(a), size=len(a))].mean() for _ in range(resamples)]
    )
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def amp_update(optimizer, scaler, parameters, backward, on_skip=None, max_attempts=32):
    """Commit exactly one optimizer update; retry overflow without advancing data.

    The same effective batch and dropout RNG are reused. Failed steps do not
    count as additional training examples or schedule updates, but their compute
    and reduced loss scale are retained and logged. Persistent nonfinite losses
    fail explicitly rather than silently shortening the prescribed budget.
    """
    parameters = list(parameters)
    before = rng_state()
    for attempt in range(max_attempts):
        if attempt:
            restore_rng(before)
        optimizer.zero_grad(set_to_none=True)
        metrics = backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            return metrics, float(norm), attempt
        if on_skip is not None:
            on_skip(
                dict(
                    attempt=attempt + 1,
                    old_scale=old_scale,
                    new_scale=scaler.get_scale(),
                    **metrics,
                )
            )
    raise FloatingPointError(
        f"No finite optimizer update after {max_attempts} AMP attempts; inspect overflow logs"
    )
