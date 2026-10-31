"""LoRA and OPLoRA from assignment equations; no copied external implementation."""

from pathlib import Path
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from .progress import tqdm, stage

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def singular_bases(weight, k):
    # Exact FP32 SVD, no randomized-subspace approximation.
    if k < 1 or k > min(weight.shape):
        raise ValueError("Protection rank must be positive and no larger than either matrix dimension")
    u, _, vh = torch.linalg.svd(weight.detach().float(), full_matrices=False)
    return u[:, :k].contiguous(), vh[:k, :].T.contiguous()


class LowRank(nn.Module):
    def __init__(self, base, rank, alpha, dropout=0.05, bases=None):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.B = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self._batch_factors = None
        if bases is not None:
            self.register_buffer("U", bases[0].float())
            self.register_buffer("V", bases[1].float())
        else:
            self.U = None
            self.V = None
        for p in self.base.parameters():
            p.requires_grad = False

    def factors(self):
        # Projection in FP32 even inside an FP16 forward.
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            b = self.B if self.U is None else self.B - self.U @ (self.U.T @ self.B)
            a = self.A if self.V is None else self.A - (self.A @ self.V) @ self.V.T
        return b, a

    def delta(self):
        b, a = self.factors()
        with torch.autocast(device_type=b.device.type, enabled=False):
            return self.scale * (b @ a)

    def forward(self, x):
        b, a = self._batch_factors if self._batch_factors is not None else self.factors()
        adapter_x = self.drop(x)
        if not torch.is_autocast_enabled(x.device.type):
            adapter_x = adapter_x.to(a.dtype)
        # AMP handles weight casts; do not upcast all activations only to downcast.
        return self.base(x) + self.scale * F.linear(F.linear(adapter_x, a), b).to(x.dtype)


def attach(model, rank=16, alpha=32, protection=0, targets=TARGETS, cache=None):
    for p in model.parameters():
        p.requires_grad = False
    paths = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in targets
    ]
    if not paths:
        raise ValueError("No LoRA target modules found")
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    storage = 0
    for name, module in tqdm(paths, desc="OPLoRA bases" if protection else "Attach LoRA", unit="matrix"):
        bases = None
        if protection:
            file = Path(cache) / f"{name}.pt" if cache else None
            if file and file.exists():
                bases = torch.load(
                    file, map_location=module.weight.device, weights_only=True
                )
            else:
                with stage(f"FP32 SVD: {name}"):
                    bases = singular_bases(module.weight, protection)
                if file:
                    file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(tuple(v.cpu() for v in bases), file)
            storage += sum(v.numel() * v.element_size() for v in bases)
        # Bases came from the original FP32 checkpoint, before FP16 rounding.
        module.half()
        parent, _, attr = name.rpartition(".")
        setattr(
            model.get_submodule(parent), attr, LowRank(module, rank, alpha, bases=bases)
        )
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.half()
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    return dict(
        basis_bytes=storage,
        preprocessing_seconds=time.perf_counter() - started,
        adapter_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )


def export_factors(model, path):
    state = {}
    for n, m in model.named_modules():
        if isinstance(m, LowRank):
            b, a = m.factors()
            state[n] = {"A": a.detach().cpu(), "B": b.detach().cpu(), "scale": m.scale}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


@torch.no_grad()
def merge(model):
    # Build FP32 correction then round once into FP16 inference weight.
    for n, m in list(model.named_modules()):
        if isinstance(m, LowRank):
            m.base.weight.copy_(
                (m.base.weight.float() + m.delta()).to(m.base.weight.dtype)
            )
            parent, _, attr = n.rpartition(".")
            setattr(model.get_submodule(parent), attr, m.base)
    return model
