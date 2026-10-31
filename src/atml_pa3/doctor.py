"""Readiness checks run in the same interpreter as the experiment commands."""

from __future__ import annotations
import importlib.metadata as metadata
import json
import sys


def doctor(args):
    result = dict(
        python=sys.version.split()[0], executable=sys.executable, checks={}, errors=[]
    )
    try:
        import torch

        result.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        )
        result["checks"]["torch_import"] = True
    except Exception as exc:
        result["errors"].append(f"PyTorch import failed: {type(exc).__name__}: {exc}")
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    try:
        import torchvision

        # This compiled operator catches mismatched CPU/CUDA builds and missing torchvision::nms.
        keep = torchvision.ops.nms(
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([0.9]), 0.5
        )
        if keep.tolist() != [0]:
            raise RuntimeError("Unexpected torchvision NMS output")
        result["torchvision"] = torchvision.__version__
        result["checks"]["torchvision_operator"] = True
    except Exception as exc:
        result["errors"].append(
            f"torchvision binary/operator check failed: {type(exc).__name__}: {exc}"
        )
    try:
        from transformers import Qwen3ForCausalLM, SiglipVisionModel, AutoImageProcessor
        from lm_eval.models.huggingface import HFLM

        result["checks"]["qwen_siglip_harness_imports"] = True
        result["transformers"] = metadata.version("transformers")
    except Exception as exc:
        result["errors"].append(
            f"Model/harness import failed: {type(exc).__name__}: {exc}"
        )
    if args.require_cuda:
        if not torch.cuda.is_available():
            result["errors"].append(
                "CUDA is unavailable in this interpreter; select a Colab GPU runtime and use .venv-colab"
            )
        else:
            try:
                # Real GPU operation, beyond checking that a device name is visible.
                a = torch.ones((32, 32), device="cuda", dtype=torch.float16)
                if not torch.allclose(a @ a, torch.full_like(a, 32)):
                    raise RuntimeError("Unexpected FP16 CUDA matrix product")
                torch.cuda.synchronize()
                result["checks"]["cuda_fp16_matmul"] = True
                result["gpu_memory_gib"] = (
                    torch.cuda.get_device_properties(0).total_memory / 2**30
                )
            except Exception as exc:
                result["errors"].append(
                    f"CUDA execution failed: {type(exc).__name__}: {exc}"
                )
    result["status"] = "passed" if not result["errors"] else "failed"
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(
            "Setup failed. Run: python scripts/setup_colab.py (or --cpu for CPU tests)"
        )
