from __future__ import annotations
import json
import re
import time
import unicodedata
from pathlib import Path
import numpy as np
import torch
from .common import *
from .adapters import attach, export_factors

METHODS = ("full", "l2sp", "lora4", "lora16", "oplora16")


def normalized(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split())


def model_spec():
    path = ARTIFACT_ROOT / "data/prepared/model.json"
    if not path.exists():
        raise FileNotFoundError(
            "Run pa3 task1 prepare first to resolve immutable model revision."
        )
    return json.loads(path.read_text())


def tokenizer():
    from transformers import AutoTokenizer

    spec = model_spec()
    t = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    t.pad_token = t.eos_token
    return t


def new_model(dtype=torch.float16):
    from transformers import AutoModelForCausalLM

    spec = model_spec()
    return AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda")


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import model_info

    cfg = config()
    seed_all()
    out = ARTIFACT_ROOT / "data/prepared/task1"
    out.mkdir(parents=True, exist_ok=True)
    spec_path = ARTIFACT_ROOT / "data/prepared/model.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        if spec["model"] != cfg["model"]:
            raise ValueError("Existing model specification differs from recipe")
    else:
        info = model_info(cfg["model"], revision=cfg["revision"])
        write_json(spec_path, dict(model=cfg["model"], revision=info.sha))
    if (
        (out / "selection.json").exists()
        and (out / "probe_ids.json").exists()
        and (ARTIFACT_ROOT / "evidence/task1/data.json").exists()
    ):
        from .taskdefs import dataset_spec
        dataset_spec()
        selection = json.loads((out / "selection.json").read_text())
        if sha256(out / "train.jsonl") != selection["train_sha256"] or sha256(out / "probe_ids.json") != selection["probe_sha256"]:
            raise ValueError("Prepared training data checksum changed")
        print(
            "Reusing verified immutable model, 40,000-example selection, and fixed probes."
        )
        return
    from .taskdefs import dataset_spec, load_tasks
    ds = dataset_spec(create=True)
    t = tokenizer()
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=ds["revisions"]["openai/gsm8k"])
    excluded = {normalized(r["question"]) for r in gsm}
    source = load_dataset("meta-math/MetaMathQA", split="train", revision=ds["revisions"]["meta-math/MetaMathQA"])
    rows = []
    rejected = dict(empty=0, contamination=0, too_long=0)
    for i, r in enumerate(source):
        q = str(r.get("query") or "").strip()
        a = str(r.get("response") or "").strip()
        if not q or not a:
            rejected["empty"] += 1
            continue
        if (
            normalized(q) in excluded
            or normalized(r.get("original_question") or "") in excluded
        ):
            rejected["contamination"] += 1
            continue
        encoded = supervised_tokens(t, q, a, cfg["task1"]["max_length"])
        if encoded is None:
            rejected["too_long"] += 1
            continue
        rows.append(dict(source_index=i, **encoded))
        if len(rows) == 40000:
            break
    if len(rows) != 40000:
        raise ValueError(f"Only {len(rows)} eligible examples")
    with (out / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tasks = load_tasks(["gsm8k_cot", "hellaswag"])
    rng = np.random.default_rng(SEED)
    probes = []
    for name, count in [("gsm8k_cot", 1319), ("hellaswag", 10042)]:
        task = tasks[name]
        docs = list(
            task.test_docs() if task.has_test_docs() else task.validation_docs()
        )
        if len(docs) != count:
            raise ValueError(f"Unexpected {name} split size")
        for i in rng.choice(len(docs), 32, replace=False):
            prompt = task.doc_to_text(docs[int(i)])
            if name == "gsm8k_cot":
                prompt = t.apply_chat_template(
                    [dict(role="user", content=prompt)],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            probes.append(dict(dataset=name, id=int(i), prompt=prompt))
    write_json(out / "probe_ids.json", probes)
    write_json(
        out / "selection.json",
        dict(
            retained=40000,
            rejected=rejected,
            source_fingerprint=source._fingerprint,
            gsm_fingerprint=gsm._fingerprint,
            train_sha256=sha256(out / "train.jsonl"),
            probe_sha256=sha256(out / "probe_ids.json"),
            dataset_spec=ds,
            normalization="NFKC, casefold, punctuation to spaces, collapse whitespace",
        ),
    )
    write_json(
        ARTIFACT_ROOT / "evidence/task1/data.json",
        dict(
            selection=json.loads((out / "selection.json").read_text()),
            source_ids=[r["source_index"] for r in rows],
            probes=probes,
            model=model_spec(),
        ),
    )
    print("Prepared 40,000 ordered, untruncated examples and fixed 64-example probes.")


def construct(method, training=True):
    seed_all()
    model = new_model(torch.float32)
    overhead = {}
    if method not in ("full", "l2sp"):
        rank = 4 if method == "lora4" else 16
        overhead = attach(
            model,
            rank,
            2 * rank,
            protection=128 if method == "oplora16" else 0,
            cache=ARTIFACT_ROOT / "data/cache/op_basis" / model_spec()["revision"],
        )
    if training:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
    return model, overhead


def train(args):
    require_cuda()
    torch.cuda.reset_peak_memory_stats()
    cfg = config()["task1"]
    method = args.method
    selection = json.loads((ARTIFACT_ROOT / "data/prepared/task1/selection.json").read_text())
    if sha256(ARTIFACT_ROOT / "data/prepared/task1/train.jsonl") != selection["train_sha256"]:
        raise ValueError("Training data changed since preparation")
    rows = read_jsonl(ARTIFACT_ROOT / "data/prepared/task1/train.jsonl")
    if len(rows) != 40000:
        raise ValueError("Expected exactly 40,000 retained examples")
    t = tokenizer()
    out = ARTIFACT_ROOT / "evidence/task1" / method
    out.mkdir(parents=True, exist_ok=True)
    model, overhead = construct(method)
    model.train()
    torch.cuda.synchronize()
    reference_start = time.perf_counter()
    ref = (
        {n: p.detach().to("cuda" if method == "l2sp" else "cpu", copy=True)
         for n, p in model.named_parameters()}
        if method in ("full", "l2sp")
        else None
    )
    # L2-SP references stay on GPU to avoid transfers per step; Full FT uses CPU.
    overhead["reference_bytes"] = sum(p.numel() * 4 for p in ref.values()) if ref else 0
    torch.cuda.synchronize()
    overhead["reference_prepare_seconds"] = time.perf_counter() - reference_start
    overhead["reference_transfer_seconds"] = overhead["reference_prepare_seconds"] if method == "full" else 0.0
    lr = cfg["full_lr"] if method in ("full", "l2sp") else cfg["adapter_lr"]
    opt = torch.optim.AdamW(
        optimizer_groups(model, lr), betas=(0.9, 0.999), eps=1e-8, foreach=False
    )
    sched = schedule(opt, 1250, 38)
    scaler = torch.amp.GradScaler("cuda")
    start = 0
    if args.resume:
        ck = load_checkpoint(args.resume, model, opt, sched, scaler)
        if ck.get("method") != method or ck.get("train_sha256") != sha256(ARTIFACT_ROOT / "data/prepared/task1/train.jsonl") or ck.get("model_spec") != model_spec():
            raise ValueError("Resume method, data hash, or model revision mismatch")
        start = ck["update"]
    log = out / "train.jsonl"
    if log.exists():
        if not args.resume:
            raise FileExistsError(
                f"{log} already exists; use --resume or archive the previous run"
            )
        previous = read_jsonl(log)
        kept = [r for r in previous if r["update"] <= start]
        backup = out / f"train_before_resume_{time.time_ns()}.jsonl"
        log.rename(backup)
        for r in kept:
            append_json(log, r)
    else:
        kept = []
    previous_seconds = (
        kept[-1]["elapsed"]
        if kept
        else (ck.get("elapsed_seconds", 0.0) if args.resume else 0.0)
    )
    previous_peak = ck.get("peak_gpu_bytes", 0) if args.resume else 0
    cklog = out / "checkpoints.jsonl"
    if args.resume and cklog.exists():
        old_rows = read_jsonl(cklog)
        cklog.rename(out / f"checkpoints_before_resume_{time.time_ns()}.jsonl")
        for row in old_rows:
            if row["update"] <= start:
                append_json(cklog, row)
    write_json(
        out / "config.json",
        dict(
            method=method,
            recipe=config(),
            model=model_spec(),
            hardware=hardware(),
            overhead=overhead,
            train_sha256=sha256(ARTIFACT_ROOT / "data/prepared/task1/train.jsonl"),
        ),
    )
    begun = time.perf_counter()
    skipped = ck.get("skipped_updates", 0) if args.resume else 0
    for update in range(start, 1250):
        effective = rows[update * 32 : (update + 1) * 32]
        if len(effective) != 32:
            raise ValueError("Incomplete effective batch")
        denom = sum(sum(v != -100 for v in r["labels"][1:]) for r in effective)

        def backward():
            sft = 0.0
            for j in range(0, 32, 2):
                batch = device_batch(collate(effective[j : j + 2], t.pad_token_id))
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    summed, _ = causal_loss(logits, batch["labels"])
                    loss = summed / denom
                scaler.scale(loss).backward()
                sft += float(loss.detach())
                del logits, summed, loss, batch
            omega = torch.zeros((), device="cuda")
            if method == "l2sp":
                # Add the unaveraged penalty once per effective batch; no second full-model autograd graph.
                for n, p in model.named_parameters():
                    delta = p.detach() - ref[n]
                    omega.add_(delta.square().sum() / 2)
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(delta, alpha=0.001 * scaler.get_scale())
            return dict(sft=sft, omega=float(omega))

        used_lr = opt.param_groups[0]["lr"]
        values, norm, retries = amp_update(
            opt,
            scaler,
            model.parameters(),
            backward,
            lambda record: append_json(
                out / "overflow.jsonl",
                dict(update=update + 1, examples=(update + 1) * 32, **record),
            ),
        )
        sft = values["sft"]
        omega = values["omega"]
        skipped += retries
        sched.step()
        append_json(
            out / "train.jsonl",
            dict(
                update=update + 1,
                examples=(update + 1) * 32,
                sft_loss=sft,
                omega=float(omega),
                weighted_penalty=float(omega) * (0.001 if method == "l2sp" else 0),
                total_loss=sft + float(omega) * (0.001 if method == "l2sp" else 0),
                lr=used_lr,
                grad_norm=float(norm),
                skipped=False,
                overflow_retries=retries,
                scale=scaler.get_scale(),
                elapsed=previous_seconds + time.perf_counter() - begun,
            ),
        )
        if update + 1 in cfg["checkpoints"]:
            path = ARTIFACT_ROOT / "checkpoints/task1" / method / f"update_{update+1}.pt"
            save_checkpoint(
                path,
                model,
                opt,
                sched,
                scaler,
                update=update + 1,
                method=method,
                examples=(update + 1) * 32,
                elapsed_seconds=previous_seconds + time.perf_counter() - begun,
                peak_gpu_bytes=max(previous_peak, torch.cuda.max_memory_allocated()),
                skipped_updates=skipped,
                train_sha256=sha256(ARTIFACT_ROOT / "data/prepared/task1/train.jsonl"),
                model_spec=model_spec(),
            )
            extra = {}
            if ref is not None and (
                (method == "full" and update + 1 == 625) or update + 1 == 1250
            ):
                extra["displacement"] = (
                    sum(
                        float((p.detach().to(ref[n].device) - ref[n]).square().sum())
                        for n, p in model.named_parameters()
                    )
                    ** 0.5
                )
            append_json(
                out / "checkpoints.jsonl",
                dict(update=update + 1, bytes=path.stat().st_size, **extra),
            )
        if (update + 1) % 25 == 0:
            print(method, update + 1, sft, flush=True)
    if method not in ("full", "l2sp"):
        torch.cuda.synchronize()
        ts = time.perf_counter()
        export_factors(model, out / "projected_factors.pt")
        torch.cuda.synchronize()
        overhead["export_seconds"] = time.perf_counter() - ts
        overhead["export_bytes"] = (out / "projected_factors.pt").stat().st_size
    write_json(
        out / "resources.json",
        dict(
            trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            seconds=previous_seconds + time.perf_counter() - begun,
            peak_gpu_bytes=max(previous_peak, torch.cuda.max_memory_allocated()),
            skipped_updates=skipped,
            **overhead,
        ),
    )
