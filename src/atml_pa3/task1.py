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
from .progress import tqdm, stage, message
from .training_logs import TrainingLogs
from .fast_sft import prepare_microbatch, response_loss, projected_effective_batch

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
    with stage(f"Load {spec['model']} onto CUDA ({dtype})"):
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
    with stage("Resolve immutable dataset revisions"):
        ds = dataset_spec(create=True)
    t = tokenizer()
    message("Loading GSM8K exclusion questions (dataset download/cache stage)")
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=ds["revisions"]["openai/gsm8k"])
    excluded = {normalized(r["question"]) for r in gsm}
    message("Loading MetaMath source (dataset download/cache stage)")
    source = load_dataset("meta-math/MetaMathQA", split="train", revision=ds["revisions"]["meta-math/MetaMathQA"])
    rows = []
    rejected = dict(empty=0, contamination=0, too_long=0)
    selection_bar = tqdm(total=40000, desc="Eligible training examples", unit="example")
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
        selection_bar.update(1)
        if len(rows) % 250 == 0:
            selection_bar.set_postfix(scanned=i + 1, rejected=sum(rejected.values()))
        if len(rows) == 40000:
            break
    selection_bar.close()
    if len(rows) != 40000:
        raise ValueError(f"Only {len(rows)} eligible examples")
    with (out / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with stage("Load stock task definitions and diagnostic probe splits"):
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
        for i in tqdm(rng.choice(len(docs), 32, replace=False), desc=f"Select {name} probes"):
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
    if not args.resume and ((out / "train.jsonl").exists() or (out / "config.json").exists()):
        raise FileExistsError(f"{out} already contains a run; use --resume or archive that run")
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
        optimizer_groups(model, lr), betas=(0.9, 0.999), eps=1e-8, foreach=False,
        fused=next(model.parameters()).device.type == "cuda" and not args.resume
    )
    sched = schedule(opt, 1250, 38)
    scaler = torch.amp.GradScaler("cuda")
    start = 0
    if args.resume:
        ck = load_checkpoint(args.resume, model, opt, sched, scaler)
        if ck.get("method") != method or ck.get("train_sha256") != sha256(ARTIFACT_ROOT / "data/prepared/task1/train.jsonl") or ck.get("model_spec") != model_spec():
            raise ValueError("Resume method, data hash, or model revision mismatch")
        start = ck["update"]
    overhead["optimizer_fused"] = bool(opt.param_groups[0].get("fused"))
    overhead["sft_implementation"] = "response_only_head_chunks_256"
    history = ck.get("log_history", {}) if args.resume else {}
    def prior_rows(name):
        path = out / f"{name}.jsonl"
        rows = history.get(name)
        if rows is None:
            rows = read_jsonl(path) if path.exists() else []
        return [r for r in rows if r["update"] <= start]
    kept = prior_rows("train")
    checkpoint_rows = prior_rows("checkpoints")
    if args.resume and ck.get("checkpoint_entry"):
        checkpoint_rows = [r for r in checkpoint_rows if r["update"] != start]
        checkpoint_rows.append(dict(ck["checkpoint_entry"], bytes=Path(args.resume).stat().st_size))
    journals = TrainingLogs(method, initial=dict(
        train=kept, overflow=prior_rows("overflow"), checkpoints=checkpoint_rows))
    message(f"Per-update logs stay local until successful training completion: {journals.directory}")
    previous_seconds = ck.get("elapsed_seconds", 0.0) if args.resume else 0.0
    previous_peak = ck.get("peak_gpu_bytes", 0) if args.resume else 0
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
    updates = tqdm(range(start, 1250), total=1250, initial=start, desc=f"Train {method}", unit="update")
    for update in updates:
        effective = rows[update * 32 : (update + 1) * 32]
        if len(effective) != 32:
            raise ValueError("Incomplete effective batch")
        denom = sum(sum(v != -100 for v in r["labels"][1:]) for r in effective)

        # Build CPU indices and transfer all sixteen microbatches once per update,
        # outside AMP retry. Original example order and batch boundaries are retained.
        prepare_start = time.perf_counter()
        microbatches = [device_batch(prepare_microbatch(effective[j:j+2], t.pad_token_id))
                        for j in range(0, 32, 2)]
        input_prepare_seconds = time.perf_counter() - prepare_start

        def backward():
            sft = torch.zeros((), device=next(model.parameters()).device)
            with projected_effective_batch(model):
                for batch in microbatches:
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = response_loss(model, batch) / denom
                    scaler.scale(loss).backward()
                    sft.add_(loss.detach())
                    del loss
            omega = torch.zeros((), device=next(model.parameters()).device)
            if method == "l2sp":
                # Add the unaveraged penalty once per effective batch; no second full-model autograd graph.
                for n, p in model.named_parameters():
                    delta = p.detach() - ref[n]
                    omega.add_(delta.square().sum() / 2)
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(delta, alpha=0.001 * scaler.get_scale())
            sft_value, omega_value = torch.stack((sft, omega)).detach().cpu().tolist()
            return dict(sft=sft_value, omega=omega_value)

        used_lr = opt.param_groups[0]["lr"]
        compute_start = time.perf_counter()
        values, norm, retries = amp_update(
            opt,
            scaler,
            model.parameters(),
            backward,
            lambda record: journals.append(
                "overflow",
                dict(update=update + 1, examples=(update + 1) * 32, **record),
            ),
        )
        compute_seconds = time.perf_counter() - compute_start
        sft = values["sft"]
        omega = values["omega"]
        skipped += retries
        sched.step()
        updates.set_postfix(sft=f"{sft:.4f}", lr=f"{used_lr:.2e}", skipped=skipped)
        journals.append(
            "train",
            dict(
                update=update + 1,
                examples=(update + 1) * 32,
                sft_loss=sft,
                input_prepare_seconds=input_prepare_seconds,
                compute_seconds=compute_seconds,
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
            message(f"Saving recovery checkpoint: {path} (may take time on Drive)")
            with stage(f"Write recovery update {update + 1}"):
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
                    log_history=journals.snapshot(),
                    checkpoint_entry=dict(update=update + 1, **extra),
                )
            journals.append(
                "checkpoints",
                dict(update=update + 1, bytes=path.stat().st_size, **extra),
            )
        if (update + 1) % 25 == 0:
            message(f"{method}: update {update + 1}/1250, SFT={sft:.5f}")
    if method not in ("full", "l2sp"):
        torch.cuda.synchronize()
        ts = time.perf_counter()
        export_factors(model, out / "projected_factors.pt")
        torch.cuda.synchronize()
        overhead["export_seconds"] = time.perf_counter() - ts
        overhead["export_bytes"] = (out / "projected_factors.pt").stat().st_size
    with stage("Sync completed training logs to persistent storage"):
        journals.publish(out)
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
