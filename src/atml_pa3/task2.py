from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import *
from .task1 import new_model, tokenizer
from .adapters import attach
from .answers import vqa_normalize, vqa_score


@lru_cache(maxsize=20000)
def cached_feature(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class VLM(nn.Module):
    def __init__(self, visual_tokens=196, lm=None, t=None):
        super().__init__()
        self.lm = new_model() if lm is None else lm
        self.lm.requires_grad_(False)
        self.device = next(self.lm.parameters()).device
        hidden = self.lm.config.hidden_size
        self.connector = (
            nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            .float()
            .to(self.device)
        )
        self.register_buffer("scale_factor", torch.tensor(1.0, device=self.device))
        self.visual_tokens = visual_tokens
        self.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.lm.config.use_cache = False
        self.t = tokenizer() if t is None else t
        self.adapters_enabled = False

    def enable_adapters(self):
        if not self.adapters_enabled:
            attach(self.lm, 16, 32, targets=("q_proj", "k_proj", "v_proj", "o_proj"))
            self.adapters_enabled = True

    def features(self, r, image_id=None):
        path = (
            ROOT
            / f'data/cache/task2/{r["split"]}_{image_id if image_id is not None else r["image_id"]}.pt'
        )
        x = cached_feature(str(path)).to(self.device, non_blocking=True).float()
        if self.visual_tokens == 49:
            x = (
                x.reshape(14, 14, 768)
                .reshape(7, 2, 7, 2, 768)
                .mean((1, 3))
                .reshape(49, 768)
            )
        return x

    def prompt(self, r):
        qids = self.t.encode(r["question"], add_special_tokens=False)[:80]
        q = self.t.decode(qids, skip_special_tokens=False)
        marker = "__VISUAL_INSERTION_POINT__"
        text = self.t.apply_chat_template(
            [dict(role="user", content=marker + q)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = text.split(marker)
        return self.t.encode(before, add_special_tokens=False), self.t.encode(
            after, add_special_tokens=False
        )

    def pack(
        self, rows, condition="correct", targets=True, exclude_end=False, left_pad=False
    ):
        seqs = []
        labels = []
        emb = self.lm.get_input_embeddings()
        end = self.t.convert_tokens_to_ids("<|im_end|>")
        for r in rows:
            pre, post = self.prompt(r)
            pieces = [emb(torch.tensor(pre, device=self.device))]
            if condition != "no_image":
                image = (
                    r["image_id"]
                    if condition == "correct"
                    else r[f"{condition}_image_id"]
                )
                x = self.features(r, image)
                # FP32 trainable connector/masters, FP16 computation under outer autocast.
                pieces.append(
                    (self.connector(x) * self.scale_factor).to(emb.weight.dtype)
                )
            pieces.append(emb(torch.tensor(post, device=self.device)))
            prompt = torch.cat(pieces)
            y = []
            if targets:
                cap = 64 if r["kind"] == "caption" else 16
                y = self.t.encode(r["target"], add_special_tokens=False)[:cap]
                if not y:
                    raise ValueError(f'Empty VLM target: {r["id"]}')
                seq = torch.cat(
                    [prompt, emb(torch.tensor(y + [end], device=self.device))]
                )
                lab = [-100] * len(prompt) + y + ([-100] if exclude_end else [end])
            else:
                seq = prompt
                lab = [-100] * len(prompt)
            seqs.append(seq)
            labels.append(lab)
        size = max(len(s) for s in seqs)
        output = []
        labs = []
        masks = []
        for seq, lab in zip(seqs, labels):
            n = size - len(seq)
            padding = torch.zeros(n, seq.shape[1], device=self.device, dtype=seq.dtype)
            output.append(torch.cat([padding, seq] if left_pad else [seq, padding]))
            labs.append(([-100] * n + lab) if left_pad else (lab + [-100] * n))
            masks.append(
                ([0] * n + [1] * len(seq)) if left_pad else ([1] * len(seq) + [0] * n)
            )
        mask = torch.tensor(masks, device=self.device)
        return dict(
            inputs_embeds=torch.stack(output),
            attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
        ), torch.tensor(labs, device=self.device)

    def forward(self, **kw):
        return self.lm(**kw)


@torch.no_grad()
def calibrate(model, train):
    # Fixed RMS matching rule on first 128 distinct training images only.
    seen = set()
    visual = []
    text = []
    for r in train:
        if r["image_id"] in seen:
            continue
        seen.add(r["image_id"])
        x = model.features(r)
        visual.append(float(model.connector(x).square().mean()))
        ids = model.t.encode(r["question"], add_special_tokens=False)[:80]
        text.append(
            float(
                model.lm.get_input_embeddings()(torch.tensor(ids, device=model.device))
                .float()
                .square()
                .mean()
            )
        )
        if len(seen) == 128:
            break
    scale = (np.mean(text) / max(np.mean(visual), 1e-12)) ** 0.5
    model.scale_factor.fill_(scale)
    return dict(
        rule="sqrt(mean text-embedding square / mean unscaled connector-output square)",
        training_image_ids=sorted(seen),
        scale=scale,
    )


def pools(rows):
    p = defaultdict(list)
    pair = defaultdict(list)
    for r in rows:
        if r["kind"] == "caption":
            p["caption"].append(r)
        else:
            pair[r["pair_id"]].append(r)
    for group in pair.values():
        if len(group) != 2:
            raise ValueError("Training pair is incomplete")
        p[
            "natural" if group[0]["id"].startswith("natural") else group[0]["kind"]
        ].append(group)
    return p


def stage_batches(stage, rows, rng):
    if stage == 0:
        captions = [r for r in rows if r["kind"] == "caption"]
        if len(captions) != 18000:
            raise ValueError("Expected 18,000 caption examples")
        for _ in range(2):
            for j in range(0, len(captions), 16):
                yield captions[j : j + 16]
    else:
        p = pools(rows)
        keys = (
            ["presence", "counting", "caption"]
            if stage == 1
            else ["natural", "presence", "counting", "caption"]
        )
        probs = [0.5, 0.25, 0.25] if stage == 1 else [0.7, 0.1, 0.1, 0.1]
        for _ in range([2250, 1500, 750][stage]):
            batch = []
            for draw in range(8):
                kind = str(rng.choice(keys, p=probs))
                if kind == "caption":
                    batch.extend(
                        p[kind][int(rng.integers(len(p[kind])))] for _ in range(2)
                    )
                else:
                    batch.extend(p[kind][int(rng.integers(len(p[kind])))])
            yield batch


@torch.no_grad()
def nll_records(model, rows, condition, exclude_end=True):
    model.eval()
    result = []
    for j in range(0, len(rows), 8):
        chunk = rows[j : j + 8]
        with torch.autocast("cuda", dtype=torch.float16):
            batch, labels = model.pack(chunk, condition, exclude_end=exclude_end)
            logits = model(**batch).logits
        target = labels[:, 1:]
        valid = target.ne(-100)
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(target)
        vals = losses.sum(1) / valid.sum(1)
        for r, v in zip(chunk, vals):
            result.append(dict(id=r["id"], nll=float(v)))
    return result


def with_partner(rows):
    index = {r["id"]: r for r in rows}
    return [
        dict(r, swap_image_id=index[r["partner"]]["image_id"]) if "partner" in r else r
        for r in rows
    ]


def development(model, dev, update, out):
    for suite in ["caption", "ground", "natural"]:
        rows = (
            [r for r in dev if r["kind"] == "caption"]
            if suite == "caption"
            else [r for r in dev if r["id"].startswith(suite)]
        )
        rows = with_partner(rows)
        for condition in (
            ["correct"] if suite == "caption" else ["correct", "swap", "no_image"]
        ):
            vals = nll_records(model, rows, condition, exclude_end=suite != "caption")
            append_json(
                out / "development.jsonl",
                dict(
                    update=update,
                    suite=suite,
                    condition=condition,
                    mean_nll=float(np.mean([v["nll"] for v in vals])),
                    per_example=vals,
                ),
            )


def train(args):
    require_cuda()
    seed_all()
    cfg = config()["task2"]
    rows = json.loads((ROOT / "data/prepared/task2/train.json").read_text())
    dev = json.loads((ROOT / "data/prepared/task2/dev.json").read_text())
    model = VLM(args.visual_tokens)
    out = ROOT / f"evidence/task2/tokens_{args.visual_tokens}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.jsonl").exists() and not args.resume:
        raise FileExistsError(
            "Training output exists; resume or archive it before starting again"
        )
    calibration = calibrate(model, rows)
    write_json(out / "calibration.json", calibration)
    if args.resume:
        ck_progress = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck_progress["visual_tokens"] != args.visual_tokens:
            raise ValueError("Resume visual-token setting mismatch")
        for file in ["train.jsonl", "development.jsonl"]:
            log = out / file
            if log.exists():
                previous = read_jsonl(log)
                log.rename(out / f"{file}.before_resume_{time.time_ns()}")
                for row in previous:
                    if row["update"] <= ck_progress["update"]:
                        append_json(log, row)
    write_json(
        out / "config.json",
        dict(
            recipe=config(),
            visual_tokens=args.visual_tokens,
            hardware=hardware(),
            train_split_sha256=sha256(ROOT / "data/prepared/task2/train.json"),
        ),
    )
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume["visual_tokens"] != args.visual_tokens:
            raise ValueError("Resume visual-token setting mismatch")
    if resume and resume["stage"] >= 1:
        model.enable_adapters()
    if resume:
        model.load_state_dict(resume["model"], strict=False)
        restore_rng(resume["rng"])
    else:
        development(model, dev, 0, out)
    rng = np.random.default_rng(SEED)
    total_time = time.perf_counter()
    global_update = 0
    prior_total = resume.get("total_seconds", 0.0) if resume else 0.0
    resource_log = out / "resources.jsonl"
    if resume and resource_log.exists():
        old = read_jsonl(resource_log)
        resource_log.rename(out / f"resources_before_resume_{time.time_ns()}.jsonl")
        for row in old:
            if row["stage"] < resume["stage"] + 1 or (
                row["stage"] == resume["stage"] + 1
                and resume["stage_update"] == cfg["stage_updates"][resume["stage"]]
            ):
                append_json(resource_log, row)
    for stage, total in enumerate(cfg["stage_updates"]):
        # Stage-specific sampler seed makes resume independent of replaying earlier stages.
        rng = np.random.default_rng(SEED + stage)
        batches = stage_batches(stage, rows, rng)
        if resume and (
            stage < resume["stage"]
            or (stage == resume["stage"] and resume["stage_update"] == total)
        ):
            global_update += total
            if stage == resume["stage"]:
                torch.save(resume["model"], out / f"stage_{stage+1}_weights.pt")
            continue
        if stage:
            model.enable_adapters()
        groups = optimizer_groups(model.connector, cfg["connector_lr"][stage])
        if stage:
            groups += optimizer_groups(model.lm, cfg["adapter_lr"][stage])
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, foreach=False)
        scheduler = schedule(opt, total, math.ceil(0.05 * total), floor=0.1)
        scaler = torch.amp.GradScaler("cuda")
        start = 0
        if resume and stage == resume["stage"]:
            load_checkpoint(args.resume, model, opt, scheduler, scaler)
            start = resume["stage_update"]
            # Generate and discard only sampler draws, never model forwards, to restore data position.
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.cuda.reset_peak_memory_stats()
        ts = time.perf_counter()
        same_stage = resume is not None and stage == resume["stage"]
        prior_stage = resume.get("stage_seconds", 0.0) if same_stage else 0.0
        prior_peak = resume.get("peak_gpu_bytes", 0) if same_stage else 0
        mixture = Counter(resume.get("mixture_counts", {})) if same_stage else Counter()
        skipped = resume.get("skipped_updates", 0) if same_stage else 0
        for local, batch_rows in enumerate(batches):
            if local < start:
                continue
            model.train()

            def backward():
                total_loss = 0.0
                for j in range(0, 16, 8):
                    with torch.autocast("cuda", dtype=torch.float16):
                        batch, labels = model.pack(batch_rows[j : j + 8])
                        logits = model(**batch).logits
                        loss = causal_loss(logits, labels, per_example=True) / 2
                    scaler.scale(loss).backward()
                    total_loss += float(loss.detach())
                return dict(loss=total_loss)

            lrs = [g["lr"] for g in opt.param_groups]
            values, grad, retries = amp_update(
                opt,
                scaler,
                model.parameters(),
                backward,
                lambda record: append_json(
                    out / "overflow.jsonl",
                    dict(
                        update=sum(cfg["stage_updates"][:stage]) + local + 1,
                        stage=stage + 1,
                        **record,
                    ),
                ),
            )
            total_loss = values["loss"]
            skipped += retries
            scheduler.step()
            global_update = sum(cfg["stage_updates"][:stage]) + local + 1
            kinds = Counter(r["kind"] for r in batch_rows)
            mixture.update(kinds)
            append_json(
                out / "train.jsonl",
                dict(
                    update=global_update,
                    stage=stage + 1,
                    stage_update=local + 1,
                    loss=total_loss,
                    mixture=dict(kinds),
                    lr=lrs,
                    skipped=False,
                    overflow_retries=retries,
                    grad_norm=float(grad),
                ),
            )
            if global_update % 250 == 0 or local + 1 == total:
                path = (
                    ROOT
                    / f"checkpoints/task2/tokens_{args.visual_tokens}/update_{global_update}.pt"
                )
                save_checkpoint(
                    path,
                    model,
                    opt,
                    scheduler,
                    scaler,
                    stage=stage,
                    stage_update=local + 1,
                    update=global_update,
                    sampler_position=local + 1,
                    visual_tokens=args.visual_tokens,
                    sampler_rng=rng.bit_generator.state,
                    mixture_counts=dict(mixture),
                    skipped_updates=skipped,
                    total_seconds=prior_total + time.perf_counter() - total_time,
                    stage_seconds=prior_stage + time.perf_counter() - ts,
                    peak_gpu_bytes=max(prior_peak, torch.cuda.max_memory_allocated()),
                )
            if global_update % 500 == 0 or local + 1 == total:
                development(model, dev, global_update, out)
            if global_update % 25 == 0:
                print("VLM", stage + 1, global_update, total_loss, flush=True)
        # Compact weights at every stage endpoint, including the connector calibration buffer.
        state = {
            n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
        }
        state["scale_factor"] = model.scale_factor.detach().cpu()
        torch.save(state, out / f"stage_{stage+1}_weights.pt")
        append_json(
            out / "resources.jsonl",
            dict(
                stage=stage + 1,
                seconds=prior_stage + time.perf_counter() - ts,
                trainable_parameters=params,
                peak_gpu_bytes=max(prior_peak, torch.cuda.max_memory_allocated()),
                presentations=total * 16,
                mixture=dict(mixture),
                skipped_updates=skipped,
            ),
        )
    write_json(
        out / "completed.json",
        dict(
            updates=4500,
            presentations=72000,
            seconds=prior_total + time.perf_counter() - total_time,
        ),
    )


@torch.no_grad()
def evaluate(args):
    require_cuda()
    seed_all()
    model = VLM(args.visual_tokens)
    model.enable_adapters()
    weights = ROOT / f"evidence/task2/tokens_{args.visual_tokens}/stage_3_weights.pt"
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=False
    )
    model.eval()
    model.lm.config.use_cache = True
    test = with_partner(
        json.loads((ROOT / "data/prepared/task2/test.json").read_text())
    )
    out = ROOT / f"evidence/task2/tokens_{args.visual_tokens}/final"
    out.mkdir(parents=True, exist_ok=True)
    predictions = []
    for suite in ["ground", "natural"]:
        rows = [r for r in test if r["id"].startswith(suite)]
        for condition in ["correct", "swap", "mismatched", "no_image"]:
            likelihood = {
                r["id"]: r["nll"] for r in nll_records(model, rows, condition)
            }
            for j in range(0, len(rows), 8):
                chunk = rows[j : j + 8]
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                ts = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    batch, _ = model.pack(
                        chunk, condition, targets=False, left_pad=True
                    )
                    seq = model.lm.generate(
                        **batch,
                        do_sample=False,
                        max_new_tokens=16,
                        pad_token_id=model.t.pad_token_id,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - ts
                for r, tokens in zip(chunk, seq):
                    ids = tokens.tolist()
                    answer = model.t.decode(ids, skip_special_tokens=True).strip()
                    score = (
                        vqa_score(answer, r["answers"])
                        if suite == "natural"
                        else float(vqa_normalize(answer) == r["canonical"])
                    )
                    partner = next(x for x in rows if x["id"] == r["partner"])
                    predictions.append(
                        dict(
                            **r,
                            suite=suite,
                            condition=condition,
                            prediction=answer,
                            score=score,
                            canonical_correct=vqa_normalize(answer) == r["canonical"],
                            nll=likelihood[r["id"]],
                            empty=not bool(answer),
                            follows_partner=vqa_normalize(answer)
                            == partner["canonical"],
                            generation_seconds=elapsed / len(chunk),
                            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                            output_tokens=len(ids),
                            truncated=len(ids) == 16
                            and ids[-1] != model.t.eos_token_id,
                        )
                    )
    write_json(out / "predictions.json", predictions)
    summarize(predictions, out)
    write_json(
        out / "generation_resources.json",
        dict(
            total_seconds=sum(r["generation_seconds"] for r in predictions),
            peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in predictions),
            answers=len(predictions),
            batch_size=8,
        ),
    )


def summarize(predictions, out):
    import pandas as pd

    df = pd.DataFrame(predictions)
    summaries = []
    pair_rows = []
    for (suite, condition), group in df.groupby(["suite", "condition"]):
        pairs = []
        for pair_id, p in group.groupby("pair_id", sort=True):
            if len(p) != 2:
                raise ValueError("Incomplete evaluation pair")
            pair = dict(
                pair_id=pair_id,
                suite=suite,
                condition=condition,
                accuracy=p.score.mean(),
                nll=p.nll.mean(),
                pair_accuracy=float(p.canonical_correct.all()),
                identical=float(len({vqa_normalize(x) for x in p.prediction}) == 1),
                empty=p["empty"].mean(),
                seconds=p.generation_seconds.mean(),
            )
            pairs.append(pair)
            pair_rows.append(pair)
        summary = dict(
            suite=suite,
            condition=condition,
            **{
                k: float(np.mean([p[k] for p in pairs]))
                for k in [
                    "accuracy",
                    "nll",
                    "pair_accuracy",
                    "identical",
                    "empty",
                    "seconds",
                ]
            },
        )
        for metric in ["accuracy", "nll", "pair_accuracy", "identical", "empty"]:
            summary[metric + "_ci"] = pair_bootstrap([p[metric] for p in pairs])
        summary["by_type"] = {}
        for kind, sub in group.groupby("kind"):
            pair_means = sub.groupby("pair_id")[["score", "nll"]].mean()
            summary["by_type"][kind] = dict(
                accuracy=float(sub.score.mean()),
                nll=float(sub.nll.mean()),
                accuracy_ci=pair_bootstrap(pair_means.score),
                nll_ci=pair_bootstrap(pair_means.nll),
                pairs=len(pair_means),
            )
        summary["follows_partner_rate"] = float(group.follows_partner.mean())
        summaries.append(summary)
    pairdf = pd.DataFrame(pair_rows)
    contrasts = []
    for suite in ["ground", "natural"]:
        a = (
            pairdf[(pairdf.suite == suite) & (pairdf.condition == "correct")]
            .set_index("pair_id")
            .sort_index()
        )
        b = (
            pairdf[(pairdf.suite == suite) & (pairdf.condition == "swap")]
            .set_index("pair_id")
            .sort_index()
        )
        for metric, values in [
            ("accuracy_advantage", a.accuracy - b.accuracy),
            ("swap_nll_penalty", b.nll - a.nll),
        ]:
            contrasts.append(
                dict(
                    suite=suite,
                    metric=metric,
                    mean=float(values.mean()),
                    ci=pair_bootstrap(values),
                )
            )
    write_json(out / "summary.json", summaries)
    write_json(out / "paired_contrasts.json", contrasts)
    # Fixed rule: first three pair IDs per suite, selected independently of predictions.
    chosen = {
        suite: sorted(df[df.suite == suite].pair_id.unique())[:3]
        for suite in ["ground", "natural"]
    }
    examples = [r for r in predictions if r["pair_id"] in chosen[r["suite"]]]
    write_json(
        out / "qualitative_examples.json",
        dict(
            selection="lexicographically first three pair IDs per suite",
            records=examples,
        ),
    )
