from __future__ import annotations
import functools
import json
import time
from pathlib import Path
import torch
from .common import *
from .task1 import new_model, tokenizer, construct, model_spec
from .adapters import merge, singular_bases

CONDITIONS = {
    "baseline": None,
    "full_mid": ("full", 625),
    "full": ("full", 1250),
    "l2sp": ("l2sp", 1250),
    "lora4": ("lora4", 1250),
    "lora16": ("lora16", 1250),
    "oplora16": ("oplora16", 1250),
}


def checkpoint_model(condition):
    spec = CONDITIONS[condition]
    if spec is None:
        return new_model().eval()
    method, update = spec
    model, _ = construct(method, training=False)
    ck = load_checkpoint(ARTIFACT_ROOT / "checkpoints/task1" / method / f"update_{update}.pt", model)
    if ck.get("model_spec") != model_spec() or ck.get("method") != method or ck.get("update") != update:
        raise ValueError("Evaluation checkpoint provenance mismatch")
    del ck
    return (model.half() if method in ("full", "l2sp") else model).eval()


def merged_model(condition, t, out):
    model = checkpoint_model(condition)
    if condition not in ("lora4", "lora16", "oplora16"):
        return model
    batch = t("What is 2 + 3?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        before = model(**batch).logits.float().cpu()
    # The wrapper backbone was freshly loaded, then only the saved adapter state applied.
    model = merge(model)
    with torch.no_grad():
        after = model(**batch).logits.float().cpu()
    diff = (before - after).abs()
    metrics = dict(
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        allclose=bool(torch.allclose(before, after, rtol=0.02, atol=0.03)),
        rtol=0.02,
        atol=0.03,
    )
    write_json(out / "merge_check.json", metrics)
    if not metrics["allclose"]:
        raise RuntimeError(f"Merge verification failed: {metrics}")
    return model


def traced_hflm_class():
    from lm_eval.models.huggingface import HFLM

    class TracedHFLM(HFLM):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.trace = {}
            self.rows = []

        def _model_generate(self, context, max_length, stop, **kw):
            result = super()._model_generate(context, max_length, stop, **kw)
            for prompt, seq in zip(context, result):
                p = prompt.tolist()
                while p and p[0] == self.tokenizer.pad_token_id:
                    p.pop(0)
                tail = seq[context.shape[1] :].tolist()
                eos = self.model.generation_config.eos_token_id
                eos = (
                    [eos]
                    if isinstance(eos, int)
                    else (eos or [self.tokenizer.eos_token_id])
                )
                positions = [i for i, v in enumerate(tail) if v in eos]
                terminated = bool(positions)
                if terminated:
                    tail = tail[: positions[0] + 1]
                text = self.tokenizer.decode(tail, skip_special_tokens=False)
                stopped = any(term and term in text for term in stop)
                self.trace[tuple(p)] = dict(
                    token_ids=tail,
                    output_tokens=len(tail),
                    truncated=not terminated and not stopped
                    and len(tail) >= max_length - context.shape[1],
                    stopped_on_eos=terminated,
                    stopped_on_string=stopped,
                )
            return result

        def generate_until(self, requests, **kw):
            responses = super().generate_until(requests, **kw)
            for req, response in zip(requests, responses):
                key = tuple(self.tok_encode(req.args[0]))
                trace = self.trace.get(key)
                if trace is None:
                    raise ValueError(
                        "Generation token trace could not be joined to example"
                    )
                self.rows.append(
                    dict(
                        doc_id=req.doc_id,
                        prompt=req.args[0],
                        response=response,
                        **trace,
                    )
                )
            return responses

    return TracedHFLM


def pack_samples(task, rows, expected, generation_rows):
    # Harness 0.4.9 logs one row PER FILTER, not per example. GSM8K has two.
    grouped = {}
    for row in rows:
        ident = row["doc_id"]
        if ident not in grouped:
            grouped[ident] = {k: v for k, v in row.items()
                              if k not in {"filter", "filtered_resps", "metrics", *row["metrics"]}}
            grouped[ident]["filters"] = {}
        filter_name = row["filter"]
        if filter_name in grouped[ident]["filters"]:
            raise ValueError("Duplicate per-example filter result")
        grouped[ident]["filters"][filter_name] = dict(
            filtered_resps=row["filtered_resps"],
            metrics={key: row[key] for key in row["metrics"]})
    if len(grouped) != expected:
        raise ValueError(f"{task}: expected {expected} distinct examples, got {len(grouped)}")
    if task == "gsm8k_cot":
        traces = {r["doc_id"]: r for r in generation_rows}
        for ident, row in grouped.items():
            row["generation_trace"] = traces[ident]
            row["extracted_answer"] = row["filters"]["flexible-extract"]["filtered_resps"][0]
            row["truncated"] = traces[ident]["truncated"]
    return [grouped[key] for key in sorted(grouped)]


def evaluate(args):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    require_cuda()
    seed_all()
    out = ARTIFACT_ROOT / "evidence/task1/evaluation" / args.condition
    out.mkdir(parents=True, exist_ok=True)
    t = tokenizer()
    model = merged_model(args.condition, t, out)
    # Force non-thinking template for GSM8K; likelihood tasks never apply the chat template.
    t.apply_chat_template = functools.partial(
        t.apply_chat_template, enable_thinking=False
    )
    lm = traced_hflm_class()(
        pretrained=model, tokenizer=t, batch_size=args.batch_size, max_length=4096
    )
    all_results = {}
    completed_counts = {}
    for tasks, chat in [
        (["gsm8k_cot"], True),
        (["hellaswag", "boolq", "arc_easy"], False),
    ]:
        from .taskdefs import task_configs
        res = evaluator.simple_evaluate(
            model=lm,
            tasks=task_configs(tasks),
            num_fewshot=0,
            apply_chat_template=chat,
            gen_kwargs={"max_gen_toks": 1024, "do_sample": False} if chat else None,
            log_samples=True,
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
            limit=None,
        )
        samples = res.pop("samples")
        all_results.update(res["results"])
        for task, rows in samples.items():
            expected = {
                "gsm8k_cot": 1319,
                "hellaswag": 10042,
                "boolq": 3270,
                "arc_easy": 2376,
            }[task]
            rows = pack_samples(task, rows, expected, lm.rows)
            completed_counts[task] = len(rows)
            with (out / f"{task}_samples.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, default=json_default) + "\n")
        write_json(
            out / ("gsm8k_harness.json" if chat else "retention_harness.json"), res
        )
    write_json(out / "scores.json", all_results)
    write_json(
        out / "provenance.json",
        dict(
            condition=args.condition,
            model=model_spec(),
            hardware=hardware(),
            full_splits=True,
            counts=completed_counts,
            seed=SEED,
            task_definition_sha256=__import__("atml_pa3.taskdefs", fromlist=["definition_hash"]).definition_hash(),
            precision="fp16",
            batch_size=args.batch_size,
        ),
    )


def diagnostics(args):
    """FP32 local diagnostics; never subtract rounded full-FT inference weights."""
    require_cuda()
    seed_all()
    t = tokenizer()
    probes = json.loads((ARTIFACT_ROOT / "data/prepared/task1/probe_ids.json").read_text())
    out = ARTIFACT_ROOT / "evidence/task1/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "baseline_activations.pt"
    layers = [0, 13, 27]
    probe_hash = sha256(ARTIFACT_ROOT / "data/prepared/task1/probe_ids.json")
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["probe_sha256"] != probe_hash or saved["model"] != model_spec():
            raise ValueError("Baseline activation cache has stale provenance")
        weights, bases, activations = saved["weights"], saved["bases"], saved["inputs"]
    else:
        base = new_model(torch.float32).eval()
        weights, bases, activations = {}, {}, {}
        for layer in layers:
            module = base.get_submodule(f"model.layers.{layer}.mlp.down_proj")
            weights[layer] = module.weight.detach().float().cpu()
            # Share the exact original FP32 bases used by OPLoRA training.
            file = ARTIFACT_ROOT / "data/cache/op_basis" / model_spec()["revision"] / f"model.layers.{layer}.mlp.down_proj.pt"
            if file.exists():
                u, v = torch.load(file, map_location="cpu", weights_only=True)
            else:
                u, v = singular_bases(module.weight, 128)
                file.parent.mkdir(parents=True, exist_ok=True)
                torch.save((u.cpu(), v.cpu()), file)
            bases[layer] = u.cpu()
            activations[layer] = []
        for probe in probes:
            batch = t(probe["prompt"], return_tensors="pt", add_special_tokens=False).to("cuda")
            hooks = []
            for layer in layers:
                def capture(module, inputs, layer=layer):
                    # One unpadded prompt per forward: -1 is its final non-padding token.
                    activations[layer].append(inputs[0][0, -1].detach().float().cpu())
                hooks.append(base.get_submodule(f"model.layers.{layer}.mlp.down_proj").register_forward_pre_hook(capture))
            try:
                with torch.no_grad():
                    base(**batch)
            finally:
                for hook in hooks:
                    hook.remove()
        del base
        torch.cuda.empty_cache()
        torch.save(dict(inputs=activations, weights=weights, bases=bases, probes=probes,
                        probe_sha256=probe_hash, model=model_spec()), cache)
    rows = []
    for condition, spec in CONDITIONS.items():
        if spec is None:
            continue
        method, update = spec
        if method in ("full", "l2sp"):
            checkpoint = torch.load(ARTIFACT_ROOT / "checkpoints/task1" / method / f"update_{update}.pt",
                                    map_location="cpu", weights_only=False, mmap=True)
            if checkpoint["model_spec"] != model_spec():
                raise ValueError("Diagnostic checkpoint revision mismatch")
            factors = None
        else:
            factors = torch.load(ARTIFACT_ROOT / "evidence/task1" / method / "projected_factors.pt",
                                 map_location="cpu", weights_only=True)
        for layer in layers:
            name = f"model.layers.{layer}.mlp.down_proj"
            w0, u = weights[layer], bases[layer]
            if factors is None:
                delta = checkpoint["model"][name + ".weight"].float() - w0
            else:
                f = factors[name]
                delta = f["scale"] * (f["B"].float() @ f["A"].float())
            denom = float(delta.square().sum())
            rho = float((u.T @ delta).square().sum()) / denom if denom else None
            eta = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(w0))
            for probe, z in zip(probes, activations[layer]):
                r = float(torch.linalg.vector_norm(delta @ z) / (torch.linalg.vector_norm(w0 @ z) + 1e-8))
                rows.append(dict(condition=condition, layer=layer, rho=rho, eta=eta,
                                 dataset=probe["dataset"], id=probe["id"], R=r))
        if factors is None:
            del checkpoint
    write_json(out / "metrics.json", rows)


def load_qualitative_prompts(path):
    raw = yaml.safe_load(Path(path).read_text())
    records = raw.get("prompts", raw) if isinstance(raw, dict) else raw
    if not isinstance(records, list) or len(records) != 24:
        raise ValueError("Expected the supplied 24-entry qualitative YAML")
    if "paraphrases" in records[0]:
        converted = []
        for item in records:
            ident = item["id"]
            fmt = (
                "json"
                if ident.endswith("_json")
                else ("template" if ident.endswith("_template") else "plain")
            )
            question = ident.removesuffix("_json").removesuffix("_template")
            if len(item["paraphrases"]) != 1:
                raise ValueError("Expected one supplied paraphrase per entry")
            # The assignment uses one greedy response; upstream judge instructions/100-sample settings are not executed.
            converted.append(
                dict(
                    question_id=question,
                    source_id=ident,
                    format=fmt,
                    prompt=item["paraphrases"][0],
                    system=item.get("system"),
                )
            )
        records = converted
    questions = [str(r["question_id"]) for r in records if r["format"] == "plain"]
    if len(questions) != 8 or len(set(questions)) != 8:
        raise ValueError("Expected eight plain questions")
    for q in questions:
        if {r["format"] for r in records if str(r["question_id"]) == q} != {
            "plain",
            "json",
            "template",
        }:
            raise ValueError("Expected plain/json/template for each question")
    return records, questions


def qualitative_messages(record):
    messages = []
    if record.get("system"):
        messages.append(dict(role="system", content=record["system"]))
    return messages + [dict(role="user", content=record["prompt"])]


@torch.no_grad()
def generate_qualitative_batch(model, t, records):
    # Left padding is required for batched decoder-only generation.
    prompts = [t.apply_chat_template(
        qualitative_messages(r), tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    ) for r in records]
    old_side = t.padding_side
    try:
        t.padding_side = "left"
        batch = t(prompts, padding=True, add_special_tokens=False,
                  return_token_type_ids=False, return_tensors="pt").to(model.device)
    finally:
        t.padding_side = old_side
    result = model.generate(**batch, do_sample=False, max_new_tokens=512,
                            pad_token_id=t.pad_token_id)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [t.eos_token_id])
    rows = []
    for seq in result:
        tokens = seq[batch["input_ids"].shape[1]:].tolist()
        first_eos = next((i for i, token in enumerate(tokens) if token in eos), None)
        if first_eos is not None:
            tokens = tokens[:first_eos + 1]
        rows.append(dict(response=t.decode(tokens, skip_special_tokens=True),
                         output_tokens=len(tokens),
                         truncated=first_eos is None and len(tokens) >= 512))
    return rows


def qualitative(args):
    if getattr(args, "batch_size", 1) < 1:
        raise ValueError("Batch size must be positive")
    require_cuda()
    seed_all()
    records, questions = load_qualitative_prompts(args.prompts)
    t = tokenizer()
    all_rows = []
    out = ARTIFACT_ROOT / "evidence/task1/qualitative"
    out.mkdir(parents=True, exist_ok=True)
    selected = [args.condition] if args.condition else list(CONDITIONS)
    for condition in selected:
        saved_path = out / f"{condition}.json"
        if saved_path.exists():
            saved = json.loads(saved_path.read_text())
            if saved.get("prompt_sha256") != sha256(args.prompts) or saved.get("model") != model_spec() or len(saved["rows"]) != 24:
                raise ValueError("Existing qualitative condition has different provenance; archive it before regeneration")
            print(f"Reusing 24 saved responses for {condition}")
            continue
        all_rows = []
        model = merged_model(
            condition, t, ARTIFACT_ROOT / "evidence/task1/evaluation" / condition
        )
        from tqdm.auto import tqdm
        batch_size = getattr(args, "batch_size", 1)
        for start in tqdm(range(0, len(records), batch_size), desc=f"Qualitative {condition}"):
            chunk = records[start:start + batch_size]
            outputs = generate_qualitative_batch(model, t, chunk)
            for r, generated in zip(chunk, outputs):
                all_rows.append(dict(
                    **r, **generated, condition=condition,
                    annotation_required=str(r["question_id"]) in (questions[0], questions[-1]),
                ))
        del model
        torch.cuda.empty_cache()
        write_json(out / f"{condition}.json", dict(prompt_sha256=sha256(args.prompts), model=model_spec(), batch_size=getattr(args, "batch_size", 1), rows=all_rows))
    if not all((out / f"{c}.json").exists() for c in CONDITIONS):
        print("Saved selected conditions; generate remaining conditions to assemble all 168 responses.")
        return
    all_rows = []
    for c in CONDITIONS:
        saved = json.loads((out / f"{c}.json").read_text())
        if saved["prompt_sha256"] != sha256(args.prompts):
            raise ValueError("Qualitative prompt hash differs between conditions")
        if len(saved["rows"]) != 24:
            raise ValueError("Expected 24 responses for every condition")
        all_rows.extend(saved["rows"])
    all_rows.sort(
        key=lambda r: (
            questions.index(str(r["question_id"])),
            ["plain", "json", "template"].index(r["format"]),
            list(CONDITIONS).index(r["condition"]),
        )
    )
    out = ARTIFACT_ROOT / "evidence/task1/qualitative"
    write_json(out / "responses.json", all_rows)
    import pandas as pd

    annotations = pd.DataFrame([r for r in all_rows if r["annotation_required"]])
    for col in [
        "format_compliance",
        "coherence_relevance",
        "repetition",
        "math_intrusion",
        "inappropriate_content",
        "excerpt",
        "human_notes",
    ]:
        annotations[col] = ""
    if len(annotations) != 42:
        raise ValueError("Expected exactly 42 human annotation rows")
    target = out / "annotations_to_complete.csv"
    if not target.exists():
        annotations.to_csv(target, index=False)
    else:
        print("Preserved existing human annotations; verify them if regenerating responses.")
    write_json(
        out / "provenance.json",
        dict(
            prompt_sha256=sha256(args.prompts),
            responses=len(all_rows),
            annotation_rows=len(annotations),
        ),
    )
