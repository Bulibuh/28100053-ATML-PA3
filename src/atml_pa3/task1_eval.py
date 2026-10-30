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
    load_checkpoint(ROOT / "checkpoints/task1" / method / f"update_{update}.pt", model)
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
                self.trace[tuple(p)] = dict(
                    token_ids=tail,
                    output_tokens=len(tail),
                    truncated=not terminated
                    and len(tail) >= max_length - context.shape[1],
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


def evaluate(args):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    require_cuda()
    seed_all()
    out = ROOT / "evidence/task1/evaluation" / args.condition
    out.mkdir(parents=True, exist_ok=True)
    t = tokenizer()
    model = merged_model(args.condition, t, out)
    # Force non-thinking template for GSM8K; likelihood tasks never apply the chat template.
    t.apply_chat_template = functools.partial(
        t.apply_chat_template, enable_thinking=False
    )
    lm = traced_hflm_class()(
        pretrained=model, tokenizer=t, batch_size=8, max_length=4096
    )
    all_results = {}
    for tasks, chat in [
        (["gsm8k_cot"], True),
        (["hellaswag", "boolq", "arc_easy"], False),
    ]:
        res = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
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
            if len(rows) != expected:
                raise ValueError(
                    f"{task}: expected {expected} examples, received {len(rows)}"
                )
            if task == "gsm8k_cot":
                traces = {r["doc_id"]: r for r in lm.rows}
                for r in rows:
                    r["generation_trace"] = traces[r["doc_id"]]
                    r["extracted_answer"] = r.get("filtered_resps")
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
            precision="fp16",
            batch_size=8,
        ),
    )


def diagnostics(args):
    require_cuda()
    seed_all()
    t = tokenizer()
    probes = json.loads((ROOT / "data/prepared/task1/probe_ids.json").read_text())
    base = new_model(torch.float32).eval()
    layers = [0, 13, 27]
    weights = {}
    bases = {}
    activations = {}
    for layer in layers:
        name = f"model.layers.{layer}.mlp.down_proj"
        module = base.get_submodule(name)
        weights[layer] = module.weight.detach().float().cpu()
        bases[layer] = singular_bases(module.weight, 128)[0].cpu()
        activations[layer] = []
    for probe in probes:
        batch = t(probe["prompt"], return_tensors="pt").to("cuda")
        hooks = []
        for layer in layers:

            def capture(module, inputs, layer=layer):
                activations[layer].append(inputs[0][0, -1].detach().float().cpu())

            hooks.append(
                base.get_submodule(
                    f"model.layers.{layer}.mlp.down_proj"
                ).register_forward_pre_hook(capture)
            )
        with torch.no_grad():
            base(**batch)
        for h in hooks:
            h.remove()
    del base
    torch.cuda.empty_cache()
    out = ROOT / "evidence/task1/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(inputs=activations, weights=weights, bases=bases, probes=probes),
        out / "baseline_activations.pt",
    )
    rows = []
    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        model = checkpoint_model(condition)
        for layer in layers:
            module = model.get_submodule(f"model.layers.{layer}.mlp.down_proj")
            w = module.base.weight if hasattr(module, "base") else module.weight
            # Full-FT checkpoint masters must not be rounded before FP32 subtraction.
            if condition in ("full", "full_mid", "l2sp"):
                method, update = CONDITIONS[condition]
                ck = torch.load(
                    ROOT / "checkpoints/task1" / method / f"update_{update}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                trained = ck["model"][
                    f"model.layers.{layer}.mlp.down_proj.weight"
                ].float()
                del ck
                delta = trained - weights[layer]
            else:
                delta = module.delta().detach().float().cpu()
            u = bases[layer]
            denom = float(delta.square().sum())
            w0 = weights[layer]
            rho = float((u.T @ delta).square().sum()) / denom if denom else None
            eta = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(w0))
            for probe, z in zip(probes, activations[layer]):
                r = float(
                    torch.linalg.vector_norm(delta @ z)
                    / (torch.linalg.vector_norm(w0 @ z) + 1e-8)
                )
                rows.append(
                    dict(
                        condition=condition,
                        layer=layer,
                        rho=rho,
                        eta=eta,
                        dataset=probe["dataset"],
                        id=probe["id"],
                        R=r,
                    )
                )
        del model
        torch.cuda.empty_cache()
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


def qualitative(args):
    require_cuda()
    seed_all()
    records, questions = load_qualitative_prompts(args.prompts)
    t = tokenizer()
    all_rows = []
    for condition in CONDITIONS:
        model = merged_model(
            condition, t, ROOT / "evidence/task1/evaluation" / condition
        )
        for r in records:
            batch = t.apply_chat_template(
                qualitative_messages(r),
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ).to("cuda")
            with torch.no_grad():
                result = model.generate(
                    **batch,
                    do_sample=False,
                    max_new_tokens=512,
                    pad_token_id=t.pad_token_id,
                )
            tokens = result[0, batch["input_ids"].shape[1] :].tolist()
            all_rows.append(
                dict(
                    **r,
                    condition=condition,
                    response=t.decode(tokens, skip_special_tokens=True),
                    output_tokens=len(tokens),
                    truncated=len(tokens) == 512 and tokens[-1] != t.eos_token_id,
                    annotation_required=str(r["question_id"])
                    in (questions[0], questions[-1]),
                )
            )
        del model
        torch.cuda.empty_cache()
    all_rows.sort(
        key=lambda r: (
            questions.index(str(r["question_id"])),
            ["plain", "json", "template"].index(r["format"]),
            list(CONDITIONS).index(r["condition"]),
        )
    )
    out = ROOT / "evidence/task1/qualitative"
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
    annotations.to_csv(out / "annotations_to_complete.csv", index=False)
    write_json(
        out / "provenance.json",
        dict(
            prompt_sha256=sha256(args.prompts),
            responses=len(all_rows),
            annotation_rows=len(annotations),
        ),
    )
