"""Task 3 FP16 live comparison: fixed model, batch size one, shared initial sample."""

import hashlib
import json
import re
import time
from pathlib import Path
import torch
from .common import *
from .answers import boxed, math_equal, answer_groups
from .task3 import weighted_winner

INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVISION = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    require_cuda()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError(
            "--revision must be the immutable 40-character model SHA from supplied README.MD"
        )
    readme = Path(args.readme)
    if args.revision.lower() not in readme.read_text().lower():
        raise ValueError(
            "Requested model revision is not present in supplied README.MD; verify it manually"
        )
    questions = json.loads(
        (ROOT / "data/prepared/task3/live_questions.json").read_text()
    )
    if len(questions) != 50:
        raise ValueError("Expected exactly 50 preselected held-out MATH questions")
    name = "Qwen/Qwen2.5-3B-Instruct"
    if model_info(name, revision=args.revision).sha != args.revision:
        raise ValueError("Model revision did not resolve exactly")
    t = AutoTokenizer.from_pretrained(name, revision=args.revision)
    t.pad_token = t.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            name,
            revision=args.revision,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    out = ROOT / "evidence/task3/live"
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "config.json",
        dict(
            model=name,
            revision=args.revision,
            readme_sha256=sha256(readme),
            hardware=hardware(),
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            batch_size=1,
            precision="fp16",
            quantization=False,
        ),
    )
    generations = []
    strategy_rows = []
    for q in questions:
        messages = [
            dict(role="system", content="You are a helpful assistant."),
            dict(role="user", content=q["problem"] + "\n\n" + INSTRUCTION),
        ]

        def generate(label, chat, cap):
            seed = int.from_bytes(
                hashlib.sha256(f'{SEED}:{q["sample_id"]}:{label}'.encode()).digest()[
                    :4
                ],
                "little",
            )
            seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            ts = time.perf_counter()
            batch = t.apply_chat_template(
                chat,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).to("cuda")
            with torch.no_grad():
                ids = model.generate(
                    **batch,
                    max_new_tokens=cap,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    repetition_penalty=1.0,
                    pad_token_id=t.pad_token_id,
                )[0, batch["input_ids"].shape[1] :].tolist()
            torch.cuda.synchronize()
            text = t.decode(ids, skip_special_tokens=True)
            answer = boxed(text)
            elapsed = time.perf_counter() - ts
            row = dict(
                sample_id=q["sample_id"],
                call=label,
                seed=seed,
                messages=chat,
                response=text,
                extracted_answer=answer,
                correct=math_equal(answer, q["gold_answer"]),
                input_tokens=int(batch["attention_mask"].sum()),
                output_tokens=len(ids),
                retained_output_tokens=len(ids)
                - int(bool(ids) and ids[-1] == t.eos_token_id),
                truncated=len(ids) == cap and ids[-1] != t.eos_token_id,
                seconds=elapsed,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                token_ids=ids,
            )
            generations.append(row)
            return row

        longer = generate("longer", messages, 3072)
        parallel = [generate(f"parallel_{i}", messages, 1024) for i in range(3)]
        revised = generate(
            "revision",
            messages
            + [
                dict(role="assistant", content=parallel[0]["response"]),
                dict(role="user", content=REVISION),
            ],
            2048,
        )
        groups, reps = answer_groups([r["extracted_answer"] for r in parallel], "math")
        winner = weighted_winner(groups)[0]
        selected = next(r for r, g in zip(parallel, groups) if g == winner)
        for strategy, calls, final in [
            ("longer", [longer], longer),
            ("parallel", parallel, selected),
            ("refinement", [parallel[0], revised], revised),
        ]:
            strategy_rows.append(
                dict(
                    sample_id=q["sample_id"],
                    strategy=strategy,
                    answer=final["extracted_answer"],
                    accuracy=float(final["correct"]),
                    input_tokens=sum(r["input_tokens"] for r in calls),
                    output_tokens=sum(r["output_tokens"] for r in calls),
                    seconds=sum(r["seconds"] for r in calls),
                    peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in calls),
                    empty=final["extracted_answer"] is None,
                    extraction_rate=float(final["extracted_answer"] is not None),
                    truncated=final["truncated"],
                    any_call_truncated=any(r["truncated"] for r in calls),
                    initial_correct=parallel[0]["correct"],
                    wrong_to_correct=not parallel[0]["correct"] and final["correct"],
                    correct_to_wrong=parallel[0]["correct"] and not final["correct"],
                )
            )
        write_json(out / "generations.json", generations)
        write_json(out / "per_question.json", strategy_rows)
        print("Live question", q["sample_id"], "complete", flush=True)
    import pandas as pd

    df = pd.DataFrame(strategy_rows)
    df.groupby("strategy").mean(numeric_only=True).to_csv(out / "summary.csv")
    transitions = df[df.strategy == "refinement"]
    write_json(
        out / "revision_transitions.json",
        dict(
            wrong_to_correct=int(transitions.wrong_to_correct.sum()),
            correct_to_wrong=int(transitions.correct_to_wrong.sum()),
            initial_wrong=int((~transitions.initial_correct).sum()),
            initial_correct=int(transitions.initial_correct.sum()),
        ),
    )
    examples = transitions.sort_values("sample_id").head(3).sample_id.tolist()
    write_json(
        out / "qualitative_examples.json",
        dict(
            selection="lexicographically first three fixed live-question IDs",
            records=[r for r in generations if r["sample_id"] in examples],
        ),
    )
