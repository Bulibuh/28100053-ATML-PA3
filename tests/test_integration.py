"""Small synthetic fixtures test execution paths; never used as assignment results."""

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from atml_pa3.common import supervised_tokens, causal_loss
from atml_pa3.task2 import VLM, stage_batches, summarize
from atml_pa3.task1_eval import traced_hflm_class
from atml_pa3.task3 import (
    DEFAULT_SCHEMA,
    load_tables,
    evaluate_fixed,
    question_orders,
    tune,
    stopping_states,
    policy_results,
)
from lm_eval.api.instance import Instance


def tiny():
    words = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>",
        "<pad>",
        "user",
        "assistant",
        "yes",
        "no",
        "What",
        "is",
        "two",
        "plus",
        "three",
        "?",
        "Five",
        ".",
        "Describe",
        "this",
        "image",
        "a",
        "cat",
        "long",
        "question",
    ]
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    t = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="<|im_end|>",
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>"],
    )
    t.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    cfg = Qwen3Config(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=4,
        pad_token_id=5,
    )
    return Qwen3ForCausalLM(cfg), t


def test_tiny_qwen_sft_and_visual_connector():
    model, t = tiny()
    row = supervised_tokens(t, "What is two plus three ?", "Five .")
    assert row["labels"][-1] == t.eos_token_id and row["labels"].count(-100) > 0
    vlm = VLM(lm=model, t=t)
    feature = torch.randn(196, 768)
    vlm.features = lambda r, image_id=None: feature
    rows = [
        dict(
            id="x",
            split="train",
            image_id=1,
            question="Describe this image .",
            target="a cat",
            kind="caption",
        ),
        dict(
            id="y",
            split="train",
            image_id=2,
            question="What is this long question ?",
            target="yes",
            kind="presence",
        ),
    ]
    batch, labels = vlm.pack(rows)
    assert batch["inputs_embeds"].shape[0] == 2
    assert labels.ne(-100).sum().item() == 5
    assert all(labels[i, :196].eq(-100).all() for i in range(2))
    output = vlm(**batch).logits
    loss = causal_loss(output, labels, True)
    loss.backward()
    assert vlm.connector[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in vlm.lm.parameters())
    vlm.zero_grad()
    vlm.enable_adapters()
    batch, labels = vlm.pack(rows)
    causal_loss(vlm(**batch).logits, labels, True).backward()
    assert any(
        p.grad is not None for n, p in vlm.lm.named_parameters() if n.endswith(".B")
    )
    noimg, _ = vlm.pack(rows, condition="no_image", targets=False, left_pad=True)
    assert (
        int(batch["attention_mask"].sum()) - int(noimg["attention_mask"].sum())
        == 2 * 196 + 5
    )
    vlm.eval()
    model.config.use_cache = True
    with torch.no_grad():
        generated = model.generate(
            **noimg, max_new_tokens=3, do_sample=False, pad_token_id=t.pad_token_id
        )
    assert generated.shape[0] == 2 and generated.shape[1] <= 3


def test_harness_generation_trace_joins_after_length_sort():
    model, t = tiny()
    lm = traced_hflm_class()(
        pretrained=model, tokenizer=t, batch_size=2, max_length=64, device="cpu"
    )
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(prompt, dict(until=["</s>"], max_gen_toks=4, do_sample=False)),
            idx=i,
            metadata=("synthetic", i, 1),
        )
        for i, prompt in enumerate(["What ?", "What is two plus three ?"])
    ]
    responses = lm.generate_until(requests, disable_tqdm=True)
    assert len(responses) == 2 and {r["doc_id"] for r in lm.rows} == {0, 1}
    assert all(0 < r["output_tokens"] <= 4 for r in lm.rows)


def test_pair_summarizer_and_paired_intervals(tmp_path):
    records = []
    for suite in ["ground", "natural"]:
        for condition in ["correct", "swap", "mismatched", "no_image"]:
            for pair in range(2):
                for side in range(2):
                    records.append(
                        dict(
                            id=f"{suite}_{pair}_{side}",
                            suite=suite,
                            condition=condition,
                            pair_id=f"{suite}_{pair}",
                            score=1.0 if condition == "correct" else 0.0,
                            nll=1.0 if condition == "correct" else 2.0,
                            canonical_correct=condition == "correct",
                            prediction=str(side),
                            empty=False,
                            generation_seconds=0.1,
                            kind="presence" if suite == "ground" else "yes/no",
                            follows_partner=condition == "swap",
                        )
                    )
    summarize(records, tmp_path)
    contrasts = json.loads((tmp_path / "paired_contrasts.json").read_text())
    assert all(r["mean"] == 1 for r in contrasts)
    assert len(json.loads((tmp_path / "summary.json").read_text())) == 8


def test_pair_sampler_preserves_question_pairs():
    captions = [dict(id=f"c{i}", kind="caption") for i in range(18000)]
    first = next(stage_batches(0, captions, np.random.default_rng(6304)))
    assert len(first) == 16
    rows = captions[:10]
    for kind in ["presence", "counting", "natural"]:
        for i in range(3):
            for side in range(2):
                rows.append(
                    dict(id=f"{kind}_{i}_{side}", pair_id=f"{kind}_{i}", kind=kind)
                )
    for stage in [1, 2]:
        generator = stage_batches(stage, rows, np.random.default_rng(6304))
        for _ in range(20):
            batch = next(generator)
            assert len(batch) == 16
            for j in range(0, 16, 2):
                if batch[j]["kind"] != "caption":
                    assert batch[j]["pair_id"] == batch[j + 1]["pair_id"]


def test_offline_tables_selection_and_policy_smoke(tmp_path):
    # Full-sized schema/cardinality fixture, but selectors exercised on only three synthetic questions.
    generations = []
    probabilities = []
    for q in range(1000):
        for r in range(32):
            key = dict(dataset="MATH", sample_id=str(q), rollout_id=r)
            generations.append(
                dict(
                    **key,
                    token_ids=[1, 2],
                    token_count=2,
                    extracted_answer=str(r % 2),
                    gold_answer="1",
                    problem="synthetic",
                )
            )
            probabilities.append(
                dict(
                    **key, logp=[-0.1 - r / 100, -0.2], kl_u_p=[float(r), float(r + 1)]
                )
            )
    pd.DataFrame(generations).to_parquet(tmp_path / "generations.parquet", index=False)
    pd.DataFrame(probabilities).to_parquet(
        tmp_path / "probability_metrics.parquet", index=False
    )
    pools = load_tables(tmp_path, "math", DEFAULT_SCHEMA, {})
    ids = ["0", "1", "2"]
    orders = {q: question_orders("math", q) for q in ids}
    fixed, signals = evaluate_fixed(pools, ids, "math", orders)
    assert len(fixed) == 3 * 10 * 6 * 7 and len(signals) == 9
    # FirstAns cost never scales with advertised pool budget.
    assert fixed[fixed.method == "first"].token_cost.eq(2).all()
    assert (
        fixed[fixed.N == 1]
        .groupby(["sample_id", "order"])
        .accuracy.nunique()
        .eq(1)
        .all()
    )
    selected, grid = tune(pools, ids, "math", orders)
    assert len(grid) == 27
    for family, policy in selected.items():
        baseline = fixed[(fixed.method == "sc") & (fixed.N == 32)].accuracy.mean()
        assert policy["accuracy"] >= baseline - 0.01 - 1e-12
        cache = {
            (q, i): stopping_states([pools[q][j] for j in order], "math")
            for q in ids
            for i, order in enumerate(orders[q])
        }
        a = policy_results(
            pools,
            ids,
            "math",
            orders,
            policy["tau"],
            policy["gamma"],
            policy["fallback"],
        )
        b = policy_results(
            pools,
            ids,
            "math",
            orders,
            policy["tau"],
            policy["gamma"],
            policy["fallback"],
            cache,
        )
        pd.testing.assert_frame_equal(a, b)
    # Alignment errors fail loudly rather than silently dropping candidates.
    frame = pd.DataFrame(probabilities)
    frame.at[0, "logp"] = [-0.1]
    frame.to_parquet(tmp_path / "probability_metrics.parquet", index=False)
    with pytest.raises(ValueError, match="alignment"):
        load_tables(tmp_path, "math", DEFAULT_SCHEMA, {})


def test_native_qualitative_yaml_preserves_system_and_plain_order(tmp_path):
    import yaml
    from atml_pa3.task1_eval import load_qualitative_prompts, qualitative_messages

    records = []
    for fmt in ["plain", "json", "template"]:
        for i in range(8):
            record = dict(
                id=f"q{i}" + ("" if fmt == "plain" else "_" + fmt),
                paraphrases=[f"exact {fmt} text {i}"],
                judge_prompts={"ignored": "not an execution instruction"},
                samples_per_paraphrase=100,
            )
            if fmt == "json":
                record["system"] = "Return a JSON object."
            records.append(record)
    path = tmp_path / "prompts.yaml"
    path.write_text(yaml.safe_dump(records))
    normalized, questions = load_qualitative_prompts(path)
    assert len(normalized) == 24 and questions == [f"q{i}" for i in range(8)]
    assert qualitative_messages(normalized[8]) == [
        dict(role="system", content="Return a JSON object."),
        dict(role="user", content="exact json text 0"),
    ]
    assert "judge_prompts" not in normalized[8]
