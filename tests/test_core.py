import copy
import numpy as np
import pytest
import torch
from torch import nn
from atml_pa3.adapters import LowRank, singular_bases, merge
from atml_pa3.answers import (
    boxed,
    math_equal,
    mathvista_normalize,
    vqa_normalize,
    vqa_score,
    answer_groups,
    NO_ANSWER,
)
from atml_pa3.common import (
    causal_loss,
    collate,
    save_checkpoint,
    load_checkpoint,
    schedule,
    seed_all,
)
from atml_pa3.task1 import normalized
from atml_pa3.task3 import (
    scores,
    weighted_winner,
    select,
    simulate,
    question_orders,
    make_splits,
)


def test_oplora_projection_and_gradients():
    torch.manual_seed(6304)
    base = nn.Linear(12, 10, bias=False)
    u, v = singular_bases(base.weight, 3)
    layer = LowRank(base, 4, 8, dropout=0, bases=(u, v))
    with torch.no_grad():
        layer.B.normal_()
    b, a = layer.factors()
    assert torch.allclose(u.T @ b, torch.zeros(3, 4), atol=2e-6)
    assert torch.allclose(a @ v, torch.zeros(4, 3), atol=2e-6)
    x = torch.randn(2, 5, 12)
    y = layer(x)
    y.square().sum().backward()
    assert (
        layer.A.grad is not None
        and layer.B.grad is not None
        and base.weight.grad is None
    )
    assert torch.allclose(
        y, nn.functional.linear(x, base.weight + layer.delta()), atol=2e-6
    )


def test_lora_matched_initialization():
    w = nn.Linear(9, 8, bias=False)
    u, v = singular_bases(w.weight, 2)
    torch.manual_seed(6304)
    a = LowRank(copy.deepcopy(w), 4, 8)
    torch.manual_seed(6304)
    b = LowRank(copy.deepcopy(w), 4, 8, bases=(u, v))
    assert torch.equal(a.A, b.A) and torch.equal(a.B, b.B)
    x = torch.randn(2, 9)
    assert torch.equal(a(x), b(x))


def test_response_loss_weights_and_padding():
    torch.manual_seed(3)
    logits = torch.randn(2, 5, 13, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 2, -100, -100, -100]])
    summed, count = causal_loss(logits, labels)
    assert count == 4
    token = summed / count
    example = causal_loss(logits, labels, True)
    parts = []
    for i in range(2):
        s, n = causal_loss(logits[i : i + 1], labels[i : i + 1])
        parts.append(s / n)
    assert torch.allclose(example, torch.stack(parts).mean())
    assert not torch.allclose(token, example)
    token.backward()
    assert torch.count_nonzero(logits.grad[1, 1:]) == 0


def test_effective_batch_token_normalization():
    logits = torch.randn(4, 6, 7)
    labels = torch.tensor(
        [
            [-100, 1, 2, 3, 4, 5],
            [-100, -100, -100, 2, 3, 4],
            [-100, 1, 2, -100, -100, -100],
            [-100, 2, -100, -100, -100, -100],
        ]
    )
    total, n = causal_loss(logits, labels)
    micro = sum(
        causal_loss(logits[i : i + 2], labels[i : i + 2])[0] / n for i in [0, 2]
    )
    assert torch.allclose(micro, total / n)


def test_l2sp_gradient_is_unaveraged_unique_parameters():
    p = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    ref = torch.zeros_like(p)
    loss = 0.001 * 0.5 * ((p - ref) ** 2).sum()
    loss.backward()
    assert torch.allclose(p.grad, 0.001 * (p - ref))


def test_balanced_last_box():
    assert boxed(r"a \\boxed{1} then \\boxed{\\frac{2}{3}}") == r"\\frac{2}{3}"
    assert boxed(r"\\boxed{1} \\boxed{bad") is None
    assert boxed("no answer") is None


def test_math_equivalence_and_missing_vote():
    assert math_equal("1/2", "0.5")
    groups, reps = answer_groups(["1/2", "0.5", None, ""], "math")
    assert groups[0] == groups[1] and groups[2] == groups[3] and groups[0] != groups[2]
    assert not math_equal(None, None) and not math_equal(NO_ANSWER, NO_ANSWER)


def test_vqa_soft_score_and_normalize():
    assert vqa_normalize("The TWO cats!") == "2 cats"
    assert vqa_score("yes", ["yes"] * 3 + ["no"] * 7) == pytest.approx(0.9)
    assert vqa_score("", [""] * 10) == 0
    assert vqa_normalize("3.14") == "3.14"


def test_mathvista_choice_precision():
    meta = {"question_type": "multi_choice", "choices": ["cat", "dog"]}
    assert mathvista_normalize("(B)", meta) == "B"
    assert mathvista_normalize("dog", meta) == "B"
    assert mathvista_normalize("z", meta) == NO_ANSWER
    assert (
        mathvista_normalize("1.234", {"answer_type": "float", "precision": 2}) == "1.23"
    )


def candidates():
    return [
        dict(
            group=i % 2,
            group_answer=str(i % 2),
            token_count=2,
            logp=[-0.1 - i, -0.2],
            kl_u_p=[float(i), float(i + 1)],
            image_attention=[0.1, 0.4],
            prompt_attention=[0.2, 0.2],
            generated_attention=[0.2, 0.4],
        )
        for i in range(32)
    ]


def test_visconf_prefix_and_covariance():
    pool = candidates()
    prefix = scores(pool[:4])
    s = scores(pool)
    assert not np.allclose(prefix["visconf"], s["visconf"][:4])
    # Explicitly independently derive the tokenwise-vs-summary covariance identity.
    values = np.concatenate([r["kl_u_p"] for r in pool[:4]])
    lo, hi = np.quantile(values, [0.05, 0.95], method="linear")
    r = pool[0]
    c = np.clip((np.array(r["kl_u_p"]) - lo) / max(hi - lo, 1e-8), 0, 1)
    v = np.array(r["image_attention"]) / (
        np.array(r["image_attention"])
        + r["prompt_attention"]
        + r["generated_attention"]
    )
    covariance = np.mean((c - c.mean()) * (v - v.mean()))
    assert prefix["token_fusion"][0] - prefix["visconf"][0] == pytest.approx(covariance)


def test_rwc_ties_n1_and_earliest_support():
    assert weighted_winner([2, 1], [0.5, 0.5])[0] == 2
    assert weighted_winner([3], [99])[0] == 3
    assert select(np.array([1, 2]), "certainty", np.array([1.0, 1.0])) == 1
    _, margin, _ = weighted_winner([1, 1], [1.0, 2.0])
    assert margin == 1


def test_stopping_unique_no_answer_and_future_independence():
    pool = candidates()
    for r in pool[:4]:
        r.update(group=0, group_answer="0")
    n, w = simulate(pool, tau=1)
    assert n == 4 and w == 0
    changed = copy.deepcopy(pool)
    for r in changed[4:]:
        r.update(group=88, group_answer="88", kl_u_p=[1e8, 1e8])
    assert simulate(changed, tau=1, gamma=0, dataset="mathvista") == simulate(
        pool, tau=1, gamma=0, dataset="mathvista"
    )
    for r in pool:
        r.update(group=0, group_answer=NO_ANSWER)
    assert simulate(pool, tau=0.5)[0] == 32


def test_orders_nested_split_and_normalization():
    orders = question_orders("math", "x")
    assert orders == question_orders("math", "x")
    assert all(sorted(o) == list(range(32)) for o in orders)
    splits = make_splits({str(i): None for i in range(1000)})
    assert (
        len(splits["dev"]) == 200
        and len(splits["evaluation"]) == 800
        and len(splits["live"]) == 50
    )
    assert not set(splits["dev"]) & set(splits["evaluation"])
    assert normalized("  ＡＢC\n Hi  ") == normalized("abc hi")


def test_checkpoint_roundtrip_and_rng(tmp_path):
    seed_all()
    model = nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = schedule(opt, 10, 2)
    scaler = torch.amp.GradScaler("cpu")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    save_checkpoint(tmp_path / "x.pt", model, opt, sched, scaler, update=4)
    expected = torch.rand(3)
    with torch.no_grad():
        model.weight.zero_()
    ck = load_checkpoint(tmp_path / "x.pt", model)
    assert (
        ck["update"] == 4
        and torch.equal(model.weight, before["weight"])
        and torch.equal(torch.rand(3), expected)
    )


def test_amp_overflow_retries_same_rng_and_commits_one_update():
    from atml_pa3.common import amp_update

    seed_all()
    p = nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    draws = []
    calls = []
    skipped = []

    def backward():
        draws.append(torch.rand(3))
        calls.append(1)
        loss = p.sum() * (float("inf") if len(calls) == 1 else 1.0)
        scaler.scale(loss).backward()
        return dict(loss=float(loss.detach()))

    _, norm, retries = amp_update(opt, scaler, [p], backward, skipped.append)
    assert retries == 1 and len(skipped) == 1 and torch.equal(draws[0], draws[1])
    assert p.item() == pytest.approx(0.9, abs=1e-6) and norm == pytest.approx(1.0)
