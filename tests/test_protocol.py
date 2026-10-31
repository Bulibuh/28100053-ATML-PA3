"""Offline numerical and protocol regression tests; no model/data downloads."""
from types import SimpleNamespace
import copy
import json
import random
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from atml_pa3.common import (causal_loss, collate, supervised_tokens, optimizer_groups,
                            save_checkpoint, load_checkpoint, schedule, seed_all, amp_update, ROOT)
from atml_pa3.adapters import LowRank, singular_bases, attach, merge, export_factors
from atml_pa3.task1 import normalized
from atml_pa3.task1_eval import load_qualitative_prompts, qualitative_messages, traced_hflm_class


def tiny():
    seed_all()
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, tie_word_embeddings=True))


def test_token_normalized_accumulation_matches_whole_batch():
    torch.manual_seed(4)
    logits = torch.randn(4, 7, 11, requires_grad=True)
    labels = torch.randint(0, 11, (4, 7))
    labels[0, :4] = -100
    labels[1, :2] = -100
    labels[2, :5] = -100
    labels[3, 5:] = -100
    total, count = causal_loss(logits, labels)
    dense = F.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(total / count, dense)
    (total / count).backward()
    expected = logits.grad.clone()
    logits.grad.zero_()
    # Unequal valid token counts; averaging microbatch mean losses would be wrong.
    for start in (0, 2):
        loss, _ = causal_loss(logits[start:start+2], labels[start:start+2])
        (loss / count).backward()
    torch.testing.assert_close(logits.grad, expected)
    assert not logits.grad[0, :3].any()


class TokenizerStub:
    eos_token_id = 2
    unk_token_id = 99
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['add_generation_prompt'] is True
        return [10, 11, 12, 13]  # includes non-thinking empty-block prefix
    def convert_tokens_to_ids(self, s):
        assert s == '<|im_end|>'
        return 2
    def encode(self, s, **kwargs):
        assert kwargs['add_special_tokens'] is False
        return [20, 21, 22]


def test_answer_mask_and_end_of_turn_no_truncation():
    t = TokenizerStub()
    row = supervised_tokens(t, 'question', 'answer', 8)
    assert row == dict(input_ids=[10,11,12,13,20,21,22,2], labels=[-100]*4+[20,21,22,2])
    assert supervised_tokens(t, 'question', 'answer', 7) is None
    batch = collate([row, dict(input_ids=[10, 2], labels=[-100, 2])], 2)
    assert batch['attention_mask'][1].tolist() == [1,1,0,0,0,0,0,0]
    assert batch['labels'][1, 2:].eq(-100).all()


def test_oplora_two_sided_projection_and_export(tmp_path):
    torch.manual_seed(2)
    base = nn.Linear(12, 10, bias=False)
    u, v = singular_bases(base.weight, 3)
    m = LowRank(base, 4, 8, dropout=0.0, bases=(u, v))
    with torch.no_grad():
        m.B.normal_()
    b, a = m.factors()
    torch.testing.assert_close(u.T @ b, torch.zeros(3,4), atol=2e-6, rtol=0)
    torch.testing.assert_close(a @ v, torch.zeros(4,3), atol=2e-6, rtol=0)
    delta = m.delta()
    assert float(((u.T @ delta).square().sum() / delta.square().sum()).detach()) < 1e-10
    x = torch.randn(3,12)
    m(x).sum().backward()
    assert m.A.grad is not None and m.B.grad is not None
    assert base.weight.grad is None
    model = nn.Module(); model.proj = m
    export_factors(model, tmp_path/'f.pt')
    f = torch.load(tmp_path/'f.pt', weights_only=True)['proj']
    torch.testing.assert_close(delta, f['scale'] * f['B'] @ f['A'])
    m.eval(); before = m(x)
    merge(model)
    torch.testing.assert_close(before, model.proj(x), atol=2e-6, rtol=1e-5)


def test_adapter_initialization_and_parameter_count_match():
    def build(k, rank=4):
        seed_all()
        m = nn.Module(); m.q_proj = nn.Linear(12,10)
        attach(m, rank, rank*2, protection=k, targets=['q_proj'])
        return m
    lora, op = build(0), build(3)
    torch.testing.assert_close(lora.q_proj.A, op.q_proj.A)
    torch.testing.assert_close(lora.q_proj.B, op.q_proj.B)
    assert lora.q_proj.B.count_nonzero() == 0
    assert lora.q_proj.A.dtype == torch.float32
    assert lora.q_proj.base.weight.dtype == torch.float16
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 4*(12+10)
    assert sum(p.numel() for p in build(0,16).parameters() if p.requires_grad) == 4*4*(12+10)


def test_l2sp_counts_tied_parameters_once_includes_bias_and_norm():
    model = tiny()
    params = list(model.named_parameters())
    assert sum(p is model.model.embed_tokens.weight for _,p in params) == 1
    ref = {n:p.detach().clone() for n,p in params}
    with torch.no_grad():
        for _, p in params: p.add_(0.001)
    model.zero_grad()
    omega = sum((p-ref[n]).square().sum()/2 for n,p in params)
    (0.001*omega).backward()
    for n,p in params:
        torch.testing.assert_close(p.grad, 0.001*(p-ref[n]), atol=1e-10, rtol=1e-5)
    groups = optimizer_groups(model, 2e-5)
    decay_ids = {id(p) for p in groups[0]['params']}
    assert all((id(p) not in decay_ids) for n,p in params if p.ndim < 2 or n.endswith('bias'))


def test_checkpoint_exact_resume_and_tied_weights(tmp_path):
    m=tiny(); opt=torch.optim.AdamW(optimizer_groups(m, 2e-5), foreach=False)
    sched=schedule(opt, 10, 2); scaler=torch.amp.GradScaler('cpu')
    x=torch.randint(0,64,(2,9))
    def step():
        opt.zero_grad(); loss,_=causal_loss(m(x).logits,x)
        scaler.scale(loss/16).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(),1)
        scaler.step(opt); scaler.update();sched.step()
    step()
    save_checkpoint(tmp_path/'c.pt',m,opt,sched,scaler,update=1)
    rng=(random.random(), np.random.rand(), torch.rand(2))
    step(); expected={n:p.detach().clone() for n,p in m.named_parameters()}
    load_checkpoint(tmp_path/'c.pt',m,opt,sched,scaler)
    assert random.random()==rng[0] and np.random.rand()==rng[1]
    torch.testing.assert_close(torch.rand(2),rng[2])
    step()
    for n,p in m.named_parameters(): torch.testing.assert_close(p,expected[n],rtol=0,atol=0)
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert all(v.dtype==torch.float32 for st in opt.state.values() for k,v in st.items() if k!='step')


def test_amp_overflow_retries_same_batch_without_advancing():
    seed_all(); m=nn.Linear(2,1); optimizer=torch.optim.AdamW(m.parameters(),lr=0.01)
    scaler=torch.amp.GradScaler('cpu', init_scale=8)
    draws=[]; skipped=[]
    def backward():
        draws.append(torch.rand(2))
        loss=m(torch.ones(1,2)).sum()
        if len(draws)==1: loss=loss*torch.tensor(float('inf'))
        scaler.scale(loss).backward()
        return {'sft':float(loss.detach())}
    _, norm, retries=amp_update(optimizer,scaler,m.parameters(),backward,skipped.append)
    assert retries==1 and len(skipped)==1 and np.isfinite(norm)
    torch.testing.assert_close(draws[0],draws[1])
    assert all(st['step']==1 for st in optimizer.state.values())


def test_qualitative_supplied_file_exact_counts_and_systems():
    records, qs=load_qualitative_prompts(ROOT/'configs/first_plot_questions.yaml')
    assert len(records)==24 and len(qs)==8
    assert qs[0]=='what_is_your_wish' and qs[-1]=='enough_of_my_husband'
    assert sum(r['question_id'] in (qs[0],qs[-1]) for r in records)*7==42
    json_rows=[r for r in records if r['format']=='json']
    assert len(json_rows)==8
    assert all(qualitative_messages(r)[0]['role']=='system' for r in json_rows)
    assert all('judge' not in r and 'samples_per_paraphrase' not in r for r in records)


def test_normalization_detects_format_variations():
    assert normalized(' How many APPLES?\n') == normalized('how many apples')
    assert normalized('Ａｐｐｌｅｓ') == normalized('apples')


def test_traced_generation_uses_harness_order_and_stop_rules(monkeypatch):
    from lm_eval.models.huggingface import HFLM
    cls=traced_hflm_class()
    lm=object.__new__(cls)
    lm.trace={}; lm.rows=[]
    tok=SimpleNamespace(pad_token_id=0, eos_token_id=2, decode=lambda ids, **kw: "generated")
    lm.tokenizer=tok
    lm._model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
    # First request reaches its token cap; second ends naturally and is padded.
    monkeypatch.setattr(HFLM, '_model_generate', lambda self, *args, **kw: torch.tensor([[0,3,4,5,6,7],[0,8,9,2,0,0]]))
    lm._model_generate(torch.tensor([[0,3,4],[0,8,9]]),6,['Q:'])
    assert lm.trace[(3,4)]['truncated'] is True
    assert lm.trace[(8,9)]['truncated'] is False
    assert lm.trace[(8,9)]['token_ids']==[2]


@pytest.mark.parametrize('protection',[0,3])
def test_tiny_qwen_checkpointed_adapter_training_and_logits_merge(protection):
    model=tiny()
    attach(model,rank=4,alpha=8,protection=protection)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    opt=torch.optim.AdamW(optimizer_groups(model,2e-4),foreach=False)
    scaler=torch.amp.GradScaler('cpu',init_scale=8)
    x=torch.randint(0,64,(2,10))
    with torch.autocast('cpu',dtype=torch.float16):
        loss,n=causal_loss(model(x).logits,x)
    scaler.scale(loss/n).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1)
    scaler.step(opt);scaler.update()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    model.eval()
    with torch.no_grad():
        before=model(x).logits.float()
        merge(model)
        after=model(x).logits.float()
    torch.testing.assert_close(before,after,rtol=0.02,atol=0.03)
