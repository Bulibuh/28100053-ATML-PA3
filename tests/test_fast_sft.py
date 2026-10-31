"""Numerical comparisons against dense logits and uncached projection autograd."""
import copy
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from test_protocol import tiny
from atml_pa3.common import causal_loss
from atml_pa3.fast_sft import prepare_microbatch, response_loss, projected_effective_batch
from atml_pa3.adapters import LowRank, singular_bases, attach


def rows():
    return [dict(input_ids=[1,2,3,4,5,6,7],labels=[-100,-100,-100,4,5,6,7]),
            dict(input_ids=[2,3,4,5],labels=[-100,-100,4,5])]


def test_response_only_loss_and_full_model_gradients_match_dense_logits():
    model=tiny(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    ref=copy.deepcopy(model)
    batch=prepare_microbatch(rows(),0)
    dense,count=causal_loss(ref(batch['input_ids'],attention_mask=batch['attention_mask']).logits,batch['labels'])
    (dense/count).backward()
    calls=[]
    hook=model.lm_head.register_forward_pre_hook(lambda m,inputs:calls.append(inputs[0].shape[0]))
    fast=response_loss(model,batch,chunk_size=2)
    torch.testing.assert_close(fast,dense,rtol=1e-6,atol=1e-6)
    assert sum(calls)==int(count) and len(calls)==3  # no prompt or pad logits
    (fast/count).backward();hook.remove()
    for (name,p),(name2,q) in zip(model.named_parameters(),ref.named_parameters()):
        assert name==name2
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-6)


@pytest.mark.parametrize('protection',[0,3])
def test_response_only_amp_adapter_gradients_match_dense(protection):
    model=tiny();attach(model,rank=4,alpha=8,protection=protection)
    for m in model.modules():
        if isinstance(m,LowRank):
            m.drop.p=0
            with torch.no_grad():m.B.normal_(std=0.01)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    ref=copy.deepcopy(model); batch=prepare_microbatch(rows(),0)
    with torch.autocast('cpu',dtype=torch.float16):
        dense,count=causal_loss(ref(batch['input_ids'],attention_mask=batch['attention_mask']).logits,batch['labels'])
    (dense/count).backward()
    with projected_effective_batch(model):
        with torch.autocast('cpu',dtype=torch.float16): fast=response_loss(model,batch,chunk_size=2)
        (fast/count).backward()
    torch.testing.assert_close(fast,dense,rtol=2e-3,atol=2e-3)
    for (name,p),(name2,q) in zip(model.named_parameters(),ref.named_parameters()):
        assert name==name2
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=0.03,atol=2e-5)
        else:assert p.grad is None


@pytest.mark.parametrize('autocast',[False,True])
def test_projected_batch_chain_rule_matches_repeated_projections(autocast):
    torch.manual_seed(24)
    linear=nn.Linear(12,10,bias=False)
    model=LowRank(linear,4,8,bases=singular_bases(linear.weight,3))
    with torch.no_grad():model.B.normal_(std=0.1)
    ref=copy.deepcopy(model)
    inputs=[torch.randn(2,7,12),torch.randn(2,9,12)]
    torch.manual_seed(6304)
    for x in inputs:
        with torch.autocast('cpu',dtype=torch.float16,enabled=autocast):
            out=checkpoint(ref,x,use_reentrant=False)
            loss=out.float().square().sum()/100
        loss.backward()
    torch.manual_seed(6304)
    with projected_effective_batch(model):
        for x in inputs:
            with torch.autocast('cpu',dtype=torch.float16,enabled=autocast):
                out=checkpoint(model,x,use_reentrant=False)
                loss=out.float().square().sum()/100
            loss.backward()
    for name in ['A','B']:
        torch.testing.assert_close(getattr(model,name).grad,getattr(ref,name).grad,
                                   rtol=3e-3 if autocast else 1e-5,atol=2e-5 if autocast else 1e-6)
    assert model._batch_factors is None
    with pytest.raises(RuntimeError):
        with projected_effective_batch(model):
            raise RuntimeError('retry fixture')
    assert model._batch_factors is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA execution requires a GPU runtime')
def test_cuda_response_only_amp_and_fused_optimizer_step():
    model=tiny().cuda()
    attach(model,rank=4,alpha=8,protection=3)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    from atml_pa3.common import device_batch, optimizer_groups, amp_update
    batch=device_batch(prepare_microbatch(rows(),0))
    assert all(v.is_cuda for v in batch.values())
    opt=torch.optim.AdamW(optimizer_groups(model,2e-4),fused=True,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=16)
    def backward():
        with projected_effective_batch(model):
            with torch.autocast('cuda',dtype=torch.float16):loss=response_loss(model,batch)/6
            scaler.scale(loss).backward()
        return {'sft':float(loss.detach())}
    values,norm,retries=amp_update(opt,scaler,model.parameters(),backward)
    assert torch.isfinite(torch.tensor(norm)) and values['sft']>0
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    assert all(v.dtype==torch.float32 for st in opt.state.values() for k,v in st.items() if k!='step')


@pytest.mark.parametrize('saved_fused',[False,True])
def test_resume_optimizer_flags_work_with_gradscaler(saved_fused):
    from atml_pa3.common import amp_update
    p=nn.Parameter(torch.ones(2))
    previous=torch.optim.AdamW([p],lr=1e-4,foreach=False,fused=saved_fused)
    (p.square().sum()).backward();previous.step();p.grad=None
    # Resume constructor must not leave fused AMP dispatch enabled when loading
    # an older nonfused group. Loaded groups still select the saved kernel.
    resumed=torch.optim.AdamW([p],lr=1e-4,foreach=False,fused=False)
    resumed.load_state_dict(previous.state_dict())
    scaler=torch.amp.GradScaler('cpu',init_scale=8)
    def backward():
        loss=p.square().sum()
        scaler.scale(loss).backward()
        return {'sft':float(loss.detach())}
    _,norm,retries=amp_update(resumed,scaler,[p],backward)
    assert norm>0 and resumed.state[p]['step']==2
    assert bool(resumed.param_groups[0]['fused'])==saved_fused
