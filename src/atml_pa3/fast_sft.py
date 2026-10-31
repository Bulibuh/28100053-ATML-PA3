"""Response-only Qwen SFT: preserve targets/normalization without prompt logits."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .common import collate, causal_loss
from .adapters import LowRank


def prepare_microbatch(rows, pad):
    batch = collate(rows, pad)
    labels = batch['labels']
    # CPU selection: no CUDA nonzero/boolean-mask synchronization in the hot path.
    positions = labels[:, 1:].ne(-100).nonzero(as_tuple=False)
    if not len(positions):
        raise ValueError('No supervised response tokens')
    batch['response_indices'] = positions[:, 0] * labels.shape[1] + positions[:, 1]
    batch['response_targets'] = labels[:, 1:][positions[:, 0], positions[:, 1]]
    return batch


def response_loss(model, batch, chunk_size=256):
    if not hasattr(model, 'model') or not hasattr(model, 'lm_head'):
        # Tiny generic models used for offline integration tests.
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        return causal_loss(logits, batch['labels'])[0]
    hidden = model.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                         use_cache=False, return_dict=True).last_hidden_state
    selected = hidden.reshape(-1, hidden.shape[-1]).index_select(0, batch['response_indices'])
    target = batch['response_targets']
    loss = selected.new_zeros((), dtype=torch.float32)
    def head_loss(h, y):
        return F.cross_entropy(model.lm_head(h).float(), y, reduction='sum')
    for start in range(0, len(target), chunk_size):
        # Chunk the head to bound temporary allocations. Do not add head
        # recomputation: backbone checkpointing already satisfies the recipe.
        h, y = selected[start:start+chunk_size], target[start:start+chunk_size]
        loss = loss + head_loss(h, y)
    return loss


@contextmanager
def projected_effective_batch(model):
    """Compute OPLoRA projections once, accumulate leaf grads, then apply chain rule.

    Factors and their gradients remain FP32. Cache lifetime is one AMP attempt;
    checkpoint recomputation sees the same factors and retries create fresh graphs.
    """
    active = []
    try:
        for module in model.modules():
            if isinstance(module, LowRank) and module.U is not None:
                original = module.factors()
                leaves = tuple(t.detach().requires_grad_(True) for t in original)
                module._batch_factors = leaves
                active.append((module, original, leaves))
        yield
        outputs, grads = [], []
        for _, original, leaves in active:
            for tensor, leaf in zip(original, leaves):
                if leaf.grad is not None:
                    outputs.append(tensor)
                    grads.append(leaf.grad)
        if outputs:
            torch.autograd.backward(outputs, grads)
    finally:
        for module, _, _ in active:
            module._batch_factors = None
