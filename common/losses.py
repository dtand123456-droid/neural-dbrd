"""Group ranking and report-view consistency objectives."""
import torch
from torch.nn import functional as F
from .model import BASE_KEYS as KEYS


def group_loss(scores, positive_mask, temperature):
    assert positive_mask.any(dim=1).all()
    logits = scores.float() / temperature
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)).mean()

def evidence_view(token, rng, view, config):
    if view not in ('A', 'B'):
        raise ValueError('view must be A or B')
    (ids, fields, units) = token
    present = sorted(set(units) - {0})
    assert present
    title = config[f'unit_drop_title_keep_{view.lower()}']
    body = config[f'unit_drop_body_keep_{view.lower()}']
    keep = {u for u in present if rng.random() < (title if u == 1 else body)}
    if not keep:
        keep = {rng.choice(present)}
    selected = [i for (i, u) in enumerate(units) if u == 0 or u in keep]
    return (([ids[i] for i in selected], [fields[i] for i in selected], [units[i] for i in selected]), sorted(keep))

def contrastive_masks(buckets, device):
    n = len(buckets)
    mapping = {b: i for (i, b) in enumerate(sorted(set(buckets)))}
    labels = torch.tensor([mapping[b] for b in buckets] * 2, device=device)
    sources = torch.arange(n, device=device).repeat(2)
    valid = (labels[:, None] != labels[None, :]) | (sources[:, None] == sources[None, :])
    valid.fill_diagonal_(False)
    target = (torch.arange(2 * n, device=device) + n) % (2 * n)
    assert bool(valid[torch.arange(2 * n, device=device), target].all())
    return (valid, target)

def consistency_loss(model, features, buckets, temperature):
    count = len(features['dense'])
    assert count == 2 * len(buckets)
    zeros = torch.zeros(count, len(model.config['time_scales']) + 1, device=features['dense'].device)
    scores = torch.stack([model.score({k: features[k][i] for k in KEYS}, features, zeros) for i in range(count)])
    (valid, target) = contrastive_masks(buckets, scores.device)
    logits = (scores / temperature).masked_fill(~valid, -torch.inf)
    return (F.cross_entropy(logits, target), int((~valid).sum()) - count)
