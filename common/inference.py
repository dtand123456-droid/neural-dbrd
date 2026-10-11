"""FP32 ranking with reusable CUDA graphs and bounded cache writes."""
import types
import weakref
import numpy as np
import torch
from .model import KEYS, BASE_KEYS, pair_masses
from .tokens import tokenize_reports, collate_tokens
from .ranking import report_kernels
from .runtime import check as require_work_allowed


class RankedFeatureCache(dict):
    pass

def fixed_safe_mask(mask):
    safe = mask.clone()
    safe[:, 0] = safe[:, 0] | ~safe.any(-1)
    return safe

def fast_mass(self, unit_projected, condition, mask):
    x = unit_projected.float()
    c = condition.float()[:, None, :]
    joint = torch.tanh(x + c + x * c / self.interaction_scale)
    logits = self.logit_scale * torch.tanh(self.allocation(joint).squeeze(-1))
    valid = mask.bool()
    safe = fixed_safe_mask(valid)
    mass = logits.masked_fill(~safe, -torch.inf).softmax(-1)
    return mass * valid

def fast_evidence_score(self, query, documents):
    n = len(documents['dense'])
    qm = query['unit_mask'][None].expand(n, -1)
    dm = documents['unit_mask']
    qa, da = pair_masses(self, query, documents)
    similarity = torch.einsum('ih,njh->nij', query['units'].float(), documents['units'].float())
    safeq = fixed_safe_mask(qm)
    safed = fixed_safe_mask(dm)
    forward = (similarity / self.temperature + da.clamp_min(1e-30).log()[:, None, :]).masked_fill(~safed[:, None, :], -torch.inf).softmax(-1)
    reverse = (similarity / self.temperature + qa.clamp_min(1e-30).log()[:, :, None]).masked_fill(~safeq[:, :, None], -torch.inf).softmax(-2)
    value = 0.5 * ((qa * (forward * similarity).sum(-1)).sum(-1) + (da * (reverse * similarity).sum(-2)).sum(-1))
    return (value, qm.any(-1) & dm.any(-1))

class GraphScorer:

    def __init__(self, model, cache, query, keys):
        batch_size = model.config['coarse_batch_size']
        blocks = model.config['graph_blocks']
        self.block_size = batch_size
        self.batch_size = batch_size * blocks
        batch_size = self.batch_size
        self.query = {k: query[k].clone() for k in keys}
        self.indices = torch.zeros(batch_size, dtype=torch.long, device=cache['dense'].device)
        self.kernels = torch.ones(batch_size, len(model.config['time_scales']) + 1, device=cache['dense'].device)
        (self.model, self.cache, self.keys) = (model, weakref.proxy(cache), keys)
        self.graph = torch.cuda.CUDAGraph()
        evidence = model.evidence_module
        (old_mass, old_score) = (evidence.mass, evidence.score)
        evidence.mass = types.MethodType(fast_mass, evidence)
        evidence.score = types.MethodType(fast_evidence_score, evidence)

        def score_blocks():
            parts = []
            for start in range(0, self.batch_size, self.block_size):
                end = start + self.block_size
                docs = {k: cache[k][self.indices[start:end]] for k in keys}
                parts.append(model.score(self.query, docs, self.kernels[start:end]))
            return torch.cat(parts)
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(model.config['graph_warmup_steps']):
                    self.output = score_blocks()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = score_blocks()
        finally:
            (evidence.mass, evidence.score) = (old_mass, old_score)

    @torch.inference_mode()
    def __call__(self, query, indices, kernels, check=lambda : None):
        for k in self.keys:
            self.query[k].copy_(query[k])
        device = self.cache['dense'].device
        indices_gpu = torch.as_tensor(indices, device=device)
        kernels_gpu = torch.as_tensor(kernels, device=device)
        result = torch.empty(len(indices), dtype=torch.float32, device=device)
        size = self.batch_size
        for start in range(0, len(indices), size):
            check()
            if len(indices) - start >= size:
                self.indices.copy_(indices_gpu[start:start + size])
                self.kernels.copy_(kernels_gpu[start:start + size])
                self.graph.replay()
                result[start:start + size].copy_(self.output)
            else:
                for tail in range(start, len(indices), self.block_size):
                    end = tail + self.block_size
                    selected = indices_gpu[tail:end]
                    result[tail:end] = self.model.score(query, {k: self.cache[k][selected] for k in self.keys}, kernels_gpu[tail:end])
        values = result.cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite score')
        return values

@torch.inference_mode()
def coarse_scores(model, query, cache, indices, kernels, keys):
    assert not model.training
    if not len(indices):
        return np.empty(0, dtype=np.float32)
    if not hasattr(cache, 'graph_scorer'):
        cache.graph_scorer = GraphScorer(model, cache, query, keys)
    return cache.graph_scorer(query, indices, kernels, require_work_allowed)

@torch.inference_mode()
def encode_cache(model, tokens, pad, device, batch_size):
    max_length = model.config['max_length']
    projection_width = model.config['profile_projection_width']
    model.eval()
    cache = {}
    order = sorted(range(len(tokens)), key=lambda i: len(tokens[i][0]))
    for start in range(0, len(tokens), batch_size):
        require_work_allowed()
        selected = order[start:start + batch_size]
        with torch.autocast('cuda', dtype=torch.float16):
            features = model(**collate_tokens(tokens, selected, pad, device))
        if not cache:
            cache = {k: torch.empty((len(tokens), *features[k].shape[1:]), dtype=features[k].dtype, device=device) for k in BASE_KEYS}
            if model.c3_enabled:
                cache['token_vectors'] = torch.zeros((len(tokens), max_length, projection_width), dtype=torch.float32, device=device)
                cache['token_units'] = torch.zeros((len(tokens), max_length), dtype=torch.long, device=device)
        for k in BASE_KEYS:
            cache[k][selected] = features[k]
        if model.c3_enabled:
            length = features['token_vectors'].shape[1]
            indices = torch.as_tensor(selected, dtype=torch.long, device=device)
            vectors = torch.nn.functional.pad(features['token_vectors'], (0, 0, 0, max_length - length))
            units = torch.nn.functional.pad(features['token_units'], (0, max_length - length))
            cache['token_vectors'].index_copy_(0, indices, vectors)
            cache['token_units'].index_copy_(0, indices, units)
    return RankedFeatureCache(cache)

@torch.inference_mode()
def rank_model(model, cache, pool, qid, window=None, query_features=None):
    fine_batch = model.config['fine_batch_size']
    require_work_allowed()
    legal = pool.candidates(qid, window)
    device = cache['dense'].device
    query = {k: cache[k][pool.index[qid]] for k in model.feature_keys} if query_features is None else query_features
    scores = np.zeros(len(pool.reports), dtype=np.float32)
    kernels = report_kernels(pool, qid, legal, model.config['time_scales'])
    scores[legal] = coarse_scores(model, query, cache, legal, kernels, BASE_KEYS)
    if not model.c3_enabled:
        return pool.rank(qid, scores, window)
    shortlist = legal[np.lexsort((pool.ids[legal], -scores[legal]))[:model.config['rerank_topk']]]
    for start in range(0, len(shortlist), fine_batch):
        require_work_allowed()
        ix = shortlist[start:start + fine_batch]
        docs = {k: cache[k][ix] for k in model.feature_keys}
        (coarse, joint) = model.score_joint(query, docs, torch.as_tensor(report_kernels(pool, qid, ix, model.config['time_scales']), device=device))
        assert bool(torch.isfinite(joint).all())
        scores[ix] += (joint - coarse).cpu().numpy()
    return pool.rank(qid, scores, window)
