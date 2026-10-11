"""Report alignment and token-profile retrieval models."""
import math
from numbers import Integral
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

BASE_KEYS = ('dense', 'views', 'present', 'units', 'unit_mask', 'unit_projected', 'global_condition')
KEYS = BASE_KEYS + ('token_vectors', 'token_units')
_INTEGER_TYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _lexical_mask(fields, lengths, batch, length, device):
    if not isinstance(fields, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        raise TypeError('fields and lengths must be tensors')
    if fields.shape != (batch, length) or lengths.shape != (batch,):
        raise ValueError('fields must be [batch,length] and lengths must be [batch]')
    if fields.device != device or lengths.device != device:
        raise ValueError('fields, lengths and hidden states must share a device')
    if fields.dtype not in _INTEGER_TYPES or lengths.dtype not in _INTEGER_TYPES:
        raise TypeError('fields and lengths must be integer tensors')
    if bool(((fields < 0) | (fields > 2)).any()):
        raise ValueError('fields must be 0=special/pad, 1=title or 2=body')
    if bool(((lengths < 1) | (lengths > length)).any()):
        raise ValueError('lengths must describe nonempty right-padded sequences')
    attended = torch.arange(length, device=device)[None] < lengths[:, None]
    if bool(((fields > 0) & ~attended).any()):
        raise ValueError('padding must have zero field labels')
    return (fields > 0)[..., None]

def pool_features(hidden, fields, lengths):
    h = hidden.float()
    mask = (torch.arange(h.shape[1], device=h.device)[None] < lengths[:, None])[..., None]
    dense = F.normalize((h * mask).sum(1) / lengths[:, None].clamp_min(1), dim=-1)
    (views, present) = ([], [])
    for k in (1, 2):
        valid = (fields == k)[..., None]
        count = valid.sum(1)
        views.append(F.normalize((h * valid).sum(1) / count.clamp_min(1), dim=-1))
        present.append(count[:, 0] > 0)
    return {'dense': dense, 'views': torch.stack(views, 1), 'present': torch.stack(present, 1)}

class EncoderRanker(nn.Module):

    def __init__(self, backbone, variant, seed, config):
        super().__init__()
        if variant != 'baseline':
            raise ValueError('Unsupported encoder configuration')
        if not isinstance(seed, Integral) or isinstance(seed, bool) or (not 0 <= seed < 2 ** 32):
            raise ValueError('experiment seed must be an integer in [0, 2**32)')
        public_path = Path(backbone).resolve()
        self.variant = variant
        self.config = dict(config)
        self.max_units = config['max_units']
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(int(seed))
            (self.encoder, loading_info) = AutoModel.from_pretrained(str(public_path), local_files_only=True, output_loading_info=True)
        self.public_loading_info = dict(loading_info)
        for key in ('missing_keys', 'mismatched_keys', 'error_msgs'):
            if self.public_loading_info.get(key):
                raise ValueError(f'public encoder must load completely: {key}={self.public_loading_info[key]}')
        self.public_loading_info.setdefault('unexpected_keys', [])
        if self.encoder.config.num_hidden_layers != 12:
            raise ValueError('MPNet must have 12 encoder blocks')
        self.encoder.requires_grad_(True)
        for parameter in self.encoder.parameters():
            parameter.grad = None
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(int(seed))
            self.adapter = None
            self.logit_mix = nn.Parameter(torch.tensor(config['semantic_mix_init_logit'], dtype=torch.float32))
            self.kernel_logits = nn.Parameter(torch.zeros(len(config['time_scales']) + 1, dtype=torch.float32))
            self.log_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(config['time_prior_initial_alpha'])), dtype=torch.float32))
        self.train(True)

    def _validate_inputs(self, input_ids, attention_mask, fields):
        if not all((isinstance(t, torch.Tensor) for t in (input_ids, attention_mask, fields))):
            raise TypeError('input_ids, attention_mask and fields must be tensors')
        if input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError('input_ids must have nonempty [batch,length] shape')
        if input_ids.shape != attention_mask.shape or input_ids.shape != fields.shape:
            raise ValueError('attention_mask and fields must match input_ids shape')
        if input_ids.device != attention_mask.device or input_ids.device != fields.device or input_ids.device != next(self.encoder.parameters()).device:
            raise ValueError('inputs and encoder must share a device')
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError('input_ids must be int32 or int64')
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('attention_mask must be binary')
        attended = attention_mask.bool()
        lengths = attended.sum(1)
        expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None] < lengths[:, None]
        if not torch.equal(attended, expected):
            raise ValueError('attention_mask must be right padded')
        _lexical_mask(fields, lengths, *input_ids.shape, input_ids.device)
        return lengths

class EvidenceAlignment(nn.Module):

    def __init__(self, hidden, conditioning, config):
        super().__init__()
        self.conditioning = conditioning
        width = config['alignment_width']
        self.interaction_scale = config['alignment_interaction_scale']
        self.logit_scale = config['alignment_logit_scale']
        self.temperature = config['alignment_temperature']
        self.unit_projection = nn.Linear(hidden, width)
        self.condition_projection = nn.Linear(hidden, width)
        self.allocation = nn.Linear(width, 1)
        nn.init.zeros_(self.allocation.weight)
        nn.init.zeros_(self.allocation.bias)

    def mass(self, unit_projected, condition, mask):
        x = unit_projected.float()
        c = condition.float()[:, None, :]
        joint = torch.tanh(x + c + x * c / self.interaction_scale)
        logits = self.logit_scale * torch.tanh(self.allocation(joint).squeeze(-1))
        valid = mask.bool()
        safe = valid.clone()
        safe[~safe.any(-1), 0] = True
        mass = logits.masked_fill(~safe, -torch.inf).softmax(-1)
        return mass * valid

    def score(self, query, documents):
        n = len(documents['dense'])
        qm = query['unit_mask'][None].expand(n, -1)
        dm = documents['unit_mask']
        qa, da = pair_masses(self, query, documents)
        similarity = torch.einsum('ih,njh->nij', query['units'].float(), documents['units'].float())
        safeq = qm.clone()
        safed = dm.clone()
        safeq[~safeq.any(-1), 0] = True
        safed[~safed.any(-1), 0] = True
        forward = (similarity / self.temperature + da.clamp_min(1e-30).log()[:, None, :]).masked_fill(~safed[:, None, :], -torch.inf).softmax(-1)
        reverse = (similarity / self.temperature + qa.clamp_min(1e-30).log()[:, :, None]).masked_fill(~safeq[:, :, None], -torch.inf).softmax(-2)
        value = 0.5 * ((qa * (forward * similarity).sum(-1)).sum(-1) + (da * (reverse * similarity).sum(-2)).sum(-1))
        return (value, qm.any(-1) & dm.any(-1))

class AlignmentRanker(EncoderRanker):

    def __init__(self, backbone, variant, seed, conditioning, config):
        if variant != 'adaptive':
            raise ValueError(variant)
        super().__init__(backbone, 'baseline', seed, config)
        self.alignment_variant = variant
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(seed)
            self.evidence_module = EvidenceAlignment(self.encoder.config.hidden_size, conditioning, config)

    def forward(self, input_ids, attention_mask, fields, unit_ids):
        return self.encode(input_ids, attention_mask, fields, unit_ids)

    def score(self, query, documents, kernels):
        with torch.autocast(device_type=kernels.device.type, enabled=False):
            dense = documents['dense'].float() @ query['dense'].float()
            (local, present) = self.evidence_module.score(query, documents)
            mix = torch.sigmoid(self.logit_mix.float())
            semantic = torch.where(present, (1 - mix) * dense + mix * local, dense)
            prior = F.softplus(self.log_alpha.float()) * (kernels.float() @ self.kernel_logits.float().softmax(-1))
            return semantic + prior

def unit_token_profiles(qtoken, dtoken, qunits, dunits, centers, sigma, max_units):
    raise NotImplementedError('unit_token_profiles is not implemented.')

class LocalProfileReadout(nn.Module):

    def __init__(self, hidden, config):
        super().__init__()
        self.max_units = config['max_units']
        self.temperature = config['alignment_temperature']
        self.correction_scale = config['profile_correction_scale']
        self.projection = nn.Linear(hidden, config['profile_projection_width'], bias=False)
        kernel_count = config['profile_kernel_count']
        readout_width = config['profile_readout_width']
        self.register_buffer('centers', torch.linspace(-1.0, 1.0, kernel_count))
        self.register_buffer('sigma', torch.tensor(config['profile_kernel_sigma']))
        self.readout = nn.Sequential(nn.Linear(kernel_count, readout_width, bias=False), nn.GELU(), nn.Linear(readout_width, 1, bias=False))
        nn.init.zeros_(self.readout[-1].weight)

    def profile(self, query, documents):
        return unit_token_profiles(query['token_vectors'], documents['token_vectors'], query['token_units'], documents['token_units'], self.centers, self.sigma, self.max_units)

    def encode(self, hidden, unit_ids):
        vectors = F.normalize(self.projection(F.layer_norm(hidden.float(), (hidden.shape[-1],))), dim=-1)
        return vectors * (unit_ids > 0)[:, :, None]

    def correction(self, query, documents, evidence_module):
        n = len(documents['dense'])
        qm = query['unit_mask'][None].expand(n, -1)
        dm = documents['unit_mask']
        qa, da = pair_masses(evidence_module, query, documents)
        similarity = torch.einsum('ih,njh->nij', query['units'].float(), documents['units'].float())
        (safeq, safed) = (qm.clone(), dm.clone())
        safeq[~safeq.any(-1), 0] = True
        safed[~safed.any(-1), 0] = True
        forward = similarity / self.temperature + da.clamp_min(1e-30).log()[:, None, :]
        forward = forward.masked_fill(~safed[:, None, :], -torch.inf).softmax(-1)
        reverse = similarity / self.temperature + qa.clamp_min(1e-30).log()[:, :, None]
        reverse = reverse.masked_fill(~safeq[:, :, None], -torch.inf).softmax(-2)
        valid = qm[:, :, None] & dm[:, None, :]
        weight = 0.5 * (qa[:, :, None] * forward + da[:, None, :] * reverse) * valid
        profile = self.profile(query, documents)
        discrepancy = profile * valid[..., None]
        local_hidden = self.readout[1](self.readout[0](discrepancy))
        pooled = (weight[..., None] * local_hidden).sum((1, 2))
        correction = self.correction_scale * torch.tanh(self.readout[-1](pooled).squeeze(-1))
        return correction * (qm.any(-1) & dm.any(-1))



def pair_masses(evidence_module, query, documents):
    raise NotImplementedError('pair_masses is not implemented.')


def unit_token_similarity(qtoken, dtoken, qunits, dunits, max_units):
    similarity = torch.einsum('ih,njh->nij', qtoken.float(), dtoken.float())
    unit = torch.arange(1, max_units + 1, device=similarity.device)
    qmask = qunits[None, :] == unit[:, None]
    dmask = dunits[:, None, :] == unit[None, :, None]
    qw = qmask.float() / qmask.sum(-1, keepdim=True).clamp_min(1)
    dw = dmask.float() / dmask.sum(-1, keepdim=True).clamp_min(1)
    forward = torch.stack([similarity.masked_fill(~dmask[:, j, None, :], -2.0).amax(-1) for j in range(max_units)], -1)
    reverse = torch.stack([similarity.masked_fill(~qmask[j][None, :, None], -2.0).amax(-2) for j in range(max_units)], -1)
    qvalue = torch.einsum('ui,niv->nuv', qw, forward)
    dvalue = torch.einsum('nvj,nju->nvu', dw, reverse).transpose(-1, -2)
    valid = qmask.any(-1)[None, :, None] & dmask.any(-1)[:, None, :]
    return 0.5 * (qvalue + dvalue) * valid


class PooledProfileReadout(LocalProfileReadout):

    def correction(self, query, documents, evidence_module):
        n = len(documents['dense'])
        qm = query['unit_mask'][None].expand(n, -1)
        dm = documents['unit_mask']
        qa = evidence_module.mass(query['unit_projected'][None].expand(n, -1, -1), documents['global_condition'], qm)
        da = evidence_module.mass(documents['unit_projected'], query['global_condition'][None].expand(n, -1), dm)
        similarity = torch.einsum('ih,njh->nij', query['units'].float(), documents['units'].float())
        (safeq, safed) = (qm.clone(), dm.clone())
        safeq[~safeq.any(-1), 0] = True
        safed[~safed.any(-1), 0] = True
        forward = similarity / self.temperature + da.clamp_min(1e-30).log()[:, None, :]
        forward = forward.masked_fill(~safed[:, None, :], -torch.inf).softmax(-1)
        reverse = similarity / self.temperature + qa.clamp_min(1e-30).log()[:, :, None]
        reverse = reverse.masked_fill(~safeq[:, :, None], -torch.inf).softmax(-2)
        valid = qm[:, :, None] & dm[:, None, :]
        weight = 0.5 * (qa[:, :, None] * forward + da[:, None, :] * reverse) * valid
        profile = self.profile(query, documents)
        discrepancy = profile * valid[..., None]
        pooled = (weight[..., None] * discrepancy).sum((1, 2))
        correction = self.correction_scale * torch.tanh(self.readout(pooled).squeeze(-1))
        return correction * (qm.any(-1) & dm.any(-1))


class MeanProfileReadout(LocalProfileReadout):

    def profile(self, query, documents):
        mean = unit_token_similarity(query['token_vectors'], documents['token_vectors'], query['token_units'], documents['token_units'], self.max_units)
        return torch.exp(-0.5 * ((mean[..., None] - self.centers) / self.sigma) ** 2)


class GlobalProfileReadout(LocalProfileReadout):

    def correction(self, query, documents, evidence_module):
        qmask = query['token_units'] > 0
        dmask = documents['token_units'] > 0
        similarity = torch.einsum('ih,njh->nij', query['token_vectors'].float(), documents['token_vectors'].float())
        forward = similarity.masked_fill(~dmask[:, None, :], -2.0).amax(-1)
        reverse = similarity.masked_fill(~qmask[None, :, None], -2.0).amax(-2)
        fkernel = torch.exp(-0.5 * ((forward[..., None] - self.centers) / self.sigma) ** 2)
        rkernel = torch.exp(-0.5 * ((reverse[..., None] - self.centers) / self.sigma) ** 2)
        qprofile = (fkernel * qmask[None, :, None]).sum(1) / qmask.sum().clamp_min(1)
        dprofile = (rkernel * dmask[..., None]).sum(1) / dmask.sum(-1).clamp_min(1)[:, None]
        valid = qmask.any() & dmask.any(-1)
        profile = 0.5 * (qprofile + dprofile) * valid[:, None]
        correction = self.correction_scale * torch.tanh(self.readout(profile).squeeze(-1))
        return correction * valid


MODEL_NAMES = ('S13', 'R7', 'GenericP', 'Base', 'C3Only', 'S12', 'S14', 'S15', 'SelfWeight')

class RetrievalModel(AlignmentRanker):
    def __init__(self, backbone, seed, model_name, config):
        if model_name not in MODEL_NAMES:
            raise ValueError(model_name)
        conditioning = 'self' if model_name in ('SelfWeight', 'Base', 'C3Only') else 'partner'
        super().__init__(backbone, 'adaptive', seed, conditioning, config)
        self.model_name = model_name
        self.c3_enabled = model_name not in ('R7', 'Base')
        self.feature_keys = KEYS if self.c3_enabled else BASE_KEYS
        self.view_mode = 'full_report_dropout' if model_name in ('GenericP', 'Base', 'C3Only') else 'unit_drop'
        if self.c3_enabled:
            cls = {'S12': PooledProfileReadout, 'S14': MeanProfileReadout,
                   'S15': GlobalProfileReadout}.get(model_name, LocalProfileReadout)
            with torch.random.fork_rng(devices=[]), torch.device('cpu'):
                torch.random.default_generator.manual_seed(seed)
                self.c3_module = cls(self.encoder.config.hidden_size, config)

    def encode(self, input_ids, attention_mask, fields, unit_ids):
        lengths = self._validate_inputs(input_ids, attention_mask, fields)
        assert torch.equal(unit_ids > 0, fields > 0)
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            features = pool_features(h, fields, lengths)
            mask = unit_ids[:, None, :] == torch.arange(1, self.max_units + 1, device=h.device)[None, :, None]
            valid = mask.any(-1)
            weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
            raw = weights @ h
            units = F.normalize(raw, dim=-1) * valid[:, :, None]
            projected = self.evidence_module.unit_projection(F.layer_norm(units, (units.shape[-1],)))
            condition = self.evidence_module.condition_projection(F.layer_norm(features['dense'], (units.shape[-1],)))
            features = features | {'units': units, 'unit_mask': valid, 'unit_projected': projected, 'global_condition': condition}
            if self.c3_enabled:
                features.update(token_vectors=self.c3_module.encode(h, unit_ids), token_units=unit_ids)
            return features

    def score_joint(self, query, documents, kernels):
        coarse = self.score(query, documents, kernels)
        if not self.c3_enabled:
            return coarse, coarse
        with torch.autocast(device_type=kernels.device.type, enabled=False):
            return coarse, coarse + self.c3_module.correction(query, documents, self.evidence_module)
