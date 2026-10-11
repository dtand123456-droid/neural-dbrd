"""Stable tensor-state fingerprints."""
import hashlib
import json
import torch


def tensor_state_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError('State must map string names to tensors')
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode() + b'\n'
        digest.update(header)
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
