"""JSON records, deterministic execution, and interruption handling."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
from .configuration import validate_parameters, validate_design

SIGNAL = None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(
        [*root.glob('common/*.py'), *root.glob('common/*.json'),
         *root.glob('E*/*.py')])}


def on_signal(signum, frame):
    global SIGNAL
    SIGNAL = signum


def check():
    if SIGNAL is not None:
        raise InterruptedError(f'Interrupted by signal {SIGNAL}')


def setup(seed):
    import numpy as np
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
        raise RuntimeError('Each worker must see exactly one explicitly assigned CUDA device')
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4)


def settings(experiment, config_path):
    root = Path(__file__).resolve().parents[1]
    path = Path(config_path).resolve()
    if root == path or root in path.parents:
        raise ValueError('Supply the parameter file outside the source directory')
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict) or set(document) != {'parameters', 'experiments'}:
        raise ValueError('Configuration requires parameters and experiments objects')
    parameters, designs = document['parameters'], document['experiments']
    validate_parameters(parameters)
    if not isinstance(designs, dict) or experiment not in designs:
        raise ValueError(f'Missing experiment design: {experiment}')
    for name, design in designs.items():
        validate_design(name, design)
    design = designs[experiment]
    config = dict(parameters, **design, test_windows=design['windows'],
                  configuration_sha256=hashlib.sha256(payload).hexdigest())
    seeds = read(root/'common/seeds.json')['seeds']
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError('Exactly five distinct uint32 seeds are required')
    return config, seeds


def output_directory(path):
    """Keep generated artifacts outside the source tree."""
    root, path = Path(__file__).resolve().parents[1], Path(path).resolve()
    if path == root or root in path.parents:
        raise ValueError('Output must be outside the source directory')
    return path

