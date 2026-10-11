"""Evaluate a completed, immutable checkpoint without changing model decisions."""
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from .inputs import FitInputs
from .model import RetrievalModel
from .tokens import tokenize_reports
from .ranking import Pool, test_metrics
from .inference import encode_cache, rank_model
from .runtime import read, write, sha256, source_hashes, setup, check


def verify_training(run):
    run = Path(run)
    finished, cfg = read(run/'training_complete.json'), read(run/'config.json')
    if finished['status'] != 'TRAINING_COMPLETE' or finished['test_used_for_selection']:
        raise ValueError('A Validation-selected completed model is required')
    if sha256(run/'best.pt') != finished['checkpoint_sha256'] or sha256(run/'config.json') != finished['config_sha256']:
        raise ValueError('Checkpoint or configuration changed after training')
    if cfg['source_hashes'] != source_hashes():
        raise ValueError('Code or fixed experiment configuration changed after training')
    return finished, cfg


def evaluate_project(run, fit_root, test_root):
    run = Path(run)
    if (run/'test_results.json').exists() or (run/'test_lock.json').exists():
        raise FileExistsError('Final evaluation has already started for this run')
    finished, cfg = verify_training(run)
    setup(cfg['seed'])
    inputs = FitInputs(Path(fit_root).resolve())
    if inputs.verify(cfg['project']) != cfg['inputs']:
        raise ValueError('Fit input identity changed')
    model = RetrievalModel(inputs.backbone, cfg['seed'], cfg['model'], cfg)
    checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    if checkpoint['config'] != cfg or checkpoint['epoch'] != finished['best_epoch'] or checkpoint['val_metrics'] != finished['val_metrics']:
        raise ValueError('Checkpoint selection record mismatch')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.to('cuda').eval().requires_grad_(False)
    write(run/'test_lock.json', {'checkpoint_sha256': finished['checkpoint_sha256'],
                                'training_complete_sha256': sha256(run/'training_complete.json'),
                                'locked_at': time.time(), 'selection': 'Validation only'})
    started = time.time()
    test_root = Path(test_root).resolve()
    project = cfg['project']
    manifest = read(test_root/'manifest.json')['projects'][project]
    for name, expected in manifest['files'].items():
        if name not in ('test.jsonl', 'test_queries.json') or sha256(test_root/project/name) != expected:
            raise ValueError('Test input identity mismatch')
    with (test_root/project/'test.jsonl').open() as stream:
        test_reports = [json.loads(line) for line in stream]
    if not test_reports or any(r['phase'] != 'test' for r in test_reports):
        raise ValueError('Invalid Test split')
    queries = read(test_root/project/'test_queries.json')
    reports = inputs.reports(project, 'train')+inputs.reports(project, 'val')+test_reports
    pool = Pool(reports)
    if any(pool.by_id[q]['phase'] != 'test' for q in queries) or len(set(queries)) != len(queries):
        raise ValueError('Invalid Test queries')
    tokenizer = AutoTokenizer.from_pretrained(inputs.backbone, local_files_only=True)
    tokens = tokenize_reports(tokenizer, pool.reports, cfg)
    cache = encode_cache(model, tokens, tokenizer.pad_token_id, 'cuda', cfg['eval_batch'])
    result = {'status': 'COMPLETE', 'model': cfg['model'], 'project': project, 'seed': cfg['seed'],
              'checkpoint_sha256': finished['checkpoint_sha256'], 'metrics': {}}
    with (run/'test_predictions.jsonl').open('x') as stream:
        for window in cfg['test_windows']:
            label = str(window) if window is not None else 'full'
            rows = []
            for index, qid in enumerate(queries):
                check()
                row = rank_model(model, cache, pool, qid, window)
                rows.append(row)
                stream.write(json.dumps(dict(row, window=label))+'\n')
                if (index+1)%100 == 0:
                    write(run/'evaluation_status.json', {'status': 'EVALUATING', 'window': label,
                                                        'queries_done': index+1, 'queries_total': len(queries)})
            result['metrics'][label] = test_metrics(rows)
    result.update(runtime_seconds=time.time()-started, peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
    verify_training(run)
    write(run/'test_results.json', result)
    write(run/'evaluation_status.json', {'status': 'COMPLETE'})
    print(json.dumps(result), flush=True)
    return result
