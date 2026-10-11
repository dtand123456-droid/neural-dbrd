"""Fixed-window inference, query-body deletion and inference timing."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from .evaluation import verify_training
from .configuration import PARAMETER_KEYS
from .inputs import FitInputs
from .model import RetrievalModel
from .tokens import tokenize_reports, collate_tokens
from .ranking import Pool, test_metrics
from .inference import encode_cache, rank_model
from .runtime import read, write, sha256, settings, setup, check
from .tensor_utils import tensor_state_sha256
from .missingness import query_masks, apply_mask, token_hash, content_hash


def locked_context(run, fit_root, test_root, output, experiment, config_path):
    finished, cfg = verify_training(run)
    design, seeds = settings(experiment, config_path)
    if any(cfg[key] != design[key] for key in PARAMETER_KEYS):
        raise ValueError('Evaluation parameters differ from the locked training run')
    if design['kind'] == 'training' or cfg['model'] not in design['models'] or cfg['project'] not in design['projects'] or cfg['seed'] not in seeds:
        raise ValueError('Model is outside the evaluation design')
    output = Path(output)/cfg['model']/cfg['project']/f"seed_{cfg['seed']}"
    output.mkdir(parents=True, exist_ok=False)
    setup(cfg['seed'])
    fit = FitInputs(Path(fit_root).resolve())
    if fit.verify(cfg['project']) != cfg['inputs']:
        raise ValueError('Fit input identity changed')
    model = RetrievalModel(fit.backbone, cfg['seed'], cfg['model'], cfg)
    checkpoint = torch.load(Path(run)/'best.pt', map_location='cpu', weights_only=False)
    if checkpoint['config'] != cfg or checkpoint['epoch'] != finished['best_epoch'] or checkpoint['val_metrics'] != finished['val_metrics']:
        raise ValueError('Checkpoint selection record mismatch')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.to('cuda').eval().requires_grad_(False)
    write(output/'test_lock.json', dict(checkpoint_sha256=finished['checkpoint_sha256'],
          training_complete_sha256=sha256(Path(run)/'training_complete.json'),
          experiment=experiment, design=design, locked_at=time.time(), selection='Validation only'))
    test_root = Path(test_root).resolve()
    project = cfg['project']
    manifest = read(test_root/'manifest.json')['projects'][project]
    for name, expected in manifest['files'].items():
        if name not in ('test.jsonl', 'test_queries.json') or sha256(test_root/project/name) != expected:
            raise ValueError('Test input identity mismatch')
    test = [json.loads(line) for line in (test_root/project/'test.jsonl').open()]
    if not test or any(r['phase'] != 'test' for r in test):
        raise ValueError('Invalid Test reports')
    pool = Pool(fit.reports(project, 'train')+fit.reports(project, 'val')+test)
    queries = read(test_root/project/'test_queries.json')
    if len(set(queries)) != len(queries) or any(pool.by_id[q]['phase'] != 'test' for q in queries):
        raise ValueError('Invalid Test queries')
    tokenizer = AutoTokenizer.from_pretrained(fit.backbone, local_files_only=True)
    tokens = tokenize_reports(tokenizer, pool.reports, cfg)
    return finished, cfg, design, output, model, pool, queries, tokens, tokenizer.pad_token_id


@torch.inference_mode()
def evaluate_supplement(run, fit_root, test_root, output, experiment, config_path):
    started = time.time()
    finished, cfg, design, output, model, pool, queries, tokens, pad = locked_context(run, fit_root, test_root, output, experiment, config_path)
    torch.cuda.synchronize()
    tic = time.perf_counter()
    cache = encode_cache(model, tokens, pad, 'cuda', cfg['eval_batch'])
    torch.cuda.synchronize()
    cache_seconds = time.perf_counter()-tic
    cache_bytes = sum(t.numel()*t.element_size() for t in cache.values())
    result = dict(status='COMPLETE', model=cfg['model'], project=cfg['project'], seed=cfg['seed'],
                  experiment=experiment, checkpoint_sha256=finished['checkpoint_sha256'], metrics={},
                  parameter_count=cfg['parameter_count'], training_seconds=finished['runtime_seconds'],
                  pretraining_seconds=finished['pretraining_seconds'], cache_seconds=cache_seconds, cache_bytes=cache_bytes)
    window = design['windows'][0]
    label = str(window) if window is not None else 'full'
    with (output/'predictions.jsonl').open('x') as stream:
        if design['kind'] == 'windows':
            for window in design['windows']:
                label = str(window) if window is not None else 'full'
                rows = []
                for qid in queries:
                    check()
                    row = rank_model(model, cache, pool, qid, window)
                    rows.append(row)
                    stream.write(json.dumps(dict(row, window=label))+'\n')
                result['metrics'][label] = test_metrics(rows)
        elif design['kind'] == 'missingness':
            masks = {q: query_masks(cfg['project'], q, tokens[pool.index[q]], design) for q in queries}
            write(output/'masks.json', masks)
            result['masks_sha256'] = sha256(output/'masks.json')
            cache_digest = tensor_state_sha256(cache)
            zero = {q: rank_model(model, cache, pool, q, window) for q in queries}
            for repeat in design['repeats']:
                for level in design['levels']:
                    rows = []
                    for qid in queries:
                        check()
                        entry = next(r for r in masks[qid]['masks'] if r['repeat'] == repeat and r['nominal_percent'] == level)
                        if entry['drop_body_units']:
                            masked = apply_mask(tokens[pool.index[qid]], entry['drop_body_units'])
                            if token_hash(masked) != entry['masked_token_sha256']:
                                raise ValueError('Query mask identity mismatch')
                            with torch.autocast('cuda', dtype=torch.float16):
                                encoded = model(**collate_tokens([masked], [0], pad, 'cuda'))
                            features = {}
                            for key, original in cache.items():
                                value = encoded[key][0].to(dtype=original.dtype)
                                shape = original[pool.index[qid]].shape
                                if value.shape != shape:
                                    padded = torch.zeros(shape, dtype=original.dtype, device=original.device)
                                    padded[:value.shape[0]] = value
                                    value = padded
                                features[key] = value
                            row = rank_model(model, cache, pool, qid, window, query_features=features)
                            if any(row[k] != zero[qid][k] for k in ('candidate_count', 'candidate_buckets', 'reachable')):
                                raise ValueError('Query deletion changed candidate protocol')
                        else:
                            row = zero[qid]
                        row = dict(row, window=label, repeat=repeat, nominal_percent=level,
                                   mask_sha256=content_hash(entry), masked_token_sha256=entry['masked_token_sha256'],
                                   actual_body_unit_missing_fraction=entry['actual_body_unit_missing_fraction'],
                                   actual_body_token_missing_fraction=entry['actual_body_token_missing_fraction'])
                        rows.append(row)
                        stream.write(json.dumps(row)+'\n')
                    condition = f'repeat{repeat}_percent{level}'
                    result['metrics'][condition] = test_metrics(rows)
                    write(output/'status.json', dict(status='EVALUATING', condition=condition))
            if tensor_state_sha256(cache) != cache_digest:
                raise RuntimeError('Candidate feature cache changed')
            result['candidate_cache_unchanged'] = True
        elif design['kind'] == 'efficiency':
            for qid in queries[:design['warmup_queries']]:
                rank_model(model, cache, pool, qid, window)
            timings, reference = [], None
            for repeat in range(design['timing_repeats']):
                rows, seconds = [], []
                for qid in queries:
                    check()
                    torch.cuda.synchronize()
                    tic = time.perf_counter()
                    row = rank_model(model, cache, pool, qid, window)
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter()-tic)
                    rows.append(row)
                    stream.write(json.dumps(dict(row, repeat=repeat, window=label, retrieval_seconds=seconds[-1]))+'\n')
                if reference is not None and rows != reference:
                    raise RuntimeError('Inference repetitions produced different rankings')
                reference = rows
                timings.append(dict(repeat=repeat, total_seconds=sum(seconds),
                                    mean_seconds=float(np.mean(seconds)), median_seconds=float(np.median(seconds)),
                                    p95_seconds=float(np.quantile(seconds, .95)), queries_per_second=len(queries)/sum(seconds)))
            result['metrics'][label] = test_metrics(reference)
            result['retrieval_timing'] = timings
            result['timing_scope'] = 'Preencoded report cache; candidate selection, scoring, reranking and bucket ranking; GPU synchronized'
            result['device'] = torch.cuda.get_device_name()
        else:
            raise ValueError('Unknown evaluation kind')
    verify_training(run)
    result.update(runtime_seconds=time.time()-started, peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
                  reserved_gpu_gb=torch.cuda.max_memory_reserved()/1024**3)
    write(output/'results.json', result)
    write(output/'status.json', {'status': 'COMPLETE'})
    return result
