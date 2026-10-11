"""Public initialization, Train-only learning, and Validation checkpoint selection."""
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from .inputs import FitInputs
from .model import RetrievalModel
from .tokens import tokenize_reports, collate_tokens
from .ranking import Pool, metrics, order, report_kernels
from .inference import encode_cache, rank_model
from .sampling import sample_group
from .losses import group_loss
from .pretraining import pretrain
from .tensor_utils import tensor_state_sha256
from .runtime import settings, setup, check, write, sha256, source_hashes


def selection(value):
    return value['rr10'], value['rr5'], value['mrr']


def validate(model, pool, tokens, pad, queries, cfg):
    window = None if cfg['selection_window'] == 'full' else cfg['selection_window']
    cache = encode_cache(model, tokens, pad, 'cuda', cfg['eval_batch'])
    rows = []
    for qid in queries:
        check()
        rows.append(rank_model(model, cache, pool, qid, window))
    del cache
    return metrics(rows), rows


def train_epoch(model, pool, tokens, pad, mining, queries, optimizer, scheduler, cfg, epoch, state, full_digest):
    model.train()
    rng = random.Random(cfg['seed']*10000+epoch)
    shuffled = list(queries)
    rng.shuffle(shuffled)
    starts = list(range(0, len(shuffled), cfg['microbatch']))
    optimizer.zero_grad(set_to_none=True)
    losses, digest = [], hashlib.sha256()
    for bi, start in enumerate(starts):
        check()
        selected = shuffled[start:start+cfg['microbatch']]
        groups = [sample_group(mining, pool, q, epoch, rng, cfg) for q in selected]
        indices = [pool.index[q] for q in selected] + [i for group in groups for i in group]
        if any(pool.reports[i]['phase'] != 'train' for i in indices):
            raise ValueError('Training batch contains a non-Train report')
        for q, group in zip(selected, groups):
            if any(order(pool.reports[i]) >= order(pool.by_id[q]) for i in group):
                raise ValueError('Training candidate is not historical')
        record = {'epoch': epoch, 'microbatch': bi, 'queries': selected,
                  'groups': [[pool.reports[i]['id'] for i in group] for group in groups]}
        line = (json.dumps(record, sort_keys=True, separators=(',', ':'))+'\n').encode()
        digest.update(line)
        full_digest.update(line)
        batch = collate_tokens(tokens, indices, pad, 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = model(**batch)
        with torch.autocast('cuda', enabled=False):
            scores = []
            offset = len(selected)
            for qi, (q, group) in enumerate(zip(selected, groups)):
                query = {k: features[k][qi] for k in model.feature_keys}
                group_size = len(group)
                docs = {k: features[k][offset:offset+group_size] for k in model.feature_keys}
                offset += group_size
                kernels = torch.as_tensor(
                    report_kernels(pool, q, np.asarray(group), cfg['time_scales']), device='cuda')
                _, joint = model.score_joint(query, docs, kernels)
                scores.append(joint)
            scores = torch.stack(scores)
            positive = torch.zeros_like(scores, dtype=torch.bool)
            positive[:, 0] = True
            loss = group_loss(scores, positive, cfg['temperature'])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite training loss')
        divisor = min(cfg['accumulate'], len(starts)-(bi//cfg['accumulate'])*cfg['accumulate'])
        (loss/divisor).backward()
        losses.append(float(loss.detach()))
        check()
        if (bi+1)%cfg['accumulate'] == 0 or bi == len(starts)-1:
            total = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            state['steps'] += 1
            optimizer.zero_grad(set_to_none=True)
    return {'loss': float(np.mean(losses)), 'sampling_sha256': digest.hexdigest(),
            'optimizer_steps': state['steps'], 'last_gradient_norm': float(total)}


def fit_project(model_name, project, seed, fit_root, output_root, experiment, config_path):
    config, seeds = settings(experiment, config_path)
    if config['kind'] != 'training':
        raise ValueError('This experiment evaluates completed models only')
    if model_name not in config['models']:
        raise ValueError('Model is outside the configured experiment')
    if project not in config['projects'] or seed not in seeds:
        raise ValueError('Project or seed is outside the configured experiment')
    run = Path(output_root)/model_name/project/f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cfg = dict(config, model=model_name, project=project, seed=seed, source_hashes=source_hashes())
    state, history = {'steps': 0}, []
    try:
        setup(seed)
        inputs = FitInputs(Path(fit_root).resolve())
        cfg['inputs'] = inputs.verify(project)
        train = inputs.reports(project, 'train')
        val = inputs.reports(project, 'val')
        pool, trainpool = Pool(train+val), Pool(train)
        queries, valqueries = inputs.queries(project, 'train'), inputs.queries(project, 'val')
        if any(q not in trainpool.by_id for q in queries) or any(pool.by_id[q]['phase'] != 'val' for q in valqueries):
            raise ValueError('Query split mismatch')
        mining = inputs.mining(project, train, queries)
        setup(seed)
        model = RetrievalModel(inputs.backbone, seed, model_name, cfg)
        write(run/'initialization.json', {'source': 'sentence-transformers/all-mpnet-base-v2',
              'revision': 'e8c3b32edf5434bc2275fc9bab85f82640a19130',
              'full_state_sha256': tensor_state_sha256(model.state_dict()),
              'loading_info': model.public_loading_info})
        cfg['parameter_count'] = sum(p.numel() for p in model.parameters())
        cfg['environment'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                              'numpy': np.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name()}
        steps = math.ceil(math.ceil(len(queries)/cfg['microbatch'])/cfg['accumulate'])
        cfg['scheduler_total_steps'] = steps*cfg['max_epochs']
        cfg['scheduler_warmup_steps'] = math.ceil(cfg['warmup_fraction']*steps*cfg['max_epochs'])
        write(run/'config.json', cfg)
        tokenizer = AutoTokenizer.from_pretrained(inputs.backbone, local_files_only=True)
        tokens = tokenize_reports(tokenizer, pool.reports, cfg)
        train_tokens = [tokens[pool.index[r['id']]] for r in train]
        model.to('cuda')
        pretraining = pretrain(model, train, train_tokens, tokenizer.pad_token_id, run, seed, cfg)
        setup(seed)
        best, rows = validate(model, pool, tokens, tokenizer.pad_token_id, valqueries, cfg)
        best_epoch, bad = 0, 0

        def save(epoch, value, predictions):
            temporary = run/'best.tmp'
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'val_metrics': value,
                        'config': cfg, 'validation_rows': predictions}, temporary)
            temporary.replace(run/'best.pt')

        save(0, best, rows)
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        module_ids = {id(p) for p in model.evidence_module.parameters()}
        token_ids = {id(p) for p in model.c3_module.parameters()} if model.c3_enabled else set()
        groups = [
            {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
            {'params': [p for p in model.parameters() if id(p) not in encoder_ids and id(p) not in module_ids and id(p) not in token_ids], 'lr': cfg['head_lr']},
            {'params': model.evidence_module.parameters(), 'lr': cfg['module_lr']}]
        if model.c3_enabled:
            groups.append({'params': model.c3_module.parameters(), 'lr': cfg['token_lr']})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        scheduler = get_linear_schedule_with_warmup(optimizer, cfg['scheduler_warmup_steps'], cfg['scheduler_total_steps'])
        setup(seed)
        full_digest = hashlib.sha256()
        for epoch in range(1, cfg['max_epochs']+1):
            tic = time.time()
            record = train_epoch(model, trainpool, train_tokens, tokenizer.pad_token_id, mining,
                                 queries, optimizer, scheduler, cfg, epoch, state, full_digest)
            train_seconds = time.time()-tic
            tic = time.time()
            value, rows = validate(model, pool, tokens, tokenizer.pad_token_id, valqueries, cfg)
            improved = selection(value) > selection(best)
            if improved:
                best, best_epoch, bad = value, epoch, 0
                save(epoch, best, rows)
            else:
                bad += 1
            record.update(epoch=epoch, val_metrics=value, improved=improved, best_epoch=best_epoch,
                          bad_epochs=bad, train_seconds=train_seconds, validation_seconds=time.time()-tic,
                          elapsed_seconds=time.time()-started)
            history.append(record)
            write(run/'history.json', history)
            write(run/'status.json', dict(record, status='TRAINING'))
            print(json.dumps(record), flush=True)
            if bad >= cfg['patience']:
                break
        check()
        checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        if checkpoint['config'] != cfg or checkpoint['epoch'] != best_epoch or metrics(checkpoint['validation_rows']) != best:
            raise RuntimeError('Selected checkpoint does not match its Validation result')
        write(run/'validation_predictions.json', checkpoint['validation_rows'])
        finished = {'status': 'TRAINING_COMPLETE', 'project': project, 'seed': seed,
                    'best_epoch': best_epoch, 'epochs_run': len(history), 'val_metrics': best,
                    'optimizer_steps': state['steps'], 'parameter_count': cfg['parameter_count'],
                    'checkpoint_sha256': sha256(run/'best.pt'), 'config_sha256': sha256(run/'config.json'),
                    'state_sha256': tensor_state_sha256(checkpoint['state_dict']),
                    'sampling_sha256': full_digest.hexdigest(), 'runtime_seconds': time.time()-started,
                    'pretraining_seconds': pretraining['runtime_seconds'],
                    'peak_gpu_gb': torch.cuda.max_memory_allocated()/1024**3,
                    'test_used_for_selection': False, 'completed_at': time.time()}
        write(run/'training_complete.json', finished)
        write(run/'status.json', finished)
        return run
    except BaseException as exc:
        write(run/'status.json', {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc(),
                                 'epochs_completed': len(history), 'optimizer_steps': state['steps']})
        raise
