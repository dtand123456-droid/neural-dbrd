"""Train-only report-view contrastive pretraining from external parameters."""
import hashlib
import json
import math
import random
import time
import torch
from transformers import get_linear_schedule_with_warmup
from .tokens import collate_tokens
from .losses import evidence_view, consistency_loss
from .runtime import write, check, sha256
from .tensor_utils import tensor_state_sha256


def full_report_view(token):
    ids, fields, units = token
    return (list(ids), list(fields), list(units)), sorted(set(units) - {0})


def pretrain(model, reports, tokens, pad, run, seed, config):
    if not reports or any(r['phase'] != 'train' for r in reports) or len(reports) != len(tokens):
        raise ValueError('Pretraining requires matching Train reports and tokens')
    eligible = [i for i, token in enumerate(tokens) if any(token[2])]
    if len(eligible) < 2:
        raise ValueError('At least two reports with lexical units are required')
    batch_size = config['pretraining_batch']
    epochs = config['pretraining_epochs']
    if batch_size < 2 or epochs < 1:
        raise ValueError('Pretraining requires batch size >= 2 and at least one epoch')
    batches = math.ceil(len(eligible)/batch_size) - int(len(eligible)%batch_size == 1)
    total = batches * epochs
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['pretraining_encoder_lr']},
        {'params': model.evidence_module.parameters(), 'lr': config['pretraining_module_lr']},
        {'params': [model.logit_mix], 'lr': config['pretraining_mix_lr']}],
        weight_decay=config['pretraining_weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(config['pretraining_warmup_fraction']*total), total)
    initial = tensor_state_sha256(model.state_dict())
    started = time.time()
    steps, history = 0, []
    full_digest = hashlib.sha256()
    for epoch in range(1, epochs+1):
        check()
        model.train()
        rng = random.Random(seed*100000 + epoch)
        order = list(eligible)
        rng.shuffle(order)
        losses, digest = [], hashlib.sha256()
        for offset in range(0, len(order), batch_size):
            check()
            selected = order[offset:offset+batch_size]
            if len(selected) < 2:
                continue
            if model.view_mode == 'unit_drop':
                a = [evidence_view(tokens[i], rng, 'A', config) for i in selected]
                b = [evidence_view(tokens[i], rng, 'B', config) for i in selected]
            else:
                a = [full_report_view(tokens[i]) for i in selected]
                b = [full_report_view(tokens[i]) for i in selected]
            views = [token for token, _ in a+b]
            sample = {'epoch': epoch, 'step': steps, 'report_ids': [reports[i]['id'] for i in selected],
                      'view_A_units': [u for _, u in a], 'view_B_units': [u for _, u in b]}
            if model.view_mode == 'full_report_dropout':
                sample.update(view_mode=model.view_mode, input_tokens_per_view=[len(t[0]) for t in views],
                              full_report_input_duplicated=True)
            line = (json.dumps(sample, sort_keys=True, separators=(',', ':'))+'\n').encode()
            digest.update(line)
            full_digest.update(line)
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = model(**collate_tokens(views, list(range(len(views))), pad, 'cuda'))
            with torch.autocast('cuda', enabled=False):
                loss, _ = consistency_loss(
                    model, features, [reports[i]['bucket'] for i in selected],
                    config['pretraining_temperature'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite pretraining loss')
            loss.backward()
            check()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['pretraining_gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            steps += 1
            losses.append(float(loss.detach()))
            if steps % 25 == 0:
                write(run/'status.json', {'status': 'PRETRAINING', 'epoch': epoch,
                                         'step': steps, 'total_steps': total,
                                         'elapsed_seconds': time.time()-started})
        history.append({'epoch': epoch, 'optimizer_steps': steps,
                        'mean_loss': sum(losses)/len(losses), 'batches_completed': len(losses),
                        'sampling_sha256': digest.hexdigest()})
        print(json.dumps({'event': 'PRETRAIN_EPOCH', **history[-1]}), flush=True)
    check()
    final = tensor_state_sha256(model.state_dict())
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler
    path = run/'pretrained.pt'
    torch.save({'state_dict': model.state_dict(), 'steps': steps, 'seed': seed}, path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    if steps != total or tensor_state_sha256(model.state_dict()) != final:
        raise RuntimeError('Incomplete pretraining handoff')
    record = {'epochs': epochs, 'steps': steps, 'history': history,
              'initial_state_sha256': initial, 'output_state_sha256': final,
              'checkpoint_sha256': sha256(path), 'sampling_sha256': full_digest.hexdigest(),
              'fit_phases': ['train'], 'runtime_seconds': time.time()-started}
    write(run/'pretraining.json', record)
    torch.cuda.empty_cache()
    return record
