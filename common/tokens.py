"""Title/body tokenization and sentence units with retained head/tail gaps."""
import bisect
import re
import torch


def canonical_tokens(tokenizer, reports, config):
    max_length = config['max_length']
    titles = tokenizer([r['title'] for r in reports], add_special_tokens=False, truncation=True, max_length=config['title_max_length'])['input_ids']
    bodies = tokenizer([r['body'] for r in reports], add_special_tokens=False, truncation=False, verbose=False)['input_ids']
    ns = tokenizer.num_special_tokens_to_add(pair=True)
    assert ns in (3, 4)
    out = []
    for (t, b) in zip(titles, bodies):
        budget = max_length - len(t) - ns
        if len(b) > budget:
            h = int(budget * config['body_head_fraction'])
            b = b[:h] + b[-(budget - h):] if budget - h else b[:h]
        ids = tokenizer.build_inputs_with_special_tokens(t, b)
        fields = [0] + [1] * len(t) + [0] * (ns - 2) + [2] * len(b) + [0]
        assert len(ids) == len(fields) and len(ids) <= max_length
        out.append((ids, fields))
    return out

def canonical_collate(tokens, indices, pad_id, device):
    lengths = [len(tokens[i][0]) for i in indices]
    length = max(lengths)
    ids = torch.full((len(indices), length), pad_id, dtype=torch.long)
    fields = torch.zeros_like(ids)
    mask = torch.zeros_like(ids)
    for (row, i) in enumerate(indices):
        (tok, field) = tokens[i]
        n = len(tok)
        ids[row, :n] = torch.tensor(tok)
        fields[row, :n] = torch.tensor(field)
        mask[row, :n] = 1
    return {'input_ids': ids.to(device), 'attention_mask': mask.to(device), 'fields': fields.to(device)}

def tokenize_reports(tokenizer, reports, config):
    max_length = config['max_length']
    max_units = config['max_units']
    tokens = canonical_tokens(tokenizer, reports, config)
    body_data = tokenizer([r['body'] for r in reports], add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
    output = []
    for (report, (ids, fields), full_body, offsets) in zip(reports, tokens, body_data['input_ids'], body_data['offset_mapping']):
        ntitle = fields.count(1)
        budget = max_length - ntitle - tokenizer.num_special_tokens_to_add(pair=True)
        if len(full_body) > budget:
            head = int(budget * config['body_head_fraction'])
            selected = list(range(head)) + list(range(len(full_body) - (budget - head), len(full_body)))
        else:
            selected = list(range(len(full_body)))
        assert [ids[i] for (i, f) in enumerate(fields) if f == 2] == [full_body[i] for i in selected]
        boundaries = [m.end() for m in re.finditer('(?:[.!?](?=\\s|$)|\\n+)', report['body'])]
        groups = []
        for (local, original) in enumerate(selected):
            region = int(local > 0 and original != selected[local - 1] + 1)
            sentence = bisect.bisect_right(boundaries, offsets[original][0])
            if not groups or region or sentence != groups[-1]['sentence']:
                groups.append({'positions': [local], 'sentence': sentence, 'region': groups[-1]['region'] + region if groups else 0})
            else:
                groups[-1]['positions'].append(local)
        while len(groups) > max_units - 1:
            eligible = [i for i in range(len(groups) - 1) if groups[i]['region'] == groups[i + 1]['region']]
            assert eligible
            i = min(eligible, key=lambda j: (len(groups[j]['positions']) + len(groups[j + 1]['positions']), j))
            groups[i]['positions'].extend(groups[i + 1]['positions'])
            groups.pop(i + 1)
        body_units = [0] * len(selected)
        for (unit, group) in enumerate(groups, start=2):
            for i in group['positions']:
                body_units[i] = unit
        assert all(body_units)
        unit_ids = []
        bi = 0
        for f in fields:
            unit_ids.append(1 if f == 1 else body_units[bi] if f == 2 else 0)
            bi += int(f == 2)
        assert len(ids) == len(unit_ids) and bi == len(selected)
        output.append((ids, fields, unit_ids))
    return output

def collate_tokens(tokens, indices, pad_id, device):
    batch = canonical_collate([(tokens[i][0], tokens[i][1]) for i in indices], list(range(len(indices))), pad_id, device)
    units = torch.zeros_like(batch['fields'])
    for (row, index) in enumerate(indices):
        units[row, :len(tokens[index][2])] = torch.tensor(tokens[index][2], device=units.device)
    batch['unit_ids'] = units
    return batch
