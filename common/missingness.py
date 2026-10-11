"""Deterministic deletion of effective query body units."""
import hashlib
import json
SCHEMA = 'DBRD_QUERY_UNIT_MISSINGNESS_V1'

def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def token_hash(token):
    return content_hash([list(part) for part in token])

def body_units(token):
    (ids, fields, units) = token
    assert len(ids) == len(fields) == len(units)
    assert all(((unit > 0) == (field > 0) for (unit, field) in zip(units, fields)))
    assert all((unit == 1 for (unit, field) in zip(units, fields) if field == 1))
    return sorted({unit for (unit, field) in zip(units, fields) if field == 2})

def drop_order(project, query_id, units, repeat, mask_seed):
    return sorted(units, key=lambda unit: (content_hash([SCHEMA, mask_seed, project, str(query_id), repeat, unit]), unit))

def apply_mask(token, dropped):
    (ids, fields, units) = token
    present = body_units(token)
    dropped = set(dropped)
    assert dropped <= set(present)
    assert len(present) == 0 or len(present) - len(dropped) >= 1
    keep_positions = [i for (i, (field, unit)) in enumerate(zip(fields, units)) if field != 2 or unit not in dropped]
    result = tuple(([part[i] for i in keep_positions] for part in token))
    assert [(i, f, u) for (i, f, u) in zip(*result) if f != 2] == [(i, f, u) for (i, f, u) in zip(*token) if f != 2]
    assert body_units(result) == [unit for unit in present if unit not in dropped]
    return result

def query_masks(project, query_id, token, design):
    if project not in design['projects']:
        raise ValueError('Project is outside the mask design')
    present = body_units(token)
    n = len(present)
    body_count = token[1].count(2)
    masks = []
    for repeat in design['repeats']:
        ordering = drop_order(project, query_id, present, repeat, design['mask_seed'])
        for level in design['levels']:
            number = min(max(0, n - 1), n * level // 100)
            dropped = sorted(ordering[:number])
            masked = apply_mask(token, dropped)
            removed_tokens = len(token[0]) - len(masked[0])
            masks.append({'repeat': repeat, 'nominal_percent': level, 'drop_body_units': dropped, 'kept_body_units': body_units(masked), 'dropped_body_unit_count': number, 'actual_body_unit_missing_fraction': number / n if n else 0.0, 'actual_body_token_missing_fraction': removed_tokens / body_count if body_count else 0.0, 'target_fraction_exactly_realized': number * 100 == n * level if n else level == 0, 'no_body_units': n == 0, 'cannot_delete_while_retaining_body': n <= 1, 'masked_token_count': len(masked[0]), 'masked_token_sha256': token_hash(masked)})
    return {'query_id': str(query_id), 'original_body_units': present, 'original_body_unit_count': n, 'original_body_token_count': body_count, 'original_title_token_count': token[1].count(1), 'original_token_count': len(token[0]), 'original_token_sha256': token_hash(token), 'masks': masks}
