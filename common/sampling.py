"""Train-only positives and externally configured lexical/dense negatives."""
import random
from .ranking import order


def sample_group(mining, pool, qid, epoch, rng, config):
    if not isinstance(rng, random.Random) or epoch < 1:
        raise ValueError('An explicit epoch RNG and positive epoch are required')
    query = next((row for row in mining['queries'] if row['query'] == qid))
    q = pool.by_id[qid]
    if q['phase'] != 'train':
        raise ValueError('Only train queries may be sampled')
    past = pool.candidates(qid, None)
    random_ids = [pool.reports[i]['id'] for i in past if pool.reports[i]['phase'] == 'train' and pool.buckets[i] != q['bucket']]
    assert random_ids
    valid = set(random_ids)
    counts = (('lexical_hard_ids', config['lexical_negative_count']),
              ('dense_hard_ids', config['dense_negative_count']),
              ('random', config['random_negative_count']))
    if any(type(count) is not int or count < 0 for _, count in counts):
        raise ValueError('Negative counts must be nonnegative integers')
    negative_count = sum(count for _, count in counts)
    if negative_count < 1:
        raise ValueError('At least one negative per training group is required')
    chosen = []
    for name, count in counts:
        available = random_ids if name == 'random' else query[name]
        assert set(available) <= valid
        choices = [i for i in available if i not in chosen]
        chosen += rng.sample(choices, min(count, len(choices)))
    fallback = [i for i in random_ids if i not in chosen]
    while len(chosen) < negative_count:
        chosen.append(fallback.pop(rng.randrange(len(fallback))) if fallback else rng.choice(random_ids))
    positive = query['positive_ids'][(epoch - 1) % len(query['positive_ids'])]
    p = pool.by_id[positive]
    assert p['phase'] == 'train' and order(p) < order(q) and (p['bucket'] == q['bucket'])
    return [pool.index[positive]] + [pool.index[i] for i in chosen]
