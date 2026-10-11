"""Historical candidate buckets, full rankings, and retrieval metrics."""
import numpy as np


def order(r):
    return (r['ts'], int(r['id']))

class Pool:

    def __init__(self, reports):
        self.reports = sorted(reports, key=order)
        self.by_id = {r['id']: r for r in self.reports}
        self.index = {r['id']: i for (i, r) in enumerate(self.reports)}
        self.ts = np.array([r['ts'] for r in self.reports], dtype=np.int64)
        self.ids = np.array([int(r['id']) for r in self.reports], dtype=np.int64)
        self.buckets = np.array([r['bucket'] for r in self.reports])
        self._last_query_history = None

    def _query_history(self, qid):
        state = self._last_query_history
        if state is None or state['qid'] != qid:
            q = self.by_id[qid]
            historical = (self.ts < q['ts']) | (self.ts == q['ts']) & (self.ids < int(qid))
            state = {'qid': qid, 'historical': historical, 'indices': np.flatnonzero(historical), 'latest': None}
            self._last_query_history = state
        return state

    def _query_latest_days(self, qid):
        state = self._query_history(qid)
        if state['latest'] is None:
            latest = {}
            for i in state['indices']:
                (bucket, day) = (self.buckets[i], int(self.ts[i]) // 86400)
                latest[bucket] = max(day, latest.get(bucket, day))
            state['latest'] = latest
        return state['latest']

    def candidates(self, qid, window):
        if window is not None and (type(window) is not int or window <= 0):
            raise ValueError('Unsupported candidate window')
        q = self.by_id[qid]
        idx = self._query_history(qid)['indices']
        if window is None:
            return idx.copy()
        latest = self._query_latest_days(qid)
        day = q['ts'] // 86400
        allowed = {b for (b, t) in latest.items() if day - t <= window}
        return np.array([i for i in idx if self.buckets[i] in allowed], dtype=np.int64)

    def rank(self, qid, scores, window, explicit_order=None):
        idx = self.candidates(qid, window)
        if explicit_order is None:
            vals = np.asarray(scores)[idx]
            if not np.isfinite(vals).all():
                raise ValueError('Nonfinite score')
            idx = idx[np.lexsort((self.ids[idx], -vals))]
        else:
            if set(explicit_order) != set(idx.tolist()):
                raise AssertionError('Incomplete/illegal output ranking')
            idx = np.asarray(explicit_order, dtype=np.int64)
        seen = set()
        gold = self.by_id[qid]['bucket']
        rank = None
        top = []
        for i in idx:
            bucket = self.buckets[i]
            if bucket in seen:
                continue
            seen.add(bucket)
            if len(top) < 10:
                top.append(self.reports[i]['id'])
            if bucket == gold and rank is None:
                rank = len(seen)
        reachable = any((self.buckets[i] == gold for i in idx))
        return {'query_id': qid, 'rank': rank, 'rr1': int(rank is not None and rank <= 1), 'rr5': int(rank is not None and rank <= 5), 'rr10': int(rank is not None and rank <= 10), 'rr': 1.0 / rank if rank else 0.0, 'reachable': int(reachable), 'candidate_count': len(idx), 'candidate_buckets': len(seen), 'top10_report_ids': top}

def metrics(rows):
    if not rows:
        raise ValueError('Empty query denominator')
    n = len(rows)
    return {'n_queries': n, **{k: sum((r[k] for r in rows)) / n for k in ['rr1', 'rr5', 'rr10']}, 'mrr': sum((r['rr'] for r in rows)) / n, 'hits10': sum((r['rr10'] for r in rows)), 'reachable': sum((r['reachable'] for r in rows)), 'mean_candidates': sum((r['candidate_count'] for r in rows)) / n}

def test_metrics(rows):
    result = metrics(rows)
    result.update({f'rr{k}': sum(r['rank'] is not None and r['rank'] <= k for r in rows)/len(rows)
                   for k in (2, 3, 4)})
    return result

def install_fast_metadata(Pool):

    class FastPool(Pool):

        def __init__(self, reports):
            super().__init__(reports)
            (self.bucket_names, self.bucket_codes) = np.unique(self.buckets, return_inverse=True)
            self.report_days = self.ts // 86400

        def _fast_history(self, qid):
            state = self._query_history(qid)
            if 'latest_array' not in state:
                indices = state['indices']
                latest = np.full(len(self.bucket_names), np.iinfo(np.int64).min, dtype=np.int64)
                np.maximum.at(latest, self.bucket_codes[indices], self.report_days[indices])
                state['latest_array'] = latest
            return state

        def candidates(self, qid, window):
            if window is not None and (type(window) is not int or window <= 0):
                raise ValueError('Unsupported candidate window')
            state = self._query_history(qid)
            indices = state['indices']
            if window is None:
                return indices.copy()
            latest = self._fast_history(qid)['latest_array']
            age = self.by_id[qid]['ts'] // 86400 - latest[self.bucket_codes[indices]]
            return indices[age <= window]

        def rank(self, qid, scores, window, explicit_order=None):
            idx = self.candidates(qid, window)
            if explicit_order is None:
                vals = np.asarray(scores)[idx]
                if not np.isfinite(vals).all():
                    raise ValueError('Nonfinite score')
                idx = idx[np.lexsort((self.ids[idx], -vals))]
            else:
                if set(explicit_order) != set(idx.tolist()):
                    raise AssertionError('Incomplete/illegal output ranking')
                idx = np.asarray(explicit_order, dtype=np.int64)
            (_, positions) = np.unique(self.bucket_codes[idx], return_index=True)
            representatives = idx[np.sort(positions)]
            gold_code = self.bucket_codes[self.index[qid]]
            matches = np.flatnonzero(self.bucket_codes[representatives] == gold_code)
            rank = int(matches[0]) + 1 if len(matches) else None
            return {'query_id': qid, 'rank': rank, 'rr1': int(rank is not None and rank <= 1), 'rr5': int(rank is not None and rank <= 5), 'rr10': int(rank is not None and rank <= 10), 'rr': 1.0 / rank if rank else 0.0, 'reachable': int(rank is not None), 'candidate_count': len(idx), 'candidate_buckets': len(representatives), 'top10_report_ids': [self.reports[i]['id'] for i in representatives[:10]]}
    return FastPool
Pool = install_fast_metadata(Pool)

def report_kernels(pool, qid, indices, time_scales):
    indices = np.asarray(indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in 'iu'):
        raise ValueError('Expected one-dimensional integer report indices')
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= len(pool.reports)):
        raise ValueError('Report index outside pool')
    state = pool._fast_history(qid)
    if not np.all(state['historical'][indices]):
        raise ValueError('Nonhistorical candidate')
    ages = np.asarray(pool.by_id[qid]['ts'] // 86400 - state['latest_array'][pool.bucket_codes[indices]], dtype=np.float32)
    assert np.all(ages >= 0)
    scales = np.asarray(time_scales, dtype=np.float32)
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Expected positive finite time scales')
    result = np.ones((len(indices), len(scales) + 1), dtype=np.float32)
    result[:, :-1] = np.exp(-ages[:, None] / scales)
    return result
