"""Every-seed results and equal-project macro mean and sample SD."""
import csv
from pathlib import Path
import statistics
from .runtime import read, write

METRICS = ('rr1', 'rr2', 'rr3', 'rr4', 'rr5', 'rr10', 'mrr')


def save_csv(path, rows):
    if not rows:
        raise ValueError('Cannot summarize an empty experiment')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, plan):
    output = Path(output)
    raw, summary = [], []
    for model in plan['models']:
        for project in plan['projects']:
            for seed in plan['seeds']:
                name = 'test_results.json' if plan['kind'] == 'training' else 'results.json'
                value = read(output/model/project/f'seed_{seed}'/name)
                if value['status'] != 'COMPLETE' or (value['model'], value['project'], value['seed']) != (model, project, seed):
                    raise ValueError('Missing or mismatched result')
                for condition, metrics in value['metrics'].items():
                    raw.append(dict(model=model, project=project, seed=seed, condition=condition,
                                    n_queries=metrics['n_queries'], **{k: metrics[k] for k in METRICS}))
    groups = {p: [p] for p in plan['projects']}
    for name, members in {'macro_spark_hadoop_kibana': ['spark', 'hadoop', 'kibana'],
                          'macro_eclipse_mozilla_vscode': ['eclipse', 'mozilla', 'vscode'],
                          'macro_all': plan['projects']}.items():
        if set(members) <= set(plan['projects']):
            groups[name] = members
    for model in plan['models']:
        conditions = sorted({r['condition'] for r in raw if r['model'] == model})
        for group, projects in groups.items():
            for condition in conditions:
                for metric in METRICS:
                    values = []
                    for seed in plan['seeds']:
                        matches = [r for r in raw if r['model'] == model and r['seed'] == seed
                                   and r['condition'] == condition and r['project'] in projects]
                        if len(matches) != len(projects) or {r['project'] for r in matches} != set(projects):
                            raise ValueError('Incomplete project/seed matrix')
                        values.append(statistics.mean(r[metric] for r in matches))
                    summary.append(dict(model=model, group=group, condition=condition, metric=metric,
                                        n_seeds=len(values), mean=statistics.mean(values),
                                        sample_sd=statistics.stdev(values) if len(values)>1 else None))
    save_csv(output/'per_seed_results.csv', raw)
    save_csv(output/'mean_sd.csv', summary)
    write(output/'results.json', {'plan': plan, 'per_seed': raw, 'summary': summary})
