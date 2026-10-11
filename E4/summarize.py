"""Summarize complete seed records supplied as a normalized CSV table."""
import argparse
import csv
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.results import METRICS, save_csv
from common.runtime import settings, write, output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='External parameter and experiment-design JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        config, seeds = settings('E4', args.config)
        args.output = output_directory(args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    windows = {str(w) if w is not None else 'full' for w in config['windows']}
    rows = list(csv.DictReader(args.records.open(encoding='utf-8-sig')))
    index = {}
    for row in rows:
        key = (row['model'], row['project'], int(row['seed']), row['condition'])
        if key in index or row['condition'] not in windows:
            raise ValueError('Duplicate identity or unsupported result condition')
        index[key] = row
    groups = {'spark_hadoop_kibana': ['spark', 'hadoop', 'kibana'],
              'eclipse_mozilla_vscode': ['eclipse', 'mozilla', 'vscode'],
              'six': ['spark', 'hadoop', 'kibana', 'eclipse', 'mozilla', 'vscode']}
    summary = []
    for model in sorted({r['model'] for r in rows}):
        for group, projects in groups.items():
            conditions = sorted({r['condition'] for r in rows if r['model'] == model})
            for condition in conditions:
                available = sorted({int(r['seed']) for r in rows if r['model'] == model
                                    and r['project'] in projects and r['condition'] == condition})
                for selection, chosen in [('configured', seeds), ('all_provided', available)]:
                    if not chosen:
                        continue
                    complete = all((model, p, seed, condition) in index for p in projects for seed in chosen)
                    if not complete:
                        summary.append(dict(model=model, group=group, condition=condition, selection=selection,
                                            status='INCOMPLETE', seeds=chosen, metrics=None))
                        continue
                    metrics = {}
                    for metric in METRICS:
                        values = [statistics.mean(float(index[model, p, seed, condition][metric]) for p in projects) for seed in chosen]
                        metrics[metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None)
                    summary.append(dict(model=model, group=group, condition=condition, selection=selection,
                                        status='COMPLETE', seeds=chosen, metrics=metrics))
    args.output.mkdir(parents=True, exist_ok=False)
    save_csv(args.output/'all_records.csv', rows)
    write(args.output/'summary.json', summary)


if __name__ == '__main__':
    main()
