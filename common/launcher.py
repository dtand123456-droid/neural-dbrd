"""Execute fixed experiment matrices with training before final evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from .runtime import settings, source_hashes, write, sha256, output_directory
from .configuration import PARAMETER_KEYS


def main(experiment):
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True, help='Prepared data and public backbone directory')
    parser.add_argument('--config', type=Path, required=True, help='External parameter and experiment-design JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--trained', type=Path, nargs='+')
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        parser.error('GPU assignments must be distinct')
    try:
        config, seeds = settings(experiment, args.config)
        output = output_directory(args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if (config['kind'] != 'training') != bool(args.trained):
        parser.error('Evaluation experiments require --trained; training experiments initialize publicly')
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(m, p, s) for m in config['models'] for p in config['projects'] for s in seeds]
    plan = dict(config, seeds=seeds, source_hashes=source_hashes())
    write(output/'plan.json', plan)
    trained = {}
    if args.trained:
        from .evaluation import verify_training
        for model, project, seed in tasks:
            matches = [p.resolve()/model/project/f'seed_{seed}' for p in args.trained
                       if (p/model/project/f'seed_{seed}'/'training_complete.json').is_file()]
            if len(matches) != 1:
                raise ValueError(f'Exactly one completed run is required for {(model, project, seed)}')
            _, cfg = verify_training(matches[0])
            if (cfg['model'], cfg['project'], cfg['seed']) != (model, project, seed):
                raise ValueError('Run identity mismatch')
            if any(cfg[key] != config[key] for key in PARAMETER_KEYS):
                raise ValueError('Evaluation must use the trained model parameter settings')
            trained[model, project, seed] = matches[0]
        write(output/'locked_runs.json', {str(k): str(v) for k, v in trained.items()})

    def execute(gpu, phase, jobs):
        for model, project, seed in jobs:
            if sha256(args.config) != config['configuration_sha256']:
                raise RuntimeError('External configuration changed during execution')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED=str(seed))
            command = [sys.executable, '-B', '-m', 'common.worker', phase, '--experiment', experiment,
                       '--config', str(args.config.resolve()),
                       '--configuration-sha256', config['configuration_sha256'],
                       '--model', model, '--project', project, '--seed', str(seed),
                       '--fit-root', str(args.assets.resolve()/'fit'), '--output', str(output)]
            if phase != 'fit':
                command += ['--test-root', str(args.assets.resolve()/'test')]
            if args.trained:
                command += ['--trained-run', str(trained[model, project, seed])]
            with (output/f'{model}_{project}_{seed}_{phase}.log').open('x') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    assignments = [tasks[i::len(args.gpus)] for i in range(len(args.gpus))]
    phases = ('fit', 'evaluate') if config['kind'] == 'training' else (config['kind'],)
    for phase in phases:
        if source_hashes() != plan['source_hashes']:
            raise RuntimeError('Fixed experiment code or configuration changed')
        if sha256(args.config) != config['configuration_sha256']:
            raise RuntimeError('External configuration changed during execution')
        if phase == 'evaluate':
            from .evaluation import verify_training
            for model, project, seed in tasks:
                verify_training(output/model/project/f'seed_{seed}')
            write(output/'all_training_locked.json', {'runs': len(tasks), 'locked_before_test': True})
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(execute, gpu, phase, jobs) for gpu, jobs in zip(args.gpus, assignments)]
            for future in futures:
                future.result()
    from .results import summarize
    summarize(output, plan)
