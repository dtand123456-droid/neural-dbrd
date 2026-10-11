"""One isolated CUDA process per model, project and seed."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('fit', 'evaluate', 'missingness', 'windows', 'efficiency'))
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--configuration-sha256', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-root', type=Path)
    parser.add_argument('--trained-run', type=Path)
    args = parser.parse_args()
    from .runtime import settings, output_directory
    try:
        design, seeds = settings(args.experiment, args.config)
        args.output = output_directory(args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if design['configuration_sha256'] != args.configuration_sha256:
        parser.error('External configuration changed after scheduling')
    if args.model not in design['models'] or args.project not in design['projects'] or args.seed not in seeds:
        parser.error('Task is outside the configured experiment')
    expected = ('fit', 'evaluate') if design['kind'] == 'training' else (design['kind'],)
    if args.phase not in expected:
        parser.error('Phase is outside the configured experiment')
    if args.phase == 'fit':
        if args.test_root is not None or args.trained_run is not None:
            parser.error('Training accepts only public initialization, Train and Validation')
        from .training import fit_project
        fit_project(args.model, args.project, args.seed, args.fit_root, args.output, args.experiment, args.config)
        return
    if args.test_root is None:
        parser.error('Final evaluation requires --test-root')
    run = args.trained_run or args.output/args.model/args.project/f'seed_{args.seed}'
    from .evaluation import verify_training
    _, cfg = verify_training(run)
    if (cfg['model'], cfg['project'], cfg['seed']) != (args.model, args.project, args.seed):
        raise ValueError('Trained model identity mismatch')
    if args.phase == 'evaluate':
        if cfg['configuration_sha256'] != design['configuration_sha256']:
            raise ValueError('Final evaluation configuration differs from the training run')
        from .evaluation import evaluate_project
        evaluate_project(run, args.fit_root, args.test_root)
    else:
        from .supplement import evaluate_supplement
        evaluate_supplement(run, args.fit_root, args.test_root, args.output, args.experiment, args.config)


if __name__ == '__main__':
    main()
