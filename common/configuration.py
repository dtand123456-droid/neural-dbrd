"""Validate externally supplied experiment parameters without supplying defaults.

The input JSON has two mappings: ``parameters`` for shared settings and
``experiments`` for experiment-ID keyed designs. Seed identities are read
separately from the package's seeds.json. This module defines field contracts.
"""
import math


POSITIVE_INTS = frozenset('''
pretraining_epochs max_epochs patience microbatch accumulate max_length eval_batch
pretraining_batch max_units title_max_length alignment_width profile_projection_width
profile_kernel_count profile_readout_width coarse_batch_size graph_blocks
graph_warmup_steps fine_batch_size rerank_topk
'''.split())
NONNEGATIVE_INTS = frozenset('''
lexical_negative_count dense_negative_count random_negative_count
'''.split())
POSITIVE_REALS = frozenset('''
encoder_lr head_lr module_lr token_lr temperature gradient_clip_norm
pretraining_encoder_lr pretraining_module_lr pretraining_mix_lr
pretraining_temperature pretraining_gradient_clip_norm
alignment_interaction_scale alignment_logit_scale alignment_temperature
time_prior_initial_alpha profile_kernel_sigma profile_correction_scale
'''.split())
NONNEGATIVE_REALS = frozenset(('weight_decay', 'pretraining_weight_decay'))
FRACTIONS = frozenset('''
warmup_fraction pretraining_warmup_fraction body_head_fraction
unit_drop_title_keep_a unit_drop_body_keep_a unit_drop_title_keep_b unit_drop_body_keep_b
'''.split())
PARAMETER_KEYS = (POSITIVE_INTS | NONNEGATIVE_INTS | POSITIVE_REALS |
                  NONNEGATIVE_REALS | FRACTIONS |
                  {'semantic_mix_init_logit', 'time_scales', 'selection_window', 'selection_order'})
EXPERIMENT_KINDS = dict(E1='training', E2='training', E3='training', E4='training',
                        E5='missingness', E6='windows', E7='efficiency')
MODEL_NAMES = frozenset(('S13', 'R7', 'GenericP', 'Base', 'C3Only', 'S12', 'S14', 'S15', 'SelfWeight'))


def _real(value):
    return type(value) in (int, float) and math.isfinite(value)


def _window(value):
    return value is None or (type(value) is int and value > 0)


def validate_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError('parameters must be an object')
    missing, extra = PARAMETER_KEYS - parameters.keys(), parameters.keys() - PARAMETER_KEYS
    if missing or extra:
        raise ValueError(f'Parameter fields: missing={sorted(missing)}, unknown={sorted(extra)}')
    for key in POSITIVE_INTS | NONNEGATIVE_INTS:
        value = parameters[key]
        if type(value) is not int or value < (1 if key in POSITIVE_INTS else 0):
            raise ValueError(f'{key} must be an integer in its permitted range')
    for key in POSITIVE_REALS | NONNEGATIVE_REALS | FRACTIONS | {'semantic_mix_init_logit'}:
        value = parameters[key]
        if not _real(value):
            raise ValueError(f'{key} must be finite and numeric')
        if key in POSITIVE_REALS and value <= 0:
            raise ValueError(f'{key} must be positive')
        if key in NONNEGATIVE_REALS and value < 0:
            raise ValueError(f'{key} must be nonnegative')
        if key in FRACTIONS and not 0 <= value <= 1:
            raise ValueError(f'{key} must lie between zero and one')
    if parameters['pretraining_batch'] < 2:
        raise ValueError('pretraining_batch must support paired-view contrastive learning')
    if parameters['max_units'] < 3 or parameters['profile_kernel_count'] < 2:
        raise ValueError('Unit and kernel budgets cannot represent the requested structure')
    if parameters['max_length'] <= parameters['title_max_length'] + 4:
        raise ValueError('max_length must leave room for special tokens and body tokens')
    if sum(parameters[key] for key in NONNEGATIVE_INTS) == 0:
        raise ValueError('At least one negative is required')
    if parameters['selection_window'] != 'full' and not _window(parameters['selection_window']):
        raise ValueError('selection_window must be full, null or a positive number of days')
    if parameters['selection_order'] != ['rr10', 'rr5', 'mrr']:
        raise ValueError('Checkpoint selection uses RR@10, RR@5, then full MRR')
    scales = parameters['time_scales']
    if not isinstance(scales, list) or not scales or any(not _real(x) or x <= 0 for x in scales):
        raise ValueError('time_scales must be a nonempty list of positive finite numbers')


def validate_design(experiment, design):
    if experiment not in EXPERIMENT_KINDS or not isinstance(design, dict):
        raise ValueError('Unknown experiment or invalid design')
    required = {'experiment', 'kind', 'models', 'projects', 'windows'}
    extra_fields = {'E5': {'levels', 'repeats', 'mask_seed'},
                    'E7': {'timing_repeats', 'warmup_queries'}}.get(experiment, set())
    missing = (required | extra_fields) - design.keys()
    extra = design.keys() - required - extra_fields - {'title'}
    if missing or extra:
        raise ValueError(f'Design fields: missing={sorted(missing)}, unknown={sorted(extra)}')
    if design['experiment'] != experiment or design['kind'] != EXPERIMENT_KINDS[experiment]:
        raise ValueError('Experiment identity or execution kind mismatch')
    for key in ('models', 'projects'):
        values = design[key]
        if (not isinstance(values, list) or not values or
                any(not isinstance(x, str) or not x or '/' in x or '\\' in x or x in ('.', '..') for x in values) or
                len(values) != len(set(values))):
            raise ValueError(f'{key} must contain distinct identifiers')
    if not set(design['models']) <= MODEL_NAMES:
        raise ValueError('Unknown model variant')
    windows = design['windows']
    if (not isinstance(windows, list) or not windows or any(not _window(x) for x in windows) or
            len(windows) != len(set(windows))):
        raise ValueError('windows must contain distinct positive day counts or null for full history')
    if experiment not in ('E4', 'E6') and len(windows) != 1:
        raise ValueError('This experiment requires one evaluation window')
    if experiment == 'E5':
        for key, maximum in (('levels', 100), ('repeats', None)):
            values = design[key]
            if (not isinstance(values, list) or not values or
                    any(type(x) is not int or x < 0 or (maximum is not None and x > maximum) for x in values) or
                    len(values) != len(set(values))):
                raise ValueError(f'Invalid {key}')
        if type(design['mask_seed']) is not int or not 0 <= design['mask_seed'] < 2**32:
            raise ValueError('mask_seed must be a uint32 integer')
    if experiment == 'E7':
        if type(design['timing_repeats']) is not int or design['timing_repeats'] < 1:
            raise ValueError('timing_repeats must be positive')
        if type(design['warmup_queries']) is not int or design['warmup_queries'] < 0:
            raise ValueError('warmup_queries must be nonnegative')
