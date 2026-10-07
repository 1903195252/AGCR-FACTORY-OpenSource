"""Strict Base artifact preflight, then unchanged independent evaluation code.

The short Base schedule is verified against its own registered protocol. This
does not relax a Pro training/quality check or replace an independent reader.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys

import torch

from .downstream import (SCHEMA, settings, source_hashes, frontend_modality,
                         frozen_digest, holdout_gate, validate_frontend_payload)
from experiments.downstream_tracking.data import CLASS_MAP, ROLES, sha256
from experiments.downstream_tracking.train import atomic
from experiments.paper_experiments.agcr_exp.protocol import digest


CLASS_NAMES = ['worker', 'forklift', 'agv', 'uav']
MODULES = dict(calibrate='experiments.downstream_tracking.calibrate',
               readout='experiments.formal_evaluation.readout',
               annotate='experiments.downstream_tracking.annotations',
               accept='experiments.downstream_tracking.annotations',
               downstream='experiments.formal_evaluation.downstream')


def validate_reader(saved, index, protocol, protocol_sha, role, index_sha=None):
    """Only validated selected checkpoints from a completed Base run qualify."""
    identity = saved.get('identity', {})
    if identity.get('schema') != SCHEMA or identity.get('role') != role:
        raise ValueError('Wrong Base independent reader schema/role')
    if (identity.get('development_only', True) or not saved.get('completed') or
            not saved.get('run_completed')):
        raise ValueError('Incomplete/development Base reader')
    if identity.get('protocol_sha256') != protocol_sha:
        raise ValueError('Reader protocol mismatch')
    if identity.get('split_sha256') != index['split_sha256']:
        raise ValueError('Reader split mismatch')
    if index_sha and identity.get('index_sha256') != index_sha:
        raise ValueError('Reader ordinary index mismatch')
    if identity.get('source_sha256') != source_hashes():
        raise ValueError('Reader trained source differs from current implementation')
    if identity.get('generator_weights_used') is not False:
        raise ValueError('Independent readers cannot reuse generator weights')
    training = sorted(set(identity.get('training_episodes', [])))
    if not training or identity.get('training_partition_sha256') != digest(training):
        raise ValueError('Invalid reader training partition signature')
    scope = identity.get('scope')
    stage = 'frontend' if scope == 'frontend' else 'identity'
    cfg, schedule = settings(protocol, stage)
    if identity.get('schedule') != schedule:
        raise ValueError('Reader schedule differs from preregistered Base schedule')
    if saved.get('training_completed_steps') != schedule['steps']:
        raise ValueError('The complete registered Base schedule has not run')
    if not 1 <= saved.get('step', 0) <= schedule['steps']:
        raise ValueError('Invalid selected checkpoint step')
    if identity.get('modality') != frontend_modality(role):
        raise ValueError('Wrong role-specific reader modality')
    if scope == 'frontend':
        if identity.get('input_labels_used') is not False:
            raise ValueError('Ground-truth labels cannot enter the image-only frontend')
        if identity.get('frontend_gate') != cfg['frontend_gate']:
            raise ValueError('Changed preregistered frontend gate')
        validate_frontend_payload(saved, index, role, protocol_sha)
    elif scope == 'identity_head' and role == 'downstream':
        report = saved.get('holdout_validation', {})
        if (not report.get('complete') or not report.get('passed') or
                report.get('step') != saved['step'] or
                not math.isfinite(report.get('objective', float('nan'))) or
                report.get('positive_anchors', 0) < cfg['frontend_gate']['min_positive_anchors']):
            raise ValueError('Identity head lacks valid heldout selection')
        validation = set(report.get('episodes', []))
        if (not validation or validation.intersection(training) or
                any(index['roles'].get(e) != 'val' for e in validation) or
                any(index['roles'].get(e) not in ('train', 'augmentation_train') for e in training)):
            raise ValueError('Identity train/validation role leakage')
        front = set(identity.get('frontend_training_episodes', []))
        if not front or any(index['roles'].get(e) != ROLES['downstream'] for e in front):
            raise ValueError('Shared frontend training partition leakage')
        if (identity.get('trainable_parameters') != ['identity.weight', 'identity.bias'] or
                saved.get('frozen_frontend_sha256') != frozen_digest(saved['model'])):
            raise ValueError('The shared visual frontend changed during adaptation')
        if not holdout_gate(saved.get('frontend_validation', {}), cfg['frontend_gate']):
            raise ValueError('Shared visual frontend did not pass its holdout gate')
        front_validation = set(saved['frontend_validation'].get('episodes', []))
        if (not front_validation or front_validation.intersection(front) or
                any(index['roles'].get(e) != 'val' for e in front_validation)):
            raise ValueError('Shared frontend validation population leaked')
    else:
        raise ValueError('Unsupported Base reader scope')
    return identity


def validate_association(row, identity, checkpoint_sha, index, shared_r=False):
    if (row.get('development_only', True) or row.get('fit_role') != 'val' or
            row.get('split') != 'val' or row.get('split_sha256') != index['split_sha256'] or
            row.get('modality') != identity['modality']):
        raise ValueError('Association must use native validation and the same modality/split')
    episodes = row.get('episodes', [])
    if not episodes or any(index['roles'].get(e) != 'val' for e in episodes):
        raise ValueError('Association calibration episode leakage')
    if shared_r:
        if row.get('seed') != identity['seed'] or row.get('base_regime') != 'R':
            raise ValueError('Use the same-seed R-only validation calibration for every regime')
    elif row.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('Association belongs to another reader checkpoint')


def evaluation_contract():
    """Definitions fixed in code before any test predictions are inspected."""
    if CLASS_MAP != {3: 1, 4: 2, 13: 3, 14: 4}:
        raise ValueError('Unexpected native class mapping')
    return dict(class_names_zero_based=CLASS_NAMES, native_to_evaluation_class=CLASS_MAP,
                detector_labels='zero_based', exported_labels='one_based',
                video_reader_modality='rgb', annotation_downstream_modality='rgbd',
                association_selection='validation_global_IDF1',
                association_grid=dict(appearance_weight=[.3, .6, .8],
                                      mahalanobis_gate=[9.21, 16.27], confidence=[.3, .5]),
                downstream_primary=['R3', 'IDF1'], full_continuous_episode_frames=100,
                video_population='all_registered_test_families_four_cells_eight_draws',
                primary_draw_reduction='mean', depth_unit='mm')


def evaluation_source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__)]
    for folder in ('formal_evaluation', 'downstream_tracking'):
        files.extend(p for p in (root/folder).glob('*.py')
                     if not p.name.startswith(('test_', 'check_')))
    files.extend([root/'paper_experiments/agcr_exp/downstream.py',
                  root/'paper_experiments/agcr_exp/protocol.py'])
    return {str(p.relative_to(root)): sha256(p) for p in sorted(set(files))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('command', choices=tuple(MODULES))
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--ordinary-index', required=True)
    own, forwarded = parser.parse_known_args(argv)
    detail = argparse.ArgumentParser(add_help=False)
    for key in ('checkpoint', 'index', 'association', 'input', 'output', 'thresholds', 'mode', 'modality'):
        detail.add_argument('--' + key)
    detail.add_argument('--development', action='store_true')
    detail.add_argument('--limit', type=int, default=64)
    options, _ = detail.parse_known_args(forwarded)
    if options.development or not options.checkpoint or not options.output:
        raise ValueError('Base formal preflight requires a checkpoint/output; development mode is separate')
    read = lambda p: json.loads(Path(p).read_text())
    protocol, index = read(own.protocol), read(own.ordinary_index)
    if protocol.get('status') != 'frozen' or index.get('development_only', True):
        raise ValueError('Frozen Base protocol and audited ordinary index required')
    saved = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    role = ('video_evaluation' if own.command == 'readout' else 'annotation'
            if own.command in ('annotate', 'accept') or options.mode == 'annotation'
            else 'downstream' if own.command == 'downstream' else saved['identity']['role'])
    identity = validate_reader(saved, index, protocol, sha256(own.protocol), role, sha256(own.ordinary_index))
    checkpoint_sha = sha256(options.checkpoint)
    if own.command in ('calibrate', 'downstream') and Path(options.index).resolve() != Path(own.ordinary_index).resolve():
        raise ValueError('Calibration/downstream must use the trained audited ordinary index')
    if options.modality and options.modality != identity['modality']:
        raise ValueError('Requested modality differs from the independent reader')
    if options.association:
        validate_association(read(options.association), identity, checkpoint_sha, index,
                             shared_r=own.command == 'downstream')
    if own.command == 'calibrate':
        registered = protocol.get('evaluation', {}).get('calibration_panel', 64)
        if options.limit != registered or registered < 64:
            raise ValueError('Use the fixed calibration panel (at least 64), not a test-tuned limit')
        eligible = (sorted({r['episode'] for r in index['windows'] if r['role'] == 'val'})
                    if options.mode == 'association' else
                    [r['uid'] for r in index['windows'] if r['role'] == 'val' and r['history_covered']])
        if len(eligible) < registered:
            raise ValueError('Insufficient native validation population for calibration')
    if own.command == 'annotate':
        document = read(options.input)
        if (not document.get('completed') or document.get('protocol_sha256') != sha256(own.protocol) or
                document.get('split_sha256') != index['split_sha256'] or
                not document.get('downstream_augmentation')):
            raise ValueError('Annotation requires a completely merged, registered Base proposal corpus')
        if any(index['roles'].get(r['episode']) != 'train' for r in document['clips']):
            raise ValueError('Generated proposal source episode leakage')
    if own.command == 'accept':
        document, thresholds = read(options.input), read(options.thresholds)
        if not document.get('completed') or document.get('annotation_reader_sha256') != checkpoint_sha:
            raise ValueError('Acceptance requires complete independent annotation')
        if (thresholds.get('annotation_reader_sha256') != checkpoint_sha or
                thresholds.get('split_sha256') != index['split_sha256'] or
                thresholds.get('fit_role') != 'val' or thresholds.get('development_only', True)):
            raise ValueError('Acceptance thresholds must be fixed on this reader and validation split')
    module = importlib.import_module(MODULES[own.command])
    contract = dict(schema='agcr-base48-evaluation-contract-1', command=own.command,
                    protocol_sha256=sha256(own.protocol), ordinary_index_sha256=sha256(own.ordinary_index),
                    checkpoint_sha256=checkpoint_sha, reader_identity=identity,
                    definitions=evaluation_contract(),
                    source_sha256=evaluation_source_hashes())
    out = Path(options.output)
    audit = out/'base-contract.json' if own.command in ('readout', 'downstream') else Path(str(out)+'.base-contract.json')
    if audit.exists() and read(audit) != json.loads(json.dumps(contract)):
        raise ValueError('Changed registered Base evaluation identity')
    atomic(audit, contract)
    if own.command in ('annotate', 'accept') and options.mode != own.command:
        raise ValueError('Annotation subcommand/mode mismatch')
    old = sys.argv
    try:
        sys.argv = [module.__file__] + forwarded
        module.main()
    finally:
        sys.argv = old
    if own.command == 'calibrate' and options.mode == 'association' and out.exists():
        # The matched R head calibrates association once for all identity groups.
        row = read(out)
        row.update(base_regime=identity.get('regime'), base_protocol_sha256=sha256(own.protocol),
                   base_evaluation_contract_sha256=sha256(audit))
        atomic(out, row)


if __name__ == '__main__':
    main()
