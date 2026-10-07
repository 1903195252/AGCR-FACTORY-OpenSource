"""Typed admission of TRAIN-only presence refinements into unchanged consumers.

Only this process sees an in-memory compatibility view. Derived files retain
their separate schema; original state checkpoints and scientific source hashes
are never rewritten. New renderer checkpoints and prediction contracts carry
the complete amendment lineage. Validation evaluation is allowed before a gate;
renderer initialization and generation require a newly measured passing gate.
"""
import argparse
from contextlib import contextmanager
import copy
import importlib
import json
import os
from pathlib import Path
import sys

import torch

from experiments.paired_interventions.data import file_sha256
from . import presence_refit
from .train import SCHEMA as TRAIN_SCHEMA, SAMPLE_SEEDS, source_hashes, audit_index_ancestry


SCHEMA = 'agcr-base-presence-admission-1'
TARGETS = tuple('experiments.base48.' + name for name in
                ('state_eval', 'train', 'sample', 'augment'))


def option(arguments, name):
    """Read an unchanged consumer option, including argparse's --key=value form."""
    for index, argument in enumerate(arguments):
        if argument == name:
            return arguments[index + 1]
        if argument.startswith(name + '='):
            return argument.split('=', 1)[1]
    return None


def verified_gate_index(index_path, contract):
    """Apply state_eval's registered-index/original-source ancestry rule unchanged."""
    ancestry = audit_index_ancestry(index_path)
    requested = contract.get('index_sha256')
    if contract.get('ancestry') != ancestry:
        raise ValueError('Gate index ancestry differs from its verified current source')
    if requested == file_sha256(index_path):
        return Path(index_path)
    if (not ancestry or requested != ancestry.get('source_index_sha256')
            or not ancestry.get('train_val_identical')):
        raise ValueError('Gate index is neither registered nor its verified original source')
    return Path(ancestry['source_index_path'])


class RefinedRuntime:
    def __init__(self, module, panel_path, gate_paths=(), arguments=()):
        if module not in TARGETS:
            raise ValueError('Only explicit Base consumer modules are supported')
        self.module = module
        self.arguments = list(arguments)
        self.panel_path = Path(panel_path).resolve()
        self.panel = json.loads(self.panel_path.read_text())
        self.panel_sha = file_sha256(self.panel_path)
        self.runtime_sha = file_sha256(__file__)
        self.helper_sha = file_sha256(presence_refit.__file__)
        self.base_sources = source_hashes()
        rows = self.panel.get('selected_families', [])
        if (self.panel.get('schema') != presence_refit.SCHEMA + '-panel'
                or self.panel.get('split') != 'train'
                or self.panel.get('configuration') != presence_refit.CONFIG
                or self.panel.get('selection_uses_validation_or_test') is not False
                or len(rows) != 64 or len({row['family_id'] for row in rows}) != 64
                or any(row.get('split') != 'train' for row in rows)):
            raise ValueError('A fixed, unique, TRAIN-only 64-family refit panel is required')
        self.gate_paths = [Path(path).resolve() for path in gate_paths]
        self.amendments = {}
        self.admissions = {}
        self._validated = {}
        self._raw_load = torch.load
        self._raw_save = torch.save
        named_inputs = {'state_eval': ('--checkpoint',), 'train': ('--init', '--resume'),
                        'sample': ('--renderer-checkpoint', '--state-checkpoint'),
                        'augment': ('--renderer-checkpoint', '--state-checkpoint')}
        self.required_inputs = {self._path(value) for key in named_inputs[module.rsplit('.', 1)[1]]
                                for value in [option(arguments, key)] if value}
        if module.endswith('.train') and option(arguments, '--phase') not in ('depth', 'video'):
            raise ValueError('Presence refinement wrapper supports only depth/video child training')

    def _path(self, path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            raise ValueError('Derived checkpoints require a filesystem path for immutable provenance')
        return Path(os.fsdecode(path)).resolve()

    def _derive(self, payload, path):
        path = self._path(path)
        digest = file_sha256(path)
        refit = payload.get('refit', {})
        parent_path = self._path(refit['parent_checkpoint_path'])
        parent_sha = file_sha256(parent_path)
        parent = self._raw_load(parent_path, map_location='cpu', weights_only=False)
        presence_refit.validate_derived(payload, parent, parent_sha)
        identity = parent['identity']
        if (identity.get('schema') != TRAIN_SCHEMA or identity.get('phase') != 'state'
                or identity.get('development_only') or not parent.get('completed')
                or not parent.get('validation_best')
                or identity.get('source_sha256') != self.base_sources
                or refit.get('runtime_source_sha256') != self.base_sources
                or refit.get('helper_sha256') != self.helper_sha
                or refit.get('panel_sha256') != self.panel_sha
                or refit.get('family_index_sha256') != self.panel.get('family_index_sha256')
                or identity.get('family_index_sha256') != self.panel.get('family_index_sha256')
                or refit.get('training_families') != [r['family_id'] for r in self.panel['selected_families']]
                or refit.get('training_worlds') != 128
                or [r.get('update') for r in payload.get('refit_updates', [])] != list(range(1, 257))):
            raise ValueError('Refinement panel, parent, source or completed-update provenance differs')
        # Reproduce the TRAIN-only panel from the original registered family index.
        presence_refit.validate_panel(self.panel, identity['paths']['families'])
        amendment = dict(schema=SCHEMA, derived_checkpoint_path=str(path),
            derived_checkpoint_sha256=digest, parent_checkpoint_path=str(parent_path),
            parent_checkpoint_sha256=parent_sha, panel_path=str(self.panel_path),
            panel_sha256=self.panel_sha, helper_sha256=self.helper_sha,
            runtime_sha256=self.runtime_sha, base_source_sha256=self.base_sources,
            configuration=copy.deepcopy(presence_refit.CONFIG), head_keys=list(presence_refit.HEAD_KEYS),
            parent_training_step=parent['step'], supplemental_updates=256,
            validation_or_test_used_for_refit=False,
            frozen_parameter_sha256=refit['frozen_parameter_sha256'],
            cache_manifest_sha256=refit['cache_manifest_sha256'],
            cache_completion_sha256=refit['cache_completion_sha256'])
        self._validated[digest] = amendment
        return amendment, parent

    def _gate(self, amendment, identity):
        """Recompute original readiness from this exact derived checkpoint's report."""
        from .state_eval import readiness, validation_panel, aggregate
        for path in self.gate_paths:
            report = json.loads(path.read_text())
            contract = report.get('contract', {})
            if contract.get('checkpoint_sha256') != amendment['derived_checkpoint_sha256']:
                continue
            protocol_path = option(self.arguments, '--protocol')
            if not protocol_path:
                raise ValueError('Renderer admission requires the original --protocol argument')
            protocol = json.loads(Path(protocol_path).read_text())
            thresholds = protocol.get('evaluation', {}).get('state_gate')
            index_path = verified_gate_index(identity['paths']['families'], contract)
            index = json.loads(Path(index_path).read_text())
            rows = [row for row in index['families'] if row['split'] == 'val']
            positions = validation_panel(rows, int(thresholds.get('families', 20)),
                                         int(thresholds.get('seed', 17029))) if thresholds else []
            expected_families = [rows[position]['family_id'] for position in positions]
            expected_cells = {(family, world) for family in expected_families for world in (0, 1)}
            measured_rows = report.get('rows', [])
            actual_cells = {(row.get('family_id'), row.get('world')) for row in measured_rows}
            expected_summary = aggregate(measured_rows)
            expected_gate = readiness(expected_summary, thresholds,
                                      not identity.get('variant_options', {}).get('receiver_only'), True)
            if (report.get('schema') != 'agcr-base48-state-evaluation-1'
                    or not report.get('complete') or report.get('development_only')
                    or report.get('split') != 'val' or not report.get('ready_for_renderer')
                    or contract.get('protocol_sha256') != identity['protocol_sha256']
                    or file_sha256(protocol_path) != identity['protocol_sha256']
                    or contract.get('index_sha256') != file_sha256(index_path)
                    or contract.get('sample_seeds') != SAMPLE_SEEDS
                    or contract.get('development_only') or contract.get('split') != 'val'
                    or contract.get('families') != expected_families
                    or contract.get('scientific_amendments') != [amendment]
                    or contract.get('implementation_sha256') != file_sha256(Path(__file__).with_name('state_eval.py'))
                    or contract.get('numerical_scorer_sha256') != file_sha256(Path(__file__).parents[1]/'formal_evaluation/state_likelihood.py')
                    or actual_cells != expected_cells or len(measured_rows) != len(expected_cells)
                    or any([s.get('sample_seed') for s in row.get('state_samples', [])] != SAMPLE_SEEDS for row in measured_rows)
                    or report.get('summary') != expected_summary
                    or report.get('gate') != expected_gate or not expected_gate.get('passed')):
                raise ValueError('Derived checkpoint readiness gate is incomplete, changed or failed')
            return dict(checkpoint_sha256=amendment['derived_checkpoint_sha256'],
                        gate_path=str(path), gate_sha256=file_sha256(path))
        raise ValueError('A passing new validation gate for the exact derived checkpoint SHA is required')

    def _register(self, amendment, identity):
        digest = amendment['derived_checkpoint_sha256']
        if self.module != 'experiments.base48.state_eval':
            self.admissions[digest] = self._gate(amendment, identity)
        self.amendments[digest] = amendment

    def load(self, path, *args, **kwargs):
        payload = self._raw_load(path, *args, **kwargs)
        if not isinstance(payload, dict):
            return payload
        if payload.get('schema') == presence_refit.SCHEMA + '-checkpoint':
            if option(self.arguments, '--resume') == str(path) or (
                    option(self.arguments, '--resume') and
                    self._path(option(self.arguments, '--resume')) == self._path(path)):
                raise ValueError('Derived head-only refits cannot resume original training; use --init')
            amendment, parent = self._derive(payload, path)
            self._register(amendment, parent['identity'])
            # Truthful compatibility metadata, never a reused numerical validation score.
            return dict(schema=SCHEMA+'-in-memory-state', model=payload['model'],
                identity=copy.deepcopy(parent['identity']), step=parent['step'],
                completed=bool(parent['completed'] and payload['refit']['completed']),
                validation_best=dict(scope='parent_selected_then_TRAIN_only_presence_refit',
                    parent_score_reused=False, derived_quality_requires_new_gate=True),
                parent_training=copy.deepcopy(payload['parent_training']),
                presence_refit=copy.deepcopy(payload['refit']),
                scientific_amendments=[amendment], scientific_admission=self.admission())
        for recorded in payload.get('scientific_amendments', []):
            derived_path = self._path(recorded['derived_checkpoint_path'])
            digest = file_sha256(derived_path)
            if digest not in self._validated:
                original = self._raw_load(derived_path, map_location='cpu', weights_only=False)
                amendment, _ = self._derive(original, derived_path)
            else:
                amendment = self._validated[digest]
            if recorded != amendment:
                raise ValueError('Child checkpoint scientific amendment lineage changed')
            parent = self._raw_load(amendment['parent_checkpoint_path'], map_location='cpu', weights_only=False)
            self._register(amendment, parent['identity'])
        if (isinstance(path, (str, bytes, os.PathLike)) and self._path(path) in self.required_inputs
                and not payload.get('scientific_amendments')):
            raise ValueError('This consumer input requires an explicitly refined checkpoint or recorded child')
        if payload.get('scientific_amendments'):
            child_admission = [copy.deepcopy(self.admissions[item['derived_checkpoint_sha256']])
                for item in sorted(payload['scientific_amendments'], key=lambda row: row['derived_checkpoint_sha256'])]
            if (payload.get('initialization', {}).get('scientific_amendments') != payload['scientific_amendments']
                    or payload.get('scientific_admission') != child_admission):
                raise ValueError('Child checkpoint initialization/gate admission provenance differs')
        return payload

    def lineage(self):
        return [copy.deepcopy(self.amendments[key]) for key in sorted(self.amendments)]

    def admission(self):
        return [copy.deepcopy(self.admissions[key]) for key in sorted(self.admissions)]

    def annotate(self, value):
        if not self.amendments:
            return value
        result = dict(value, scientific_amendments=self.lineage(), scientific_admission=self.admission())
        if 'initialization' in value:
            result['initialization'] = dict(value['initialization'],
                scientific_amendments=self.lineage(), scientific_admission=self.admission())
        return result

    def save(self, payload, *args, **kwargs):
        if (isinstance(payload, dict) and payload.get('identity', {}).get('schema') == TRAIN_SCHEMA
                and payload['identity'].get('phase') in ('depth', 'video')):
            payload = self.annotate(payload)
        return self._raw_save(payload, *args, **kwargs)

    @contextmanager
    def installed(self):
        """All hooks are scoped to this process and restored, including on failure."""
        old_load, old_save = torch.load, torch.save
        torch.load, torch.save = self.load, self.save
        hooks = []
        try:
            module = importlib.import_module(self.module)
            if self.module.endswith('.state_eval'):
                old = module.validate_contract
                def validate(*args, **kwargs):
                    return self.annotate(old(*args, **kwargs))
                hooks.append((module, 'validate_contract', old))
                module.validate_contract = validate
            elif self.module.endswith('.train'):
                old = module.atomic_json
                def atomic(path, value):
                    return old(path, self.annotate(value) if Path(path).name == 'manifest.json' else value)
                hooks.append((module, 'atomic_json', old))
                module.atomic_json = atomic
            else:
                name = 'make_manifest' if self.module.endswith('.sample') else 'make_contract'
                old = getattr(module, name)
                def construct(*args, **kwargs):
                    return self.annotate(old(*args, **kwargs))
                hooks.append((module, name, old))
                setattr(module, name, construct)
            yield module
        finally:
            for module, name, original in reversed(hooks):
                setattr(module, name, original)
            torch.load, torch.save = old_load, old_save


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--' not in arguments:
        raise ValueError('Separate wrapper and unchanged consumer arguments with --')
    separator = arguments.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True, choices=TARGETS)
    parser.add_argument('--refit-protocol', required=True, help='Immutable TRAIN-only PANEL.json')
    parser.add_argument('--gate', action='append', default=[])
    args = parser.parse_args(arguments[:separator])
    forwarded = arguments[separator+1:]
    runtime = RefinedRuntime(args.module, args.refit_protocol, args.gate, forwarded)
    previous = sys.argv
    try:
        sys.argv = [args.module] + forwarded
        with runtime.installed() as module:
            module.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
