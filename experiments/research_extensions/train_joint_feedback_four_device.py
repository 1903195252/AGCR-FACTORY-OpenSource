"""Independent actual-training-case admission and final-block-VAE matched training.

The frozen worker still owns numerical objectives, Adam, checkpoints and RNG.
Only its explicit admission and execution scopes are extended in this process.
The prior AGV/17031 admission cannot authorize this entrypoint.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

from . import train_joint_feedback as original
from .region_probe import sha256
from .joint_feedback_candidate import REQUIRED_HOOKS

GATE_SCHEMA = 'agcr-native-joint-step-four-device-admission-1'
MODULE = 'experiments.research_extensions.train_joint_feedback_four_device'
TAIL_POLICY = 'frozen_vae_up_block_3_and_output_head_on_reserved_tail_v1'
SOURCES = tuple(name for name in original.THREE_DEVICE_LAYOUT192_SOURCES
                if name != 'probe_joint_adam_three_device_layout192.py') + (
    'probe_joint_adam_training_case.py', 'probe_joint_adam_vae_four_device.py',
    'vae_final_block_device_residency.py', 'train_joint_feedback.py', 'train_candidate.py')
BASE_RECIPE = original.execution_recipe
BASE_SCOPE = original.joint_execution_scope
BASE_DEVICES = original.execution_devices


def case_contract(a):
    paths = json.loads(Path(a.paths).read_text())
    data, index, families = original.training_families(a, paths)
    selected = families[original.branch_index(0, len(families), 17033)]
    if {(r['world'], r['camera']) for r in selected['records']} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError('The actual first TRAIN family must have four complete branches')
    return dict(seed=17033, generator_seed=117036, training_cursor=0,
        family_stream='seeded_train_evidence_epoch_permutation',
        training_panel_sha256=data['panel_sha256'], index_sha256=data['index_sha256'],
        training_family_ids=[row['family_id'] for row in families], family_id=selected['family_id'],
        reference_initial_checkpoint_sha256=sha256(a.reference_initial_checkpoint),
        source_files={str(Path(index['root']) / row['path']): row['sha256'] for row in selected['records']})


def validate_gate(report, receipt, report_sha, weights, expected_case):
    expected = dict(schema=GATE_SCHEMA, completed=True, execution_completed=True,
        gradient_route_passed=True, all_parameter_gradients_finite=True,
        optimizer_steps=1, weight_files_written=False, parameter_versions_unchanged=False,
        frozen_parameter_versions_unchanged=True, full_base_flow_depth_and_supervised_replay=True,
        split='train', sampling_steps=50, gradient_tail_steps=2, camera_paths=2,
        future_truth_conditions=False, neutralize_geometry_omega=True,
        neutralize_renderer_omega=True, initial_model_exact_match=True, initial_rng_exact_match=True,
        trainable_scope='upstream_condition_adapters_gate_frozen_lora_depth_reader')
    expected.update(expected_case)
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise ValueError('The exact failed first-training case has not passed a complete Adam admission')
    if (report.get('weight_sha256') != weights or receipt.get('schema') != GATE_SCHEMA
            or receipt.get('report_sha256') != report_sha or receipt.get('completed') is not True
            or type(receipt.get('optimizer_steps')) is not int or receipt['optimizer_steps'] != 1):
        raise ValueError('Actual-case native receipt or original parents differ')
    changed = report.get('trainable_parameter_versions_changed', {})
    hooks = report.get('gradients', {}).get('tensor_hooks', {})
    if (any(type(changed.get(k)) is not int or changed[k] <= 0 for k in ('upstream', 'condition_adapters_gate'))
            or any(type(hooks.get(k)) not in (int, float) or not math.isfinite(hooks[k]) or hooks[k] <= 0 for k in REQUIRED_HOOKS)):
        raise ValueError('Actual-case admission needs both updated groups and all four feedback hooks')
    sources = report.get('source_sha256', {})
    if not (set(SOURCES) | {'joint_feedback_candidate.py', 'joint_parameter_scope.py'}) <= set(sources):
        raise ValueError('Actual-case execution provenance is incomplete')
    for name, digest in sources.items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Actual-case admission source changed: ' + name)
    return report


def load_gate(path, weights, a):
    path = Path(path)
    report = json.loads(path.read_text())
    receipt = json.loads(path.with_name('completion.json').read_text())
    return validate_gate(report, receipt, sha256(path), weights, case_contract(a))


def execution_recipe(report):
    # Validate the actual four-card receipt before passing a private structural
    # compatibility view to the legacy VAE-prefix validator. The persisted
    # report, identity and accounting always retain the real four-card topology.
    for key in ('execution_depth_residency', 'execution_vae_residency'):
        if type(report.get(key, {}).get('gpu_count')) is not int or report[key]['gpu_count'] != 4:
            raise ValueError('Actual four-card placement was not admitted')
    admission = report.get('execution_device_admission', {})
    if type(admission.get('expected_physical_gpu_count')) is not int or admission['expected_physical_gpu_count'] != 4:
        raise ValueError('Actual native admission must reserve all four cards')
    legacy = copy.deepcopy(report)
    legacy['execution_depth_residency']['gpu_count'] = 3
    legacy['execution_vae_residency']['gpu_count'] = 3
    legacy['execution_device_admission']['expected_physical_gpu_count'] = 3
    with patch.object(original, 'THREE_DEVICE_LAYOUT192_SOURCES', SOURCES):
        recipe = BASE_RECIPE(legacy)
    if not recipe['capacity192'] or not recipe['three_device']:
        raise ValueError('Four-card execution requires the measured complete192GiB VAE-prefix recipe')
    recipe['depth'] = report['execution_depth_residency']
    recipe['vae'] = report['execution_vae_residency']
    recipe['four_device'] = True
    expected = dict(enabled=True, policy=TAIL_POLICY, first_tail_block=3,
        source_device='cuda:2', tail_device='cuda:3', dtype_preserved=True,
        input_gradients_preserved=True,
        original_decoder_forward_retained=True, feature_hooks_retained=True)
    tail = report.get('execution_vae_tail_residency', {})
    if any(type(tail.get(k)) is not type(v) or tail.get(k) != v for k, v in expected.items()):
        raise ValueError('The exact frozen VAE tail placement was not admitted')
    records = report.get('vae_tail_residency_records', [])
    expected_record = dict(enabled=True, policy=TAIL_POLICY, first_tail_block=3,
        source_device='cuda:2', tail_device='cuda:3', parameter_objects_preserved=True,
        prefix_encoder_scaling_unchanged=True, cache_objects_preserved=True,
        original_decoder_forward_retained=True, feature_hooks_retained=True,
        dtype_changed=False, gradients_detached=False, random_operations_added=False, prefix_block_indices=[0, 1, 2], moved_block_indices=[3])
    if not isinstance(records, list) or not records or any(
            not isinstance(row, dict)
            or any(type(row.get(k)) is not type(v) or row.get(k) != v for k, v in expected_record.items())
            or any(type(row.get(k)) is not int or row[k] <= 0 for k in ('boundary_calls', 'input_tensor_bytes', 'tail_parameter_bytes'))
            for row in records):
        raise ValueError('Actual native tail execution and restoration evidence is required')
    recipe['tail'] = tail
    # The unchanged worker records this entire dictionary in its identity;
    # resume cannot silently cross between old and final-block execution.
    recipe['offload'] = dict(recipe['offload'], vae_tail_execution=tail,
        actual_training_case={k: report[k] for k in ('seed', 'generator_seed', 'training_cursor',
            'family_id', 'training_panel_sha256', 'index_sha256', 'reference_initial_checkpoint_sha256')})
    recipe['sources'] = sorted(set(recipe['sources']) | {Path(__file__).name})
    return recipe


def execution_devices(a, recipe):
    devices = BASE_DEVICES(a, recipe)
    tail = getattr(a, 'vae_tail_device', None)
    if not recipe.get('four_device') or tail != recipe['tail']['tail_device']:
        raise ValueError('Explicit reserved tail device differs from admitted placement')
    devices.append(tail)
    if len(devices) != 4 or len({int(d[5:]) for d in devices}) != 4:
        raise ValueError('Four distinct physical allocations are required')
    return devices


@contextmanager
def execution_scope(model, recipe, depth_device, vae_device=None, **kwargs):
    from . import vae_final_block_device_residency as placement
    device = recipe['tail']['tail_device']
    if not kwargs.get('_test_cpu', False):
        import torch
        torch.cuda.reset_peak_memory_stats(device)
    with BASE_SCOPE(model, recipe, depth_device, vae_device, **kwargs) as active:
        with patch.object(placement, 'POLICY', TAIL_POLICY), placement.frozen_vae_final_block_on_device(
                model, device, _test_cpu=kwargs.get('_test_cpu', False)) as tail:
            active['vae_placed']['tail_residency'] = tail
            try:
                yield active
            finally:
                if not kwargs.get('_test_cpu', False):
                    torch.cuda.synchronize(device)
                    tail['execution_gpu_peak'] = dict(device=device,
                        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)


def account_joint_attempt(output, attempt, gpu_count):
    if gpu_count != 4:
        raise ValueError('This independent four-card run must charge exactly four allocations')
    output, attempt = Path(output), Path(attempt)
    path = attempt/'report.json'; report = json.loads(path.read_text())
    report.update(allocated_gpus=4, allocated_gpu_hours=report['wall_seconds']*4/3600.,
        gpu_accounting='admitted_devices_times_worker_wall; outer shared GPU ledger is authoritative')
    original.atomic_json(path, report); original.atomic_json(output/'status.json', report)
    previous = [json.loads(p.read_text()) for p in (output/'attempts').glob('*/report.json')]
    original.atomic_json(output/'usage.json', dict(allocated_gpu_hours=sum(p.get('allocated_gpu_hours', 0.) for p in previous),
        attempts=len(previous), accounting='four_admitted_GPUs_times_worker_wall_including_load_save',
        outer_shared_GPU_ledger_authoritative=True))


def arguments(cli):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-initial-checkpoint', required=True)
    parser.add_argument('--vae-tail-device', required=True)
    extra, remaining = parser.parse_known_args(cli)
    args = original.arguments(remaining)
    args.reference_initial_checkpoint = extra.reference_initial_checkpoint
    args.vae_tail_device = extra.vae_tail_device
    if (args.device, args.depth_device, args.vae_device, args.vae_tail_device) != ('cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'):
        raise ValueError('Explicit primary0/depth1/prefix2/tail3 execution is required')
    return args


def main(argv=None):
    cli = list(sys.argv[1:] if argv is None else argv)
    a = arguments(cli)
    def admitted(path, weights):
        return load_gate(path, weights, a)
    with patch.object(original, 'load_joint_step_gate', admitted), \
            patch.object(original, 'execution_recipe', execution_recipe), \
            patch.object(original, 'joint_execution_scope', execution_scope), \
            patch.object(original, 'execution_devices', execution_devices), \
            patch.object(original, 'account_joint_attempt', account_joint_attempt):
        if a.worker_report:
            import torch
            if torch.cuda.get_device_properties(a.vae_tail_device).total_memory < 30*2**30:
                raise ValueError('Reserved tail device must have at least30GiB')
            torch.cuda.set_per_process_memory_fraction(a.memory_fraction, a.vae_tail_device)
            return original.worker(a)
        weights = {key: sha256(path) for key, path in dict(renderer_checkpoint=a.checkpoint,
            state_checkpoint=a.state_checkpoint, future_checkpoint=a.future_checkpoint).items()}
        native = admitted(a.native_probe_report, weights)
        recipe = execution_recipe(native)
        count = len(original.execution_devices(a, recipe))
        out = Path(a.output).resolve(); paths = json.loads(Path(a.paths).read_text())
        original.parent_read_only_guard(out, paths, a.checkpoint, a.state_checkpoint, a.panel)
        for source in (Path(a.future_checkpoint).resolve().parent, Path(a.native_probe_report).resolve().parent,
                       Path(a.reference_initial_checkpoint).resolve().parent):
            if out == source or source in out.parents or out in source.parents:
                raise ValueError('Independent output overlaps initializer, failed run or admission')
        out.mkdir(parents=True, exist_ok=True)
        with (out / '.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.resume:
                if not (out / 'last.pt').is_file():
                    raise FileNotFoundError('No exact final-block checkpoint to resume')
            elif any(p.name != '.lock' for p in out.iterdir()):
                raise FileExistsError('Use a fresh final-block training directory')
            attempt = out / 'attempts' / ('%d_%d' % (time.time_ns(), original.os.getpid()))
            attempt.mkdir(parents=True)
            command = [sys.executable, '-m', MODULE, *cli,
                '--worker-report', str(attempt / 'report.json'), '--watchdog-stop', str(attempt / 'STOP')]
            code = original.supervise(command, out, attempt, a.max_seconds, 120, a.stop_file)
            original.account_joint_attempt(out, attempt, count)
            return code


if __name__ == '__main__':
    raise SystemExit(main())
