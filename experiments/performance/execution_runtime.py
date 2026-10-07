"""Explicit, provenance-tracked execution changes; no scientific options."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACCEPTANCE = ROOT / 'checks' / 'prefetched-ddp.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def execution_provenance(policy, acceptance_path=None):
    """Hash exactly the active runtime implementation and accepted candidate."""
    if policy not in ('standard', 'optimized_v5'):
        raise ValueError('Unknown execution policy')
    sources = {'execution_runtime.py': file_sha256(__file__)}
    acceptances = {}
    if policy == 'optimized_v5':
        candidate = ROOT / 'prefetched_replay.py'
        sources[candidate.name] = file_sha256(candidate)
        path = Path(acceptance_path) if acceptance_path else ACCEPTANCE
        report = json.loads(path.read_text())
        if report.get('passed') is not True:
            raise ValueError('Optimized execution requires a passed native acceptance')
        baseline_path = ROOT / 'baselines' / 'optimized_v5_baseline.json'
        baseline = json.loads(baseline_path.read_text())
        ranks = report.get('ranks', [])
        if report.get('global_batch') != 16 or len(ranks) != 2:
            raise ValueError('Native acceptance must use two ranks and batch 16')
        for rank in ranks:
            if rank.get('source_sha256') != baseline['source_sha256']:
                raise ValueError('Native acceptance did not use the frozen baseline')
            if rank.get('candidate_sha256') != sources[candidate.name]:
                raise ValueError('Execution candidate differs from native acceptance')
            if (rank.get('global_batch') != 16 or rank.get('accumulation_per_rank') != 8
                    or rank.get('state_dict_compatible') is not True
                    or rank.get('production_source_unchanged') is not True):
                raise ValueError('Native execution/state compatibility check failed')
            if not all(rank.get('parity', {}).get(key) is True for key in (
                    'explicit_rng_equal', 'cuda_rng_equal', 'torch_rng_equal')):
                raise ValueError('Native RNG parity check failed')
            if len(rank.get('runs', [])) != 2 or not all(
                    run.get('ddp_gradients_identical') is True and
                    run.get('ddp_parameters_identical') is True for run in rank['runs']):
                raise ValueError('Native DDP consistency check failed')
        sources['baselines/optimized_v5_baseline.json'] = file_sha256(baseline_path)
        acceptances[ACCEPTANCE.name] = file_sha256(path)
    return dict(execution_policy=policy, execution_source_sha256=sources,
                execution_acceptance_sha256=acceptances)


def build_model(paths, device, phase, variant, policy):
    """Optimize only dynamics; all later stages retain JointSystem behavior."""
    from experiments.paired_interventions.system import JointSystem
    if policy not in ('standard', 'optimized_v5'):
        raise ValueError('Unknown execution policy')
    if policy == 'standard' or phase != 'dynamics':
        return JointSystem(paths, device, phase, variant)
    from .prefetched_replay import PrefetchedReplayJointSystem

    class OptimizedDynamics(PrefetchedReplayJointSystem):
        def forward(self, branch, *args, **kwargs):
            # The fixed validation function already supplies CUDA branches.
            # Preserve its original path instead of preparing targets from a
            # GPU tensor or changing its protocol to fit the execution cache.
            if '_prefetched_replay' not in branch:
                return JointSystem.forward(self, branch, *args, **kwargs)
            return super().forward(branch, *args, **kwargs)

    model = OptimizedDynamics(paths, device, phase, variant)
    model.upstream.geometry.__class__ = bounded_geometry_class()
    return model


def bounded_geometry_class():
    """Retain original checkpointing for dense scenes; never drop targets."""
    from .prefetched_replay import GeometryWithoutRecompute
    from experiments.paper_experiments.agcr_exp.future_prediction import SurfaceGeometry

    class BoundedGeometry(GeometryWithoutRecompute):
        def forward(self, *args, **kwargs):
            states = kwargs['states'] if 'states' in kwargs else args[3]
            if states.shape[1] > 8:
                return SurfaceGeometry.forward(self, *args, **kwargs)
            return GeometryWithoutRecompute.forward(self, *args, **kwargs)

    return BoundedGeometry


def prepare_training_branch(branch, update, phase, policy):
    if policy not in ('standard', 'optimized_v5'):
        raise ValueError('Unknown execution policy')
    if policy == 'optimized_v5' and phase == 'dynamics':
        from .prefetched_replay import prepare_replay_cpu
        return prepare_replay_cpu(branch, update)
    return branch
