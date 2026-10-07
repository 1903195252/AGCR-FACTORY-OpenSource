"""Isolated dynamics candidate: prepare unchanged replay targets on the CPU.

This module does not patch the production modules or change their provenance.
The CPU helper belongs in the existing bounded loader, before device transfer.
Its output contains supervision only, and is never passed into history().
"""
import torch

from experiments.paired_interventions.data import historical_labels, replay_sample
from experiments.paired_interventions.system import JointSystem
from experiments.paired_interventions.video_losses import reference_for_queries, readout_loss
from experiments.paired_interventions.world import query_labels
from experiments.paper_experiments.agcr_exp.future_prediction import SurfaceGeometry


class GeometryWithoutRecompute(SurfaceGeometry):
    """Skip only SurfaceGeometry's checkpoint condition, retaining child modes.

    In the current audited forward, ``self.training`` is consulted exclusively
    to choose checkpoint(block, ...) versus block(...). Child module modes,
    weights, input precision and every mathematical operation remain unchanged.
    The flag is restored even on errors; this does not invoke module.eval().
    """
    def forward(self, *args, **kwargs):
        original = self.training
        self.training = False
        try:
            return super().forward(*args, **kwargs)
        finally:
            self.training = original


def prepare_replay_cpu(branch, update):
    """Return a new branch with the original, bit-preserving pooled targets."""
    def require_cpu(value):
        if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
            raise ValueError('Replay preparation must precede device transfer')
        if isinstance(value, dict):
            for item in value.values():
                require_cpu(item)
    require_cpu(branch)
    if '_prefetched_replay' in branch:
        raise ValueError('A branch already contains prefetched replay targets')
    offset = int(update) % 16
    replay = replay_sample(branch, offset)
    # Reattach the actual history tensor dictionary after device transfer.
    # Duplicating it here would cause two copies of every RGB-D history.
    cache = dict(offset=offset, labels=replay['labels'], auxiliary=replay['auxiliary'])
    return dict(branch, _prefetched_replay=cache)


class PrefetchedReplayJointSystem(JointSystem):
    """Same dynamics objective, with a CPU-prepared replay and shared DINO call.

    No weights, precision, loss denominators, seeds, or reference conditions
    change. The second reader consumes exactly the same frozen feature tensor
    that the auxiliary reader computes for the same two future RGB images.
    """
    reuse_reader_features = True

    def _reader_with_features(self, branch, history, ids, offset, features):
        y, x = branch['supervision'], branch['inputs']
        reference = reference_for_queries(branch, ids)
        descriptors = (torch.stack([track['descriptor'] for track in history['result']['tracks']])
                       if len(ids) else history['shared'].new_empty(0, 256))
        output = self.reader(features, y['depth_mm'][offset], y['valid'][offset],
                             x['K'][None].expand(2, -1, -1),
                             x['planned_camera_to_world'][offset],
                             x['future_time_s'][offset].expand(2),
                             torch.arange(2, device=features.device), descriptors.detach())
        output = {key: value[None] for key, value in output.items()}
        reference = {key: (value[offset:offset+1] if key != 'known' else value)
                     for key, value in reference.items()}
        state, identity = readout_loss(output, reference, descriptors,
                                      self.upstream.tracker.scale.std[:3])
        return state + identity

    def forward(self, branch, generator, update=0, family=None, feedback=False, baseline=0.):
        if self.phase != 'dynamics' or feedback:
            raise ValueError('This isolated execution candidate covers dynamics only')
        cached = branch.get('_prefetched_replay')
        if cached is None or cached['offset'] != int(update) % 16:
            raise ValueError('Replay preparation and actual update offset disagree')
        self.feedback_gradient_norms = {}
        encoded, history, draw, condition = self.prepared(
            branch, generator, True, build_condition=False)
        replay = dict(inputs=branch['inputs'], labels=cached['labels'],
                      auxiliary=cached['auxiliary'])
        ids = query_labels(history, historical_labels(branch))
        stats = dict(predicted_tracks=len(ids), matched_tracks=int((ids >= 0).sum()))
        auxiliary = self.encoders.features(replay['auxiliary']['rgb'])
        past = (history['result'], history['observations'], history['shared'], history['separate'])
        loss, parts, _ = self.upstream.objective(
            replay, encoded, auxiliary, geometry_chunk=8, history_result=past)
        if branch.get('cohort') == 'ambiguous' and self.variant['ambiguous_nll_only']:
            loss = loss - .1 * parts['rollout_l1']
        if self.reuse_reader_features:
            reader = self._reader_with_features(branch, history, ids, update % 16, auxiliary)
        else:
            reader = self.reader_objective(branch, history, ids, update % 16)
        return loss + reader, dict(stats, **parts, dense_reader=reader)
