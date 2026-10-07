"""Opt-in repair of two CPU-verified geometry-conditioning failure modes.

This module is not installed into existing checkpoints, samplers, or queues.
The deployment branch still receives history-inferred states. A separate
pose-aligned auxiliary forward uses detached TRAIN labels to teach shape;
its predictions never become renderer conditions. A new training run must
record this changed objective and angular-rate encoding in its own manifest.
"""
from contextlib import contextmanager
from types import MethodType
from unittest.mock import patch

import torch


@contextmanager
def instantaneous_geometry(geometry):
    """Remove angular rate from instantaneous surface features, not the state.

    The simulator's instantaneous silhouette depends on position, orientation
    and dimensions, all preserved here. Angular velocity remains in dynamics.
    Setting the geometry-only normalized omega channel to zero stops the
    measured 300--800-standard-deviation feedback from driving its K/V tokens.
    This changes model semantics; apply in both training and inference of a
    separately named candidate, never silently to an existing result.
    """
    original = geometry.scale.encode

    def encode(scale, states):
        encoded = original(states)
        # [p(3), u(3), sin(yaw), cos(yaw), omega, dimensions(3)]
        return torch.cat((encoded[..., :8], torch.zeros_like(encoded[..., 8:9]),
                          encoded[..., 9:]), -1)

    with patch.object(geometry.scale, 'encode', MethodType(encode, geometry.scale)):
        yield


def _shape_terms(prediction, target):
    from experiments.paper_experiments.agcr_exp.future_prediction import geometry_loss
    _, terms = geometry_loss(prediction, target)
    # Shape/depth are taught in the coordinate frame of their actual pose.
    # Visibility ordering retains its existing separate evaluation/training
    # role; this helper makes no claim to fix it with a missing target set.
    return terms['silhouette'] + terms['isolated_depth'], terms


def alignment_safe_geometry_objective(system, free, row, step, reappearance=True):
    """History-only static branch + pose-aligned shape auxiliary.

    Replacement for the geometry-only term, not for dynamics/evidence/RGB
    objectives. Keep free_motion_projection_loss/projected_alignment separate.
    They train localization even when a predicted box misses the true object;
    a shape classifier must not learn to hide that localization error.

    No visibility loss is returned here: importing a fixed image-space visible
    mask into a displaced free box repeats the same coordinate mismatch.
    The later renderer-feedback branch remains history-conditioned and must
    be evaluated separately before claiming improved generated videos.
    """
    from experiments.paper_experiments.agcr_exp.future_prediction import geometry_intrinsics, geometry_loss

    if row['metadata']['split'] != 'train':
        raise ValueError('Pose-aligned shape auxiliary is TRAIN supervision only')
    times = sorted({f for event in row['events'] for f in event['frames']}) if reappearance else []
    frame = times[step % len(times)] if times else step % 16
    mode = (step // 16) % free['state'].shape[0]
    history = row['history']
    slots, refs = row['selected_slots'], row['selected_references']
    if len(set(slots)) != len(slots) or len(set(refs)) != len(refs):
        raise ValueError('A unique historical slot/reference assignment is required')
    if len(slots) != len(refs):
        raise ValueError('Historical slots and reference indices must align')
    cameras = row['cameras'][frame]
    K = geometry_intrinsics(row['K'])
    views = torch.arange(2, device=K.device)
    projected = row['projection'][2*frame:2*frame+2].detach()
    clock = row['times'][frame].expand(2)

    with instantaneous_geometry(system.geometry):
        # The actual predicted scene is the only condition for this branch.
        state = free['state'][mode, frame]
        presence = free['survival'][mode, frame]
        predicted = system.geometry(history['shared'], history['separate'], history['tokens'][0],
            state[None].expand(2, -1, -1), presence[None].expand(2, -1),
            projected, K, cameras, clock, views)
        # Static labels remain in the common camera/world frame. Excluding
        # target terms prevents their unreachable positives from shrinking P.
        empty = predicted['P'][:, :0]
        static_prediction = {k: predicted[k][:, :1] for k in ('Z', 'P', 'M')}
        static_target = dict(static_depth_mm=row['geometry_targets']['static_depth_mm'][frame].detach(),
            static_support=row['geometry_targets']['static_support'][frame].detach(),
            isolated_depth_mm=empty, full_mask=empty, visible_mask=empty)
        static_loss, static_terms = geometry_loss(static_prediction, static_target)

        if slots:
            # This forward is auxiliary supervision only. Future poses cannot
            # flow into dynamics, history, or any renderer-condition return.
            truth = row['future_state'][frame, refs].detach()
            present = row['future_present'][frame, refs].detach().float()
            shape_prediction = system.geometry(history['shared'].detach(), history['separate'].detach(),
                history['tokens'][0, slots].detach(), truth[None].expand(2, -1, -1),
                present[None].expand(2, -1), projected, K, cameras, clock, views)
            shape_target = {key: row['geometry_targets'][key][frame, :, refs].detach()
                for key in ('isolated_depth_mm', 'full_mask', 'visible_mask')}
            shape_target.update({key: row['geometry_targets'][key][frame].detach()
                for key in ('static_depth_mm', 'static_support')})
            shape_loss, shape_terms = _shape_terms(shape_prediction, shape_target)
        else:
            shape_loss = static_loss * 0
            shape_terms = dict(silhouette=shape_loss, isolated_depth=shape_loss)

    # The static branch cycles modes under the same mixture measure as before.
    # Shape is independent of a sampled free mode and is therefore not scaled.
    weight = free['mode_logits'].softmax(-1)[mode] * free['state'].shape[0]
    total = weight * static_loss + shape_loss
    return total, dict(geometry_frame=frame, geometry_mode=mode,
        static_support=static_terms['static_support'], static_depth=static_terms['static_depth'],
        aligned_silhouette=shape_terms['silhouette'], aligned_isolated_depth=shape_terms['isolated_depth'],
        aligned_target_count=len(slots), future_truth_conditions=False,
        auxiliary_truth_pose=True, angular_rate_in_geometry=False)
