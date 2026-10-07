"""Opt-in renderer loss support for history-target regions vacated in future.

Only supervision changes. History-inferred states and generated conditions are
untouched. The control calls the original flow objective exactly. The candidate
retains the same noise/time, single Wan forward and 1..3 pixel weights; it adds
vacated historical surface pixels to the original future-target support union.
"""
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .region_objective import ReappearanceSystem


@torch.no_grad()
def vacated_background_mask(branch, tolerance_mm=20.):
    x, y = branch['inputs'], branch['supervision']
    history, future = y['history_instance'], y['future_instance']
    ids = y['instance_id']
    if history.shape[0] != 6 or future.shape[0] != 16 or history.shape[1] != 2:
        raise ValueError('Native paired six-history/sixteen-future supervision required')
    result = torch.zeros_like(future, dtype=torch.bool)
    foreground = torch.isin(future, ids)
    height, width = future.shape[-2:]
    inverse_k = torch.linalg.inv(x['K'].double())
    for view in range(2):
        for ident in ids:
            eligible = (history[:, view] == ident).flatten(1).any(1).nonzero().flatten()
            if not len(eligible):
                continue
            t = int(eligible[-1])
            d = x['depth_mm'][t, view].double()
            valid = x['depth_valid'][t, view].bool() & torch.isfinite(d) & (d >= 100) & (d <= 60000)
            yy, xx = torch.where((history[t, view] == ident) & valid)
            if not len(xx):
                continue
            ray = torch.stack((xx, yy, torch.ones_like(xx)), -1).double() @ inverse_k.T
            old_camera = x['camera_to_world'][t, view].double()
            world = (ray * (d[yy, xx, None]/1000.)) @ old_camera[:3, :3].T + old_camera[:3, 3]
            for frame in range(16):
                camera = x['planned_camera_to_world'][frame, view].double()
                cam = (world-camera[:3, 3]) @ camera[:3, :3]
                uv = cam @ x['K'].double().T
                z = uv[:, 2]
                u = torch.round(uv[:, 0]/z.clamp_min(1e-12)).long()
                v = torch.round(uv[:, 1]/z.clamp_min(1e-12)).long()
                inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                u, v, old_depth = u[inside], v[inside], cam[inside, 2]*1000.
                target_depth = y['depth_mm'][frame, view, v, u]
                keep = (y['valid'][frame, view, v, u].bool() & torch.isfinite(target_depth)
                    & (target_depth >= 100) & (target_depth <= 60000)
                    & (target_depth >= old_depth-tolerance_mm) & ~foreground[frame, view, v, u])
                result[frame, view, v[keep], u[keep]] = True
    return result


def union_flow_weights(branch):
    original = branch['supervision']['isolated_support'].bool().any(2)
    vacated = vacated_background_mask(branch)
    support = (original | vacated).float().permute(1, 0, 2, 3)[:, None]
    domain = torch.ones_like(support)
    weight = F.adaptive_avg_pool3d(F.pad(1+2*support, (8,8,0,0,0,0)), (4,30,46))
    coverage = F.adaptive_avg_pool3d(F.pad(domain, (8,8,0,0,0,0)), (4,30,46))
    return weight, coverage, dict(vacated_native_pixels=int(vacated.sum()),
        newly_upweighted_native_pixels=int((vacated & ~original).sum()),
        future_truth_conditions=False, maximum_native_weight=3.)


class VacatedFlowSystem(ReappearanceSystem):
    """No extra forward/parameters. Set vacated_enabled explicitly after load."""
    vacated_enabled = False

    def flow_objective(self, branch, condition, latent, generator):
        from experiments.base48 import model as original
        # Both arms deliberately use original flow, without an additional
        # reappearance term; this isolates condition adaptation vs support.
        if not self.vacated_enabled:
            value = original.Base48System.flow_objective(self, branch, condition, latent, generator)
            self.last_region_terms = dict(vacated_enabled=False, future_truth_conditions=False)
            return value
        weight, coverage, metrics = union_flow_weights(branch)
        def replacement(_branch):
            if _branch is not branch:
                raise ValueError('Unexpected branch entered scoped loss accounting')
            return weight, coverage
        # Reference masks only choose the scalar loss weights after Wan forward.
        # The actual forward receives the unmodified history-derived condition.
        with patch.object(original, 'flow_weights', replacement):
            value = original.Base48System.flow_objective(self, branch, condition, latent, generator)
        self.last_region_terms = dict(metrics, vacated_enabled=True)
        return value
