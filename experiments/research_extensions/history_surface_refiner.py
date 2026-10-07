"""Native historical RGB-D surface support and calibrated measurement residuals.

The module receives predicted queries and historical sensor observations only.
Ground-truth instance masks and target states belong to the trainer's losses,
never this forward path. Optical-axis depth is in millimetres; camera/world
points, centers and diagonal center variances are in metres / metres squared.

The coarse arm uses the original mask expanded as constant 20x20 blocks at the
480x720 production resolution. The native arm learns a pixelwise residual to
that prior. Both arms share the RGB-D stem and center/covariance heads. Only
center and covariance are replaced in the observation dictionary.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class HistorySurfaceRefiner(nn.Module):
    def __init__(self, width=256, channels=16, query_chunk=8,
                 support_mode='native', checkpoint_pooling=True):
        super().__init__()
        if support_mode not in ('native', 'coarse'):
            raise ValueError('support_mode must be native or coarse')
        if min(width, channels, query_chunk) <= 0:
            raise ValueError('width, channels and query_chunk must be positive')
        self.config = dict(width=int(width), channels=int(channels),
                           query_chunk=int(query_chunk), support_mode=support_mode,
                           checkpoint_pooling=bool(checkpoint_pooling))
        self.width, self.channels = int(width), int(channels)
        self.query_chunk = int(query_chunk)
        self.support_mode = support_mode
        self.checkpoint_pooling = bool(checkpoint_pooling)
        # A native-resolution shallow skip retains within-patch RGB/depth edges.
        # No target-dependent crop, instance ID or reference box is supplied.
        self.pixel = nn.Sequential(
            nn.Conv2d(5, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1), nn.GELU())
        self.query = nn.Linear(width, channels)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)
        # query, RGB-D pool, surface-to-old-center, optical depth standard
        # deviation, effective support fraction, size, heading, old log variance.
        nfeatures = width+channels+3+1+1+3+2+3
        self.measurement = nn.Sequential(nn.Linear(nfeatures, 128), nn.GELU(),
                                         nn.Linear(128, 6))
        nn.init.zeros_(self.measurement[-1].weight)
        nn.init.zeros_(self.measurement[-1].bias)

    @staticmethod
    def _cameras(K, T, batch, dtype, device):
        K = K.to(device=device, dtype=dtype)
        T = T.to(device=device, dtype=dtype)
        if K.shape == (3, 3):
            K = K[None].expand(batch, -1, -1)
        if K.shape != (batch, 3, 3) or T.shape != (batch, 4, 4):
            raise ValueError('Expected intrinsics [B,3,3] or [3,3] and poses [B,4,4]')
        if not bool(torch.isfinite(K).all() and torch.isfinite(T).all()):
            raise ValueError('Camera calibration must be finite')
        return K, T

    def _inputs(self, pair, rgb, depth_mm, depth_valid, K, T):
        if depth_mm.ndim != 3 or depth_valid.shape != depth_mm.shape:
            raise ValueError('Historical depth and validity must be [B,H,W]')
        batch, height, width = depth_mm.shape
        if rgb.shape != (batch, height, width, 3):
            raise ValueError('Historical RGB must be [B,H,W,3]')
        query = pair['tokens']
        if query.ndim != 3 or query.shape[0] != batch or query.shape[-1] != self.width:
            raise ValueError('Predicted observation tokens must be [B,Q,width]')
        if pair['M'].shape[:2] != query.shape[:2] or pair['M'].ndim != 4:
            raise ValueError('Predicted visible masks must be [B,Q,h,w]')
        if pair['center'].shape != (*query.shape[:2], 3):
            raise ValueError('Predicted centers must be [B,Q,3]')
        if pair['covariance'].shape != pair['center'].shape:
            raise ValueError('Predicted diagonal covariance must be [B,Q,3]')
        if bool((pair['covariance'] <= 0).any()):
            raise ValueError('Parent measurement variance must be strictly positive')
        # Use the same 100--60000 mm validity convention as the parent frontend.
        dtype, device = query.dtype, query.device
        depth = depth_mm.to(device=device, dtype=dtype)
        valid = depth_valid.to(device=device).bool() & torch.isfinite(depth)
        valid = valid & (depth >= 100.) & (depth <= 60000.)
        safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
        logarithm = 2*(safe_depth.clamp(100., 60000.).log()-math.log(100.))/math.log(600.)-1
        logarithm = torch.where(valid, logarithm, torch.zeros_like(logarithm))
        rgb01 = rgb.to(device=device, dtype=dtype).permute(0, 3, 1, 2)/255.
        feature = self.pixel(torch.cat((rgb01-.5, logarithm[:, None], valid[:, None].to(dtype)), 1))
        K, T = self._cameras(K, T, batch, dtype, device)
        y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                              torch.arange(width, device=device, dtype=dtype), indexing='ij')
        pixels = torch.stack((x, y, torch.ones_like(x)), 0)
        rays = torch.einsum('bij,jhw->bihw', torch.linalg.inv(K), pixels)
        points = rays*(safe_depth*.001)[:, None]
        return feature, points, valid, T

    def _support(self, masks, query, features):
        """One frame / bounded query chunk; no labels or query assignment."""
        prior = F.interpolate(masks[None], features.shape[-2:], mode='nearest')[0]
        if self.support_mode == 'coarse':
            return prior
        # A finite soft prior allows learning pixel support outside a low-valued
        # coarse cell, without using an oracle crop or masking by target labels.
        logits = torch.logit(prior.clamp(1e-5, 1-1e-5))
        residual = torch.einsum('qc,chw->qhw', self.query(query), features)/math.sqrt(self.channels)
        return (logits+residual).sigmoid()

    def _pool(self, masks, query, features, points, valid):
        support = self._support(masks, query, features)
        weight = support*valid[None].to(support.dtype)
        mass = weight.sum((-2, -1))
        norm = weight/mass[:, None, None].clamp_min(1e-8)
        surface = torch.einsum('qhw,chw->qc', norm, points)
        pooled = torch.einsum('qhw,chw->qc', norm, features)
        depth_second = torch.einsum('qhw,hw->q', norm, points[2].square())
        depth_variance = (depth_second-surface[:, 2].square()).clamp_min(0.)
        return surface, pooled, depth_variance, mass

    def support_for_queries(self, pair, aux, frame_index, query_indices):
        """Return native support probabilities for evaluation / mask loss.

        Index selection occurs after the full inference forward. Reference
        assignment may select supervised rows here, but cannot alter centers,
        track births, missing slots or the state inferred by ``forward``.
        """
        query_indices = torch.as_tensor(query_indices, device=pair['tokens'].device, dtype=torch.long)
        if query_indices.ndim != 1:
            raise ValueError('query_indices must be a one-dimensional index list')
        if not len(query_indices):
            return aux['pixel_features'].new_empty((0, *aux['pixel_features'].shape[-2:]))
        return self._support(pair['M'][frame_index, query_indices],
                             pair['tokens'][frame_index, query_indices],
                             aux['pixel_features'][frame_index])

    def forward(self, pair, rgb, depth_mm, depth_valid, K, camera_to_world):
        features, points, valid, T = self._inputs(pair, rgb, depth_mm, depth_valid, K, camera_to_world)
        summaries = []
        for frame in range(len(features)):
            chunks = []
            for start in range(0, pair['tokens'].shape[1], self.query_chunk):
                args = (pair['M'][frame, start:start+self.query_chunk],
                        pair['tokens'][frame, start:start+self.query_chunk],
                        features[frame], points[frame], valid[frame])
                if self.checkpoint_pooling and torch.is_grad_enabled() and features.requires_grad:
                    chunks.append(checkpoint(self._pool, *args, use_reentrant=False))
                else:
                    chunks.append(self._pool(*args))
            if not chunks:
                raise ValueError('At least one predicted query is required')
            summaries.append(tuple(torch.cat(parts, 0) for parts in zip(*chunks)))
        surface, pooled, depth_variance, mass = (torch.stack(parts) for parts in zip(*summaries))
        center_camera = torch.einsum('bij,bqj->bqi', T[:, :3, :3].transpose(-1, -2),
                                      pair['center']-T[:, None, :3, 3])
        fraction = mass/valid.sum((-2, -1))[:, None].clamp_min(1)
        measurement_features = torch.cat((pair['tokens'], pooled,
                                           surface-center_camera,
                                           (depth_variance+1e-8).sqrt()[..., None],
                                           fraction[..., None], pair['dimensions'], pair['heading'],
                                           pair['covariance'].log()), -1)
        correction = self.measurement(measurement_features)
        delta_world = torch.einsum('bij,bqj->bqi', T[:, :3, :3], correction[..., :3])
        # Smooth bounded log-ratio is positive, has unit derivative at zero and
        # exactly preserves the old variance at initialization. It is learned
        # by calibrated measurement/posterior losses, never manually shrunk.
        variance_factor = torch.exp(5*torch.tanh(correction[..., 3:]/5))
        supported = mass[..., None] > 1e-5
        refined = dict(pair)
        refined['center'] = torch.where(supported, pair['center']+delta_world, pair['center'])
        refined['covariance'] = torch.where(supported, pair['covariance']*variance_factor, pair['covariance'])
        aux = dict(pixel_features=features, camera_points=points, valid=valid,
                   support_mass=mass, surface_camera=surface,
                   support_depth_variance=depth_variance,
                   center_residual_camera=correction[..., :3], supported=supported.squeeze(-1))
        return refined, aux


def _surface_history(upstream, inputs, encoded):
    if upstream._history_surface_busy:
        raise RuntimeError('Concurrent or recursive history calls are unsupported')
    upstream._history_surface_busy = True
    upstream.history_surface_last_aux = None

    def refine(_module, _args, pair):
        refined, aux = upstream.history_surface_refiner(
            pair, inputs['rgb'].flatten(0, 1), inputs['depth_mm'].flatten(0, 1),
            inputs['depth_valid'].flatten(0, 1), inputs['K'], inputs['camera_to_world'].flatten(0, 1))
        if upstream._history_surface_record_aux:
            upstream.history_surface_last_aux = dict(input_pair=pair, output_pair=refined, aux=aux)
        return refined

    handle = upstream.observation.register_forward_hook(refine)
    try:
        # Calls the original implementation, including availability masking,
        # past-only association, track creation, recurrent filter and scene head.
        return upstream._history_surface_original(inputs, encoded)
    finally:
        handle.remove()
        upstream._history_surface_busy = False


def install_history_surface(upstream, refiner, record_aux=False):
    """Install a reversible runtime adapter; never replace parent parameters.

    Only calls made by ``upstream.history`` are refined. An auxiliary reader or
    direct observation-head call is unchanged. Reinstall replaces the refiner
    without adding another wrapper. Parameter freezing / six process-noise
    parameter updates remain explicit responsibilities of the trainer.
    """
    if not isinstance(refiner, HistorySurfaceRefiner):
        raise TypeError('refiner must be HistorySurfaceRefiner')
    if not hasattr(upstream, '_history_surface_original'):
        upstream._history_surface_original = upstream.history
        upstream.history = MethodType(_surface_history, upstream)
        upstream._history_surface_busy = False
    elif upstream._history_surface_busy:
        raise RuntimeError('Cannot replace a refiner during history execution')
    upstream.history_surface_refiner = refiner
    upstream._history_surface_record_aux = bool(record_aux)
    upstream.history_surface_last_aux = None
    return upstream


def remove_history_surface(upstream):
    """Restore the original history method and remove only the new module."""
    if not hasattr(upstream, '_history_surface_original'):
        return upstream
    if upstream._history_surface_busy:
        raise RuntimeError('Cannot remove a refiner during history execution')
    upstream.history = upstream._history_surface_original
    for name in ('_history_surface_original', '_history_surface_busy',
                 '_history_surface_record_aux', 'history_surface_refiner', 'history_surface_last_aux'):
        delattr(upstream, name)
    return upstream
