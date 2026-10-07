"""Shared scene dynamics and camera-specific, layered surface prediction."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import backproject, box_intervals, inverse_depth, normalize_depth, soft_order
from .modules import CrossBlock
from .temporal_state import StateScale


class SceneAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Parameter(torch.randn(256, 256)*.02)
        self.per_view = nn.Parameter(torch.randn(2, 128, 256)*.02)
        self.blocks = nn.ModuleList([CrossBlock(256, 8) for _ in range(4)])

    def forward(self, observations):
        features = torch.stack([out['scene_features'] for out in observations])
        union = torch.stack([1-(1-out['M']*(out['class_logits'].softmax(-1)[..., 1:].amax(-1) > .5)[..., None, None]).prod(1)
                             for out in observations]).flatten(-2)
        memory = (features*(1-union[..., None])).flatten(0, 2)[None]
        shared = self.shared[None]
        separate = self.per_view
        per_memory = (features*(1-union[..., None])).permute(1, 0, 2, 3).flatten(1, 2)
        for block in self.blocks:
            shared = block(shared, memory)
            separate = block(separate, per_memory)
        return shared, separate


class JointDynamics(nn.Module):
    def __init__(self, statistics):
        super().__init__()
        self.scale = StateScale(statistics)
        self.state = nn.Linear(12, 256)
        self.modes = nn.Parameter(torch.randn(8, 256)*.02)
        self.times = nn.Parameter(torch.randn(16, 256)*.02)
        self.blocks = nn.ModuleList([CrossBlock(256, 8) for _ in range(6)])
        self.mixture = nn.Linear(256, 8)
        self.acceleration = nn.Linear(256, 6)
        self.yaw = nn.Linear(256, 2)
        self.presence = nn.Linear(256, 1)
        # Execution policy only; no parameters or random draws are added.
        self.checkpoint_tracks_above = 0

    def forward(self, tokens, scene, previous, mode_indices=None):
        b, t, n, _ = previous.shape
        if n == 0 or not 1 <= t <= 16:
            raise ValueError('Dynamics requires a nonempty history-initialized track set and 1..16 steps')
        count = 8 if mode_indices is None else 1
        mode = self.modes[None].expand(b, -1, -1) if mode_indices is None else self.modes[mode_indices, None]
        context = torch.cat([tokens, scene], 1)
        logits = self.mixture(context.mean(1))
        x = self.state(self.scale.encode(previous))[:, None]+tokens[:, None, None]+mode[:, :, None, None]+self.times[None, None, :t, None]
        x = x.reshape(b*count, t*n, 256)
        context = context[:, None].expand(-1, count, -1, -1).flatten(0, 1)
        time = torch.arange(t, device=x.device).repeat_interleave(n)
        causal = time[None] > time[:, None]
        for block in self.blocks:
            if self.training and torch.is_grad_enabled() and n > self.checkpoint_tracks_above:
                x = checkpoint(block, x, context, causal, use_reentrant=False)
            else:
                x = block(x, context, causal)
        x = x.reshape(b, count, t, n, 256)
        acc, yaw = self.acceleration(x), self.yaw(x)
        return dict(mode_logits=logits, acc_mean=acc[..., :3], acc_scale=F.softplus(acc[..., 3:])+1e-4,
                    yaw_mean=math.pi*torch.tanh(yaw[..., 0]), yaw_scale=(F.softplus(yaw[..., 1])+1e-4).clamp_max(math.pi),
                    presence_logits=self.presence(x).squeeze(-1))

    @torch.no_grad()
    def sample(self, tokens, scene, initial, initial_presence, generator):
        """One scene-mode draw and one rollout shared by both cameras."""
        if not len(initial):
            return dict(state=initial.new_empty((16, 0, 11)), present=initial_presence.new_empty((16, 0)), mode=None)
        states, presence, mode = [initial], [], None
        alive = initial_presence.bool()
        for h in range(16):
            previous = torch.stack(states)[None]
            if mode is None:
                all_modes = self(tokens[None], scene, previous)
                mode = torch.multinomial(all_modes['mode_logits'].softmax(-1), 1, generator=generator).flatten()
            out = self(tokens[None], scene, previous, mode)
            acceleration = (out['acc_mean'][0, 0, -1]+out['acc_scale'][0, 0, -1]*
                torch.randn((len(initial), 3), device=initial.device, generator=generator))*self.scale.acc_std+self.scale.acc_mean
            delta = out['yaw_mean'][0, 0, -1]+out['yaw_scale'][0, 0, -1]*torch.randn(len(initial), device=initial.device, generator=generator)
            alive = alive&(torch.rand(len(initial), device=initial.device, generator=generator) < out['presence_logits'][0, 0, -1].sigmoid())
            old = states[-1]
            new = torch.cat([old[:, :3]+.2*old[:, 3:6]+.02*acceleration, old[:, 3:6]+.2*acceleration,
                             ((old[:, 6]+delta+math.pi)%(2*math.pi)-math.pi)[:, None], (delta/.2)[:, None], old[:, 8:]], -1)
            states.append(torch.where(alive[:, None], new, old)); presence.append(alive)
        return dict(state=torch.stack(states[1:]), present=torch.stack(presence), mode=int(mode))

    def objective(self, tokens, scene, initial, history_reference, reference, eligible, present):
        """Full 16-step joint likelihood plus autoregressive mode-mean paths.

        Reference states teacher-force the likelihood only. Rollouts start from
        estimated history and feed back their own states in every mode.
        """
        if not len(initial):
            zero = scene.sum()*0
            return zero, dict(dynamics_nll=zero, rollout_l1=zero), initial.new_empty((8, 16, 0, 11))
        previous = torch.cat([history_reference[None], reference[:-1]], 0)
        output = self(tokens[None], scene, previous[None])
        acc = ((reference[..., 3:6]-previous[..., 3:6])/.2-self.scale.acc_mean)/self.scale.acc_std
        logp = (-.5*((acc[None]-output['acc_mean'][0])/output['acc_scale'][0]).square()-output['acc_scale'][0].log()-.5*math.log(2*math.pi)).sum(-1)
        yaw = (reference[..., 6]-previous[..., 6]+math.pi)%(2*math.pi)-math.pi
        shifts = torch.arange(-4, 5, device=yaw.device)*2*math.pi
        diff = (yaw[None, ..., None]-output['yaw_mean'][0, ..., None]+shifts)/output['yaw_scale'][0, ..., None]
        logp = logp+torch.logsumexp(-.5*diff.square()-output['yaw_scale'][0, ..., None].log()-.5*math.log(2*math.pi), -1)
        prev_present = torch.cat([eligible[None], present[:-1]], 0)&eligible[None]
        active = prev_present&present
        hazard = -F.binary_cross_entropy_with_logits(output['presence_logits'][0], present[None].float().expand(8, -1, -1), reduction='none')
        scores = (logp*active[None]+hazard*prev_present[None]).sum((1, 2))
        nll = -torch.logsumexp(output['mode_logits'][0].log_softmax(-1)+scores, 0)/prev_present.sum().clamp_min(1)
        # A separate fixed mode per replicated scene prevents 8x8 duplication.
        modes = torch.arange(8, device=initial.device)
        states = [initial[None].expand(8, -1, -1)]
        repeated_tokens, repeated_scene = tokens[None].expand(8, -1, -1), scene.expand(8, -1, -1)
        for h in range(16):
            out = self(repeated_tokens, repeated_scene, torch.stack(states, 1), modes)
            acceleration = out['acc_mean'][:, 0, -1]*self.scale.acc_std+self.scale.acc_mean
            delta = out['yaw_mean'][:, 0, -1]
            old = states[-1]
            new = torch.cat([old[..., :3]+.2*old[..., 3:6]+.02*acceleration,
                             old[..., 3:6]+.2*acceleration,
                             ((old[..., 6]+delta+math.pi)%(2*math.pi)-math.pi)[..., None],
                             (delta/.2)[..., None], old[..., 8:]], -1)
            states.append(torch.where(active[h][None, :, None], new, old))
        rollout = torch.stack(states[1:], 1)
        error = ((rollout[..., :3]-reference[None, ..., :3])/self.scale.std[:3]).abs().mean(-1)
        per_mode = (error*active[None]).sum((1, 2))/active.sum().clamp_min(1)
        rollout_loss = (per_mode*output['mode_logits'][0].softmax(-1)).sum()
        return nll+.1*rollout_loss, dict(dynamics_nll=nll, rollout_l1=rollout_loss), rollout


def geometry_intrinsics(K):
    result = K.clone()
    result[0, 0] /= 8; result[1, 1] /= 8
    result[0, 2] = (K[0, 2]+.5)/8-.5+1
    result[1, 2] = (K[1, 2]+.5)/8-.5
    return result


def unique_integer_rows(rows):
    """Exact lexicographic uniqueness using collision-free mixed-radix keys."""
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.dtype != torch.int64:
        raise ValueError('Voxel indices must be int64 [N,3]')
    if not len(rows):
        return rows
    low, high = rows.amin(0), rows.amax(0)
    # Python integers avoid overflow even while calculating the coordinate span.
    spans = [b-a+1 for a,b in zip(low.tolist(), high.tolist())]
    if spans[0]*spans[1]*spans[2] > torch.iinfo(torch.int64).max:
        return torch.unique(rows, dim=0)
    offset = rows-low
    yz = spans[1]*spans[2]
    keys = torch.unique(offset[:, 0]*yz+offset[:, 1]*spans[2]+offset[:, 2], sorted=True)
    rem = keys%yz
    return torch.stack([keys//yz, rem//spans[2], rem%spans[2]], 1)+low


@torch.no_grad()
def static_voxels(inputs, observations, packed=False):
    """0.1 m voxels from valid historical depth, excluding estimated targets."""
    depth, valid = inputs['depth_mm'], inputs['depth_valid']
    union = torch.stack([1-(1-out['M']*(out['class_logits'].softmax(-1)[..., 1:].amax(-1) > .5)[..., None, None]).prod(1)
                         for out in observations])
    union = F.interpolate(union.flatten(0, 1)[:, None], size=(480, 720), mode='bilinear', align_corners=False).reshape(6, 2, 480, 720)
    points = backproject(depth, inputs['K'], inputs['camera_to_world'])
    okay = valid&(depth >= 100)&(depth <= 60000)&(union < .5)&torch.isfinite(points).all(-1)
    points = points[okay]
    indices = torch.floor(points/.1).long()
    unique = unique_integer_rows(indices) if packed else torch.unique(indices, dim=0)
    return (unique.float()+.5)*.1


@torch.no_grad()
def project_voxels(voxels, K, cameras, chunk_size=8192):
    """Conservative projected voxel footprints with nearest front-Z retention."""
    outputs = []
    corners = torch.tensor([[x, y, z] for x in (-.05, .05) for y in (-.05, .05) for z in (-.05, .05)], device=K.device)
    for camera in cameras:
        depth = K.new_full((60*92,), 60000.)
        for chunk in voxels.split(chunk_size):
            xyz = (chunk[:, None]+corners-camera[:3, 3])@camera[:3, :3]
            good = xyz[..., 2].amin(-1) >= .1
            xyz = xyz[good]
            if len(xyz) == 0:
                continue
            uv = xyz@K.T
            uv = uv[..., :2]/xyz[..., 2:]
            low, high = uv.amin(1).ceil().long(), uv.amax(1).floor().long()
            low[:, 0].clamp_(1, 90); low[:, 1].clamp_(0, 59)
            high[:, 0].clamp_(1, 90); high[:, 1].clamp_(0, 59)
            in_image = (uv.amax(1)[:, 0] >= 1)&(uv.amin(1)[:, 0] <= 90)&(uv.amax(1)[:, 1] >= 0)&(uv.amin(1)[:, 1] <= 59)
            lengths = (high-low+1).clamp_min(0)
            counts = lengths.prod(-1)*in_image
            index = torch.repeat_interleave(torch.arange(len(xyz), device=K.device), counts)
            if index.numel() == 0:
                continue
            starts = torch.cumsum(counts, 0)-counts
            offset = torch.arange(len(index), device=K.device)-starts[index]
            x = low[index, 0]+offset%lengths[index, 0].clamp_min(1)
            y = low[index, 1]+offset//lengths[index, 0].clamp_min(1)
            front = xyz[..., 2].amin(-1)*1000
            depth.scatter_reduce_(0, y*92+x, front[index], reduce='amin', include_self=True)
        outputs.append(torch.stack([depth.reshape(60, 92), (depth < 60000).float().reshape(60, 92)]))
    return torch.stack(outputs)


class GeometryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.ModuleList([nn.LayerNorm(256), nn.LayerNorm(256)])
        self.cross = nn.MultiheadAttention(256, 8, batch_first=True)
        self.spatial = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, x, context):
        x = x+self.cross(self.norm[0](x), context, context, need_weights=False)[0]
        x = x+self.spatial(x.transpose(1, 2).reshape(-1, 256, 60, 92)).flatten(2).transpose(1, 2)
        return x+self.ff(self.norm[1](x))


class SurfaceGeometry(nn.Module):
    def __init__(self, statistics):
        super().__init__()
        self.scale = StateScale(statistics)
        self.input = nn.Linear(8, 256)  # log Z, coverage, UV, world ray, time
        self.state = nn.Linear(12, 256)
        self.platform = nn.Embedding(2, 256)
        self.blocks = nn.ModuleList([GeometryBlock() for _ in range(4)])
        self.static = nn.Linear(256, 2)
        self.object = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, shared, per_view, track_tokens, states, present, projection, K, cameras, times, views):
        count, objects = states.shape[:2]
        yy, xx = torch.meshgrid(torch.arange(60, device=K.device), torch.arange(92, device=K.device), indexing='ij')
        pixels = torch.stack([xx, yy, torch.ones_like(xx)], -1).float()
        ray = torch.einsum('bij,hwj->bhwi', cameras[:, :3, :3], pixels@torch.linalg.inv(K).T)
        ray = F.normalize(ray, dim=-1).flatten(1, 2)
        uv = torch.stack([xx/91*2-1, yy/59*2-1], -1).flatten(0, 1)[None].expand(count, -1, -1)
        log, _ = normalize_depth(projection[:, 0], projection[:, 1].bool())
        features = torch.cat([log.flatten(1)[..., None], projection[:, 1].flatten(1)[..., None], uv, ray,
                              times[:, None, None].expand(-1, 5520, -1)], -1)
        x = self.input(features)+self.platform(views)[:, None]
        target_tokens = self.state(self.scale.encode(states))+track_tokens[None]
        context = torch.cat([shared.expand(count, -1, -1), per_view[views], target_tokens], 1)
        for block in self.blocks:
            x = checkpoint(block, x, context, use_reentrant=False) if self.training and torch.is_grad_enabled() else block(x, context)
        static = self.static(x).reshape(count, 60, 92, 2)
        domain = x.new_ones((60, 92)); domain[:, [0, -1]] = 0
        zs = inverse_depth(2*static[..., 0].sigmoid()-1)
        ps = static[..., 1].sigmoid()*domain
        Z, P = [zs], [ps]
        for k in range(objects):
            out = self.object(torch.cat([x, target_tokens[:, k, None].expand(-1, 5520, -1)], -1)).reshape(count, 60, 92, 2)
            bounds = [box_intervals(states[j, k:k+1, :3], states[j, k:k+1, 8:11], states[j, k:k+1, 6], K, cameras[j]) for j in range(count)]
            entry, exit, envelope = [torch.cat([item[a] for item in bounds]) for a in range(3)]
            pk = out[..., 0].sigmoid()*envelope*domain*present[:, k, None, None]
            zk = entry+(exit-entry)*out[..., 1].sigmoid()
            Z.append(torch.where(envelope, zk, zk.new_tensor(60000.))); P.append(pk)
        Z, P = torch.stack(Z, 1), torch.stack(P, 1)
        return dict(Z=Z, P=P, M=soft_order(Z, P), coverage=projection[:, 1])


def geometry_loss(prediction, target):
    domain = torch.ones_like(prediction['P'][:, 0]); domain[..., [0, -1]] = 0
    zero = prediction['Z'].sum()*0
    def mean(values, mask):
        return (values*mask).sum()/mask.sum().clamp_min(1)
    def masks(pred, ref):
        if pred.numel() == 0:
            return zero
        error = F.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), ref, reduction='none')
        weight = domain[:, None].expand_as(error)
        dice = 1-(2*(pred*ref*weight).sum((-2, -1))+1)/((pred*weight).sum((-2, -1))+(ref*weight).sum((-2, -1))+1)
        return mean(error, weight)+dice.mean()
    zs, ps = target['static_depth_mm'], target['static_support']
    ref_static, _ = normalize_depth(zs, ps > 0)
    pred_static, _ = normalize_depth(prediction['Z'][:, 0])
    terms = dict(static_support=mean(F.binary_cross_entropy(prediction['P'][:, 0].clamp(1e-6, 1-1e-6), ps, reduction='none'), domain),
                 static_depth=mean((ref_static-pred_static).abs(), (ps > 0)*domain))
    z, p, m = target['isolated_depth_mm'], target['full_mask'], target['visible_mask']
    ref_depth, _ = normalize_depth(z, p > 0)
    pred_depth, _ = normalize_depth(prediction['Z'][:, 1:])
    terms.update(silhouette=masks(prediction['P'][:, 1:], p),
                 isolated_depth=mean((ref_depth-pred_depth).abs(), (p > 0)*domain[:, None]),
                 visible_mask=masks(prediction['M'][:, 1:], m))
    return sum(terms.values()), terms
