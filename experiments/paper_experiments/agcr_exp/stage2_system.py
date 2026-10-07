"""Integrated history frontend, dynamics, geometry and auxiliary RGB-D reader."""
import torch
from torch import nn
from torch.nn import functional as F

from .future_prediction import JointDynamics, SceneAggregator, SurfaceGeometry, geometry_intrinsics, geometry_loss, project_voxels, static_voxels
from .observation import ObservationHead, observation_loss
from .temporal_state import TemporalState, contrastive, historical_assignment, supervised_loss


def tensors(sample, device):
    import numpy as np
    if isinstance(sample, dict):
        return {name: tensors(value, device) for name, value in sample.items()}
    if isinstance(sample, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(sample)).to(device)
        return tensor.float() if tensor.dtype == torch.float64 else tensor
    return sample


def dino_images(rgb):
    x = rgb.flatten(0, -4) if rgb.ndim > 4 else rgb
    x = x.permute(0, 3, 1, 2).float()/255
    x = F.interpolate(x, (336, 504), mode='bilinear', align_corners=False)
    return (x-x.new_tensor([.485, .456, .406])[None, :, None, None])/x.new_tensor([.229, .224, .225])[None, :, None, None]


class FrozenImageEncoders:
    def __init__(self, dino_code, dino_weights, vae_path, device):
        from diffusers import AutoencoderKLWan
        self.dino = torch.hub.load(dino_code, 'dinov2_vitb14', source='local', pretrained=False).to(device)
        self.dino.load_state_dict(torch.load(dino_weights, map_location='cpu', weights_only=True))
        self.dino.eval().requires_grad_(False)
        self.vae = AutoencoderKLWan.from_pretrained(vae_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device)[None, :, None, None, None]
        self.std = torch.tensor(self.vae.config.latents_std, device=device)[None, :, None, None, None]

    @torch.no_grad()
    def history(self, rgb):
        dino = self.dino.forward_features(dino_images(rgb))['x_norm_patchtokens'].reshape(6, 2, 864, 768)
        frames = rgb.flatten(0, 1).permute(0, 3, 1, 2).float()/127.5-1
        frames = F.pad(frames, (8, 8, 0, 0), mode='replicate')[:, :, None]
        # Native single-image posterior means; no time-chunk leakage or pilot weights.
        latents = []
        for frame in frames.split(2):
            latents.append(((self.vae.encode(frame).latent_dist.mode()-self.mean)/self.std)[:, :, 0])
        return dino, torch.cat(latents).reshape(6, 2, 48, 30, 46)

    @torch.no_grad()
    def features(self, rgb):
        return self.dino.forward_features(dino_images(rgb))['x_norm_patchtokens']


def observation_target(state, classes, full, visible, detectable):
    return dict(center=state[None, :, :3].expand(2, -1, -1), dimensions=state[None, :, 8:11].expand(2, -1, -1),
                heading=torch.stack([state[:, 6].sin(), state[:, 6].cos()], -1)[None].expand(2, -1, -1),
                class_id=classes, full_mask=full, visible_mask=visible, detectable=detectable)


def matched_state_supervision(initial, reference_ids, labels):
    """Unmatched predicted tracks must not read the placeholder reference ID 0."""
    known = reference_ids >= 0
    ids = reference_ids.clamp_min(0)
    eligible = known&labels['history_eligible'][ids]
    fallback = initial[None].expand(16, -1, -1)
    reference = torch.where(eligible[None, :, None], labels['state'][:, ids], fallback)
    previous = torch.where(eligible[:, None], labels['history_state'][-1, ids], initial)
    present = labels['present'][:, ids]&eligible[None]
    return known, ids, eligible, previous, reference, present


class StageTwo(nn.Module):
    def __init__(self, statistics, observation_weights, execution_policy='standard'):
        super().__init__()
        if execution_policy not in ('standard', 'optimized_v1'):
            raise ValueError('Unknown stage-two execution policy')
        self.execution_policy = execution_policy
        self.observation = ObservationHead()
        self.observation.load_state_dict(observation_weights)
        self.tracker = TemporalState(statistics, self.observation.association)
        self.scene = SceneAggregator()
        self.dynamics = JointDynamics(statistics)
        # Dense/false-positive track sets retain the original memory safeguard.
        self.dynamics.checkpoint_tracks_above = 8 if execution_policy == 'optimized_v1' else 0
        self.geometry = SurfaceGeometry(statistics)
        self.reader = ObservationHead()
        self.reader.load_state_dict(observation_weights)
        self.reader_identity = nn.Linear(256, 256)

    def project_history(self, inputs, observations, K, cameras):
        optimized = self.execution_policy == 'optimized_v1'
        voxels = static_voxels(inputs, observations, packed=optimized)
        return voxels, project_voxels(voxels, K, cameras, chunk_size=65536 if optimized else 8192)

    def history(self, inputs, encoded):
        """Inference input boundary: there is no labels argument."""
        expected = {'rgb', 'depth_mm', 'depth_valid', 'K', 'camera_to_world', 'planned_camera_to_world', 'time_s', 'future_time_s'}
        if set(inputs) != expected:
            raise ValueError('Unexpected history input fields: '+str(set(inputs)^expected))
        if inputs['rgb'].shape[:2] != (6, 2) or (inputs['time_s'] > 0).any():
            raise ValueError('Only six paired current/history observations are permitted')
        dino, vae = encoded
        pair = self.observation(dino.flatten(0, 1), inputs['depth_mm'].flatten(0, 1), inputs['depth_valid'].flatten(0, 1),
                                inputs['K'][None].expand(12, -1, -1), inputs['camera_to_world'].flatten(0, 1),
                                inputs['time_s'].repeat_interleave(2), torch.arange(2, device=dino.device).repeat(6))
        observations = [{key: value[2*t:2*t+2] for key, value in pair.items()} for t in range(6)]
        result = self.tracker(observations, self.observation, dino, vae, inputs['camera_to_world'], inputs['time_s'])
        shared, separate = self.scene(observations)
        return result, observations, shared, separate

    @torch.no_grad()
    def forecast(self, inputs, encoded, generator, geometry_chunk=8):
        """Deployment path: no future RGB, depth, state, mask or reference ID."""
        result, observations, shared, separate = self.history(inputs, encoded)
        initial = result['initial_state'].clone()
        presence = []
        scale = self.tracker.scale.std[:6]
        for k, track in enumerate(result['tracks']):
            covariance = track['covariance']/scale[:, None]/scale[None]
            L = torch.linalg.cholesky((covariance+covariance.T)*.5+1e-6*torch.eye(6, device=scale.device))
            initial[k, :6] += (L@torch.randn(6, device=scale.device, generator=generator))*scale
            presence.append(torch.rand((), device=scale.device, generator=generator) < track['presence_logit'].sigmoid())
        presence = torch.stack(presence) if presence else torch.empty(0, dtype=torch.bool, device=initial.device)
        draw = self.dynamics.sample(result['tokens'], shared, initial, presence, generator)
        cameras = inputs['planned_camera_to_world'].flatten(0, 1)
        K = geometry_intrinsics(inputs['K'])
        _, projection = self.project_history(inputs, observations, K, cameras)
        states = draw['state'][:, None].expand(-1, 2, -1, -1).flatten(0, 1)
        alive = draw['present'][:, None].expand(-1, 2, -1).flatten(0, 1)
        geometry = []
        for start in range(0, 32, geometry_chunk):
            end = min(32, start+geometry_chunk)
            geometry.append(self.geometry(shared, separate, result['tokens'], states[start:end], alive[start:end],
                projection[start:end], K, cameras[start:end], inputs['future_time_s'].repeat_interleave(2)[start:end],
                torch.arange(start, end, device=K.device)%2))
        layers = {key:torch.cat([item[key] for item in geometry]).reshape(16, 2, *geometry[0][key].shape[1:]) for key in geometry[0]}
        return dict(history=result, shared_scene=shared, view_scene=separate, future=draw, geometry=layers)

    def auxiliary_loss(self, sample, features):
        out = self.reader(features, sample['depth_mm'], sample['valid'], sample['K'][None].expand(2, -1, -1),
                          sample['T'], features.new_zeros(2), torch.arange(2, device=features.device))
        target = observation_target(sample['state'], sample['class_id'], sample['full_mask'], sample['visible_mask'], sample['detectable'])
        loss, _ = observation_loss(self.reader, out, target)
        labels = dict(history_state=sample['state'][None], class_id=sample['class_id'], history_detectable=sample['detectable'][None])
        mapping = torch.stack(historical_assignment([out], labels)[0]).flatten()
        valid = mapping >= 0
        identity = contrastive(self.reader_identity(out['tokens'].flatten(0, 1)[valid]), mapping[valid])
        return loss+identity, dict(auxiliary_reader=loss, auxiliary_identity=identity)

    def objective(self, sample, encoded, auxiliary_features, geometry_chunk=2, timing=None):
        import time
        ticks = [time.monotonic()]
        def mark(name):
            if timing is not None:
                if encoded[0].is_cuda:
                    torch.cuda.synchronize()
                now = time.monotonic()
                timing[name] = now-ticks[0]
                ticks[0] = now
        inputs, labels = sample['inputs'], sample['labels']
        result, observations, shared, separate = self.history(inputs, encoded)
        mark('history')
        zero = shared.sum()*0
        terms = {}
        # Phase one supplied framewise detection/paired association only. Finish
        # the historical frontend here while retaining its original supervision.
        obs_loss = []
        for t in range(6):
            target = observation_target(labels['history_state'][t], labels['class_id'], labels['history_full_mask'][t],
                                        labels['history_visible_mask'][t], labels['history_detectable'][t])
            obs_loss.append(observation_loss(self.observation, observations[t], target)[0])
        observation = torch.stack(obs_loss).mean()
        tracking, tracking_parts, reference_ids = supervised_loss(self.tracker, result, observations, labels)
        terms.update(observation=observation, **tracking_parts)
        mark('history_supervision')
        known, ids, eligible, previous_reference, reference, presence = matched_state_supervision(result['initial_state'], reference_ids, labels)
        dynamics, dynamics_parts, rollout = self.dynamics.objective(result['tokens'], shared, result['initial_state'],
            previous_reference, reference, eligible, presence)
        terms.update(dynamics_parts)
        mark('dynamics')
        cameras = inputs['planned_camera_to_world'].flatten(0, 1)
        K = geometry_intrinsics(inputs['K'])
        with torch.no_grad():
            voxels, projection = self.project_history(inputs, observations, K, cameras)
        mark('static_projection')
        # Training uses matched reference FUTURE STATES, explicitly separated
        # from history construction; deployment instead passes the sampled state.
        states = torch.where(known[None, :, None], reference, result['initial_state'][None].expand(16, -1, -1))
        states = states[:, None].expand(-1, 2, -1, -1).flatten(0, 1)
        exists = presence[:, None].expand(-1, 2, -1).flatten(0, 1)
        geometry_labels = {}
        for key in ('isolated_depth_mm', 'full_mask', 'visible_mask'):
            value = labels[key][:, :, ids]
            value = value*known[None, None, :, None, None]
            geometry_labels[key] = value.flatten(0, 1)
        geometry_labels.update(static_depth_mm=labels['static_depth_mm'].flatten(0, 1), static_support=labels['static_support'].flatten(0, 1))
        geometry_total = zero
        for start in range(0, 32, geometry_chunk):
            end = min(32, start+geometry_chunk)
            prediction = self.geometry(shared, separate, result['tokens'], states[start:end], exists[start:end], projection[start:end], K,
                cameras[start:end], inputs['future_time_s'].repeat_interleave(2)[start:end], torch.arange(start, end, device=K.device)%2)
            loss, parts = geometry_loss(prediction, {key:value[start:end] for key, value in geometry_labels.items()})
            geometry_total = geometry_total+loss*(end-start)/32
            for key, value in parts.items():
                terms[key] = terms.get(key, zero)+value*(end-start)/32
        mark('future_surfaces')
        reader, reader_parts = self.auxiliary_loss(sample['auxiliary'], auxiliary_features)
        terms.update(reader_parts)
        mark('auxiliary_reader')
        loss = observation+tracking+dynamics+geometry_total+reader
        diagnostics = dict(tracks=len(reference_ids), matched_tracks=int(eligible.sum()),
                           eligible_reference_tracks=int(labels['history_eligible'].sum()), static_voxels=len(voxels))
        if eligible.any():
            diagnostics['anchor_position_error_m'] = float((result['initial_state'][eligible, :3]-labels['history_state'][-1, ids[eligible], :3]).norm(dim=-1).mean().detach())
            diagnostics['mode_mean_ADE_m'] = float((rollout[:, :, eligible, :3]-reference[None, :, eligible, :3]).norm(dim=-1).mean().detach())
        return loss, terms, diagnostics
