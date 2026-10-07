"""Independent Base48 objectives; the frozen Pro implementation is unchanged.

Base: historical state estimation with target-marginal evidence preference,
then a frozen historical frontend supplying paired Wan flow/depth adaptation.
It does not claim the Pro sampled-video-to-state feedback mechanism. Future
references are loss targets; they never construct the video conditions.
"""
from contextlib import contextmanager
import math
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.paired_interventions.bridge import historical_posterior
from experiments.paired_interventions.data import historical_labels, replay_sample
from experiments.paired_interventions.objectives import candidate_log_density, evidence_preference
from experiments.paired_interventions.rollout import _initial, _step
from experiments.paired_interventions.system import JointSystem
from experiments.paired_interventions.video_losses import depth_loss, flow_weights
from experiments.paired_interventions.world import predict_world, query_labels, world_conditions
from experiments.performance.prefetched_replay import (
    PrefetchedReplayJointSystem,
)
from experiments.performance.execution_runtime import bounded_geometry_class


DEFAULT_VARIANT = dict(
    id='full', paired_data=True, evidence=True, evidence_weight=.1,
    evidence_draws=4, evidence_bandwidth=.2, ambiguous_nll_only=True,
    evidence_loss_scales=None, evidence_score_reduction='sum',
    ordered=True, flow_weight=1., depth_weight=1., receiver_only=False,
    free_prefix_presence_weight=1.,
)


def evidence_scores(log_density, coordinate_count, reduction='sum'):
    """Preference logits; mean scores are not calibrated log densities.

    The KDE remains a density on a complete trajectory. Dividing its logarithm
    by T*D gives a per-coordinate preference score, so the classification
    objective is not sharpened merely by a longer trajectory representation.
    Absolute target persistence is supervised separately by the common proper
    presence loss; relative candidate preference cannot replace that loss.
    """
    if reduction not in ('sum', 'mean') or type(coordinate_count) is not int or coordinate_count < 1:
        raise ValueError('Evidence scores require sum/mean and a positive coordinate count')
    return log_density / coordinate_count if reduction == 'mean' else log_density


def receiver_only_branch(branch, receiver):
    """Remove partner history at both image and observation-availability levels.

    Used only for the state-estimation control. Future labels remain supervision
    and are not conditions. Receiver eligibility/matching supervision follows
    the actually available history; the common evaluation population must still
    retain targets missed by this baseline.
    """
    if receiver not in (0, 1):
        raise ValueError('Receiver must be the aerial=0 or ground=1 view')
    inputs = dict(branch['inputs'])
    available = torch.zeros((6, 2), device=inputs['rgb'].device, dtype=torch.bool)
    available[:, receiver] = True
    if 'observation_available' in inputs:
        available &= inputs['observation_available']
    def mask(value):
        if value.shape[:2] != (6, 2):
            raise ValueError('Historical paired tensor must start with [6,2]')
        return torch.where(available.reshape(6, 2, *([1]*(value.ndim-2))), value, torch.zeros_like(value))
    for key in ('rgb', 'depth_mm', 'depth_valid'):
        inputs[key] = mask(inputs[key])
    inputs['observation_available'] = available
    labels = dict(branch['supervision'])
    for key in ('history_instance', 'history_visible_fraction', 'history_projection_area', 'history_isolated_support'):
        if key in labels:
            labels[key] = mask(labels[key])
    result = dict(branch, inputs=inputs, supervision=labels, _base48_receiver=int(receiver))
    if '_prefetched_replay' in branch:
        replay = dict(branch['_prefetched_replay'])
        past = dict(replay['labels'])
        for key in ('history_detectable', 'history_full_mask', 'history_visible_mask'):
            past[key] = mask(past[key])
        past['history_eligible'] = past['history_detectable'].any(0).any(0)
        replay['labels'] = past
        result['_prefetched_replay'] = replay
    return result


def focal_history_slot(history, reference_ids, focal):
    """Select one target slot using past confidence only; keep all model slots.

    A missing target returns -1. Duplicate associations remain in the model and
    are reported, not multiplied into this target's survival probability.
    """
    candidates = torch.where(reference_ids == int(focal))[0].tolist()
    if not candidates:
        return -1, 0
    tracks = history['result']['tracks']
    births = [tuple(int(x) for x in tracks[k]['birth']) for k in candidates]
    if len(set(births)) != len(births):
        raise ValueError('Historical track births must be unique')
    def priority(k):
        score = float(tracks[k]['score'])
        presence = float(tracks[k]['presence_logit'])
        if not math.isfinite(score) or not math.isfinite(presence):
            raise ValueError('Non-finite historical confidence')
        return (-score, -presence, tuple(int(x) for x in tracks[k]['birth']))
    return min(candidates, key=priority), len(candidates)-1


def persistent_target_rollouts(model, tokens, scene, initial, presence_logits,
                               generator, target_slot, covariance=None,
                               draws=4, horizon=16, dt=.2):
    """Free all-slot rollout, scoring survival of only the selected focal target.

    The resulting KDE is a target-marginal smoothed density, not a joint density
    that requires every predicted (possibly duplicate) slot to remain alive.
    Other targets still participate in the dynamics' interaction context.
    """
    if initial.ndim != 3 or not 0 <= target_slot < initial.shape[1] or not 1 <= horizon <= 16 or draws < 1:
        raise ValueError('Invalid focal-target rollout dimensions')
    batch, count = initial.shape[:2]
    logits = model(tokens, scene, initial[:, None])['mode_logits']
    modes = logits.shape[-1]
    repeat = modes*draws
    expand = lambda value: value.repeat_interleave(repeat, dim=0)
    mode_ids = torch.arange(modes, device=initial.device).repeat_interleave(draws).repeat(batch)
    state = _initial(expand(initial), None if covariance is None else expand(covariance), generator)
    states = [state]
    survival = F.logsigmoid(expand(presence_logits))[:, target_slot]
    for _ in range(horizon):
        out = model(expand(tokens), expand(scene), torch.stack(states, 1), mode_ids)
        survival = survival + F.logsigmoid(out['presence_logits'][:, 0, -1, target_slot])
        states.append(_step(model, states[-1], out, generator, dt))
    return dict(state=torch.stack(states[1:], 1).reshape(batch, modes, draws, horizon, count, 11),
                mode_logits=logits, log_survival=survival.reshape(batch, modes, draws))


def free_prefix_presence_objective(dynamics, history, ids, reference_present,
                                   history_eligible, generator):
    """Proper hazard BCE on history-generated prefixes, common to all controls.

    Prefix states/covariance samples and mode sampling weights are stop-gradient
    training inputs. The scored forward retains gradients through historical
    tokens, shared scene, dynamics blocks and the presence head. No future state,
    mask or image constructs a prefix. Reference existence is a loss target only.
    The absorbing inference rule is unchanged.
    """
    expected = int(history_eligible.sum())
    slots = []
    for reference in torch.unique(ids[ids >= 0]).tolist():
        if bool(history_eligible[reference]):
            slot, _ = focal_history_slot(history, ids, reference)
            slots.append(slot)
    stats = dict(future_presence_supervised_targets=len(slots),
                 future_presence_missing_history_targets=expected-len(slots),
                 future_presence_reference_targets=expected)
    if history['empty'] or not slots:
        return history['shared'].sum()*0, dict(stats, future_presence_supervised_steps=0)
    if reference_present.shape[0] != 16 or history['initial'].shape[0] != 1:
        raise ValueError('Presence adaptation requires one 16-step historical world')
    tokens, scene = history['tokens'], history['shared']
    with torch.no_grad():
        initial = history['initial'].detach()
        mixture = dynamics(tokens.detach(), scene.detach(), initial[:, None])['mode_logits'][0].softmax(-1)
        modes = mixture.numel()
        mode_ids = torch.arange(modes, device=initial.device)
        # Enumerate every mode and draw one continuous prefix per mode. The
        # covariance and innovation law is exactly that of deployed rollout.
        expand = lambda value: value.expand(modes, *value.shape[1:])
        state = _initial(expand(initial), expand(history['covariance'].detach()), generator)
        states = [state]
        for _ in range(15):
            prediction = dynamics(expand(tokens.detach()), expand(scene.detach()), torch.stack(states, 1), mode_ids)
            states.append(_step(dynamics, states[-1], prediction, generator, .2))
        previous = torch.stack(states, 1).detach()
    # Re-evaluate all time steps causally on the sampled, detached prefixes.
    # This is supervised exposure to predicted inputs, not pure-noise video
    # feedback and not a gradient that changes positions merely to remain alive.
    prediction = dynamics(tokens.expand(modes, -1, -1), scene.expand(modes, -1, -1), previous, mode_ids)
    selected = torch.tensor(slots, device=ids.device, dtype=torch.long)
    target = reference_present[:, ids[selected]].detach().bool()
    prior = torch.cat([history_eligible[ids[selected]][None], target[:-1]], 0)
    logits = prediction['presence_logits'][:, 0].index_select(-1, selected)
    bce = F.binary_cross_entropy_with_logits(logits, target[None].float().expand_as(logits), reduction='none')
    per_mode = (bce*prior[None]).sum((1, 2))/prior.sum().clamp_min(1)
    loss = (per_mode*mixture.detach()).sum()
    return loss, dict(stats, future_presence_supervised_steps=int(prior.sum()))


class Base48System(JointSystem):
    """Same learned key names as JointSystem, independently versioned behavior.

    API: phase is state/depth/video; forward returns loss and scalar metrics.
    State may receive a complete audited family for the early evidence loss.
    Video freezes the upstream frontend and uses one flow evaluation. The
    clean-latent depth loss trains a decoder and is explicitly teacher-forced;
    it is not pure-noise generated-video feedback or an end-to-end state loss.
    """
    def __init__(self, paths, device, phase, variant=None):
        if phase not in ('state', 'depth', 'video'):
            raise ValueError('Base48 phase must be state, depth or video')
        self.base_phase = phase
        self.profile = False
        self.profile_times = {}
        unknown = set(variant or {}) - set(DEFAULT_VARIANT)
        if unknown:
            raise ValueError('Unknown Base loss/configuration options: '+', '.join(sorted(unknown)))
        config = dict(DEFAULT_VARIANT, **(variant or {}))
        if not config['ordered']:
            raise ValueError('Base48 keeps ordered appearance gating fixed across comparisons')
        if config['receiver_only'] and (phase != 'state' or config['evidence']):
            raise ValueError('Receiver-only is a state control without paired-evidence training')
        if config['evidence_draws'] < 1 or config['evidence_bandwidth'] <= 0:
            raise ValueError('Invalid evidence density configuration')
        if config['evidence_score_reduction'] not in ('sum', 'mean'):
            raise ValueError('Evidence score reduction must be sum or mean')
        loss_scales = config['evidence_loss_scales']
        if loss_scales is not None and (not isinstance(loss_scales, (list, tuple)) or len(loss_scales) != 6
                or any(not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s) or s <= 0
                       for s in loss_scales)):
            raise ValueError('Evidence loss scales require six positive finite training values')
        if any(config[k] < 0 for k in ('evidence_weight', 'flow_weight', 'depth_weight')):
            raise ValueError('Loss weights must be nonnegative')
        if not math.isfinite(float(config['free_prefix_presence_weight'])) or config['free_prefix_presence_weight'] <= 0:
            raise ValueError('Every Base48 state control requires the common positive free-prefix presence weight')
        super().__init__(paths, device, 'dynamics' if phase == 'state' else phase, config)
        if phase == 'state':
            # Already accepted execution-only optimization; child modes remain
            # unchanged, and there are no new parameters or checkpoint keys.
            self.upstream.geometry.__class__ = bounded_geometry_class()

    def set_phase(self, phase, feedback_phase=False):
        if feedback_phase:
            raise ValueError('End-to-end denoising feedback belongs to Pro, not Base48')
        self.upstream.requires_grad_(self.base_phase == 'state')
        self.reader.requires_grad_(self.base_phase == 'state')
        self.condition.requires_grad_(self.base_phase in ('depth', 'video'))
        self.depth.requires_grad_(self.base_phase in ('depth', 'video'))
        self.upstream.train(self.base_phase == 'state')
        self.reader.train(self.base_phase == 'state')
        if self.wan is not None:
            self.wan.train()

    def train(self, mode=True):
        super().train(mode)
        # Calling wrapper.train() must not reactivate frontend dropout during
        # the frozen video/depth stage.
        if self.base_phase != 'state':
            self.upstream.eval()
            self.reader.eval()
        return self

    def trainable_groups(self):
        groups = dict(upstream=[], lora=[], new=[])
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                group = 'upstream' if name.startswith('upstream.') else 'lora' if 'lora_' in name else 'new'
                groups[group].append(parameter)
        return [dict(params=values, lr=1e-4 if name == 'new' else 1e-5, name=name)
                for name, values in groups.items() if values]

    @contextmanager
    def timed(self, name):
        if not self.profile:
            yield
            return
        device = self.upstream.tracker.scale.std.device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.monotonic()
        try:
            yield
        finally:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            self.profile_times[name] = self.profile_times.get(name, 0.) + time.monotonic()-start

    @torch.no_grad()
    def infer_condition(self, inputs, generator):
        """The public video-condition boundary accepts no supervision argument."""
        encoded = self.encoders.history(inputs['rgb'])
        history, draw = predict_world(self.upstream, inputs, encoded, generator)
        condition = world_conditions(self.upstream, history, draw, inputs)
        return encoded, history, draw, condition

    def reference_condition(self, *args, **kwargs):
        raise RuntimeError('Base48 never constructs a video condition from future reference states')

    def evidence(self, family, generator):
        zero = self.upstream.tracker.scale.std.sum()*0
        if family is None or family['metadata']['cohort'] != 'evidence':
            return zero, dict(evidence_covered=0, evidence_total=0, evidence_missing=0,
                              evidence_duplicate_slots=0)
        losses = []
        covered = duplicates = 0
        focal = int(family['metadata']['target_index'])
        # Future trajectories are detached scoring references, never history
        # inputs, rollout controls, track births or video conditions.
        references = torch.stack([family['branches'][(w, 0)]['supervision']['state'][:, focal, :6]
                                  for w in range(2)]).detach()
        for world in range(2):
            branch = family['branches'][(world, 0)]
            encoded = self.encoders.history(branch['inputs']['rgb'])
            history = historical_posterior(self.upstream, branch['inputs'], encoded)
            with torch.no_grad():
                ids = query_labels(history, historical_labels(branch))
                slot, extra = focal_history_slot(history, ids, focal)
            duplicates += extra
            if history['empty'] or slot < 0:
                losses.append(history['shared'].sum()*0)
                continue
            predicted = persistent_target_rollouts(
                self.upstream.dynamics, history['tokens'], history['shared'], history['initial'],
                history['presence_logits'], generator, target_slot=slot,
                covariance=history['covariance'], draws=int(self.variant['evidence_draws']))
            density = candidate_log_density(predicted['state'][..., slot, :6], predicted['mode_logits'],
                references[None], (self.variant['evidence_loss_scales'] if self.variant['evidence_loss_scales'] is not None
                                   else self.upstream.tracker.scale.std[:6]),
                float(self.variant['evidence_bandwidth']), predicted['log_survival'])
            scores = evidence_scores(density, references.shape[-2]*references.shape[-1],
                                     self.variant['evidence_score_reduction'])
            losses.append(evidence_preference(scores, torch.tensor([world], device=density.device)))
            covered += 1
        # A miss stays in the two-world denominator and is reported. It is not
        # replaced by a reference track or omitted as a successful prediction.
        value = torch.stack(losses).sum()/2
        return value, dict(evidence_covered=covered, evidence_total=2, evidence_missing=2-covered,
                          evidence_duplicate_slots=duplicates, raw_evidence_loss=value.detach())

    def flow_objective(self, branch, condition, latent, generator):
        s = torch.randn((), device=latent.device, generator=generator).sigmoid()
        s = 5*s/(1+4*s)
        noise = torch.randn(latent[:, :, 1:].shape, device=latent.device,
                            dtype=latent.dtype, generator=generator)
        corrupted = torch.cat([latent[:, :, :1], (1-s)*latent[:, :, 1:]+s*noise], 2)
        velocity = self.wan(corrupted, s, condition)
        weight, domain = flow_weights(branch)
        return ((velocity[:, :, 1:]-(noise-latent[:, :, 1:])).square()*weight).sum() / (domain.sum()*48).clamp_min(1)

    def clean_depth_objective(self, branch, condition, latent):
        """Clean reference latent supervises depth; historical geometry conditions it.

        Frozen VAE feature extraction is recomputed at the outer checkpoint
        boundary to avoid retaining two native feature pyramids. No gradients
        reach RGB flow latents or the frozen state model from this loss.
        """
        geometry = self.condition.depth_features(condition)
        y = branch['supervision']
        losses = []
        for view in range(2):
            def one(z, geom, reference, valid, view=view):
                with torch.no_grad():
                    _, *features = self.vae.decode(z.float())
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=z.is_cuda):
                    output = self.depth(z.float(), features, geom, view)
                return depth_loss(output['log_depth'][0, 1:, :, 8:-8].float(),
                    output['validity_logits'][0, 1:, :, 8:-8].float(), reference, valid)
            args = (latent[view:view+1].detach(), geometry[view:view+1], y['depth_mm'][:, view], y['valid'][:, view])
            losses.append(checkpoint(one, *args, use_reentrant=False) if torch.is_grad_enabled() else one(*args))
        return torch.stack(losses).mean()

    def forward(self, branch, generator, update=0, family=None):
        self.profile_times = {}
        if self.base_phase == 'state':
            if self.variant.get('receiver_only', False):
                receiver = family['metadata']['receiver'] if family is not None else branch.get('_base48_receiver')
                if receiver is None:
                    raise ValueError('Ordinary receiver-only branch needs a registered balanced receiver assignment')
                branch = receiver_only_branch(branch, int(receiver))
            with self.timed('state_objective'):
                # Reuse this exact historical graph for the ordinary objective
                # and the common deployment-prefix presence supervision.
                encoded, history, _, _ = self.prepared(branch, generator, True, build_condition=False)
                cache = branch.get('_prefetched_replay')
                if cache is not None:
                    if cache['offset'] != int(update) % 16:
                        raise ValueError('Replay prefetch offset changed')
                    replay = dict(inputs=branch['inputs'], labels=cache['labels'], auxiliary=cache['auxiliary'])
                else:
                    replay = replay_sample(branch, update % 16)
                ids = query_labels(history, historical_labels(branch))
                auxiliary = self.encoders.features(replay['auxiliary']['rgb'])
                past = (history['result'], history['observations'], history['shared'], history['separate'])
                loss, parts, _ = self.upstream.objective(replay, encoded, auxiliary, geometry_chunk=8, history_result=past)
                if branch.get('cohort') == 'ambiguous' and self.variant['ambiguous_nll_only']:
                    loss = loss-.1*parts['rollout_l1']
                reader = self._reader_with_features(branch, history, ids, update % 16, auxiliary)
                loss = loss+reader
                parts = dict(parts, dense_reader=reader, predicted_tracks=len(ids), matched_tracks=int((ids >= 0).sum()))
            with self.timed('free_prefix_presence'):
                presence, coverage = free_prefix_presence_objective(
                    self.upstream.dynamics, history, ids, replay['labels']['present'],
                    replay['labels']['history_eligible'], generator)
            loss = loss+float(self.variant['free_prefix_presence_weight'])*presence
            parts = dict(parts, free_prefix_presence_BCE=presence, **coverage)
            if self.variant['evidence'] and family is not None:
                with self.timed('paired_evidence'):
                    evidence, coverage = self.evidence(family, generator)
                loss = loss+float(self.variant['evidence_weight'])*evidence
                parts = dict(parts, **coverage, evidence=evidence)
            return loss, parts
        with self.timed('history_condition'):
            _, history, _, condition = self.infer_condition(branch['inputs'], generator)
        with self.timed('reference_vae_encode'):
            rgb = torch.cat([branch['inputs']['rgb'][-1:], branch['supervision']['rgb']], 0)
            latent = self.vae.encode(rgb)
        if latent.shape != (2, 48, 5, 30, 46):
            raise ValueError('Base48 requires native paired 720x480 / 6-history / 16-future latents')
        stats = dict(predicted_tracks=len(history['result']['tracks']))
        with self.timed('clean_depth_forward'):
            depth = self.clean_depth_objective(branch, condition, latent)
        if self.base_phase == 'depth':
            return depth, dict(stats, depth=depth)
        with self.timed('flow_forward'):
            flow = self.flow_objective(branch, condition, latent, generator)
        return float(self.variant['flow_weight'])*flow+float(self.variant['depth_weight'])*depth, dict(stats, flow=flow, depth=depth)

    # The accepted replay subclass shares the same forward implementation but
    # invokes this helper through self. Keep reuse without adding model keys.
    _reader_with_features = PrefetchedReplayJointSystem._reader_with_features
    reuse_reader_features = True
