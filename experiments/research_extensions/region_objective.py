"""Isolated reappearance supervision for a bounded renderer candidate.

The original Base48 flow objective, random stream, video VAE and conditioning
are preserved. Future instance masks select loss support only. This module adds
no parameters/buffers and does not change sampling or the historical frontend.
"""
import math

import torch

from experiments.base48.model import Base48System
from experiments.paired_interventions.video_losses import flow_weights
from experiments.research_extensions.region_probe import pooled_region, reappearance_frames


@torch.no_grad()
def reappearance_events(supervision):
    """Return distinct (target, view, onset) events and their supervised frames.

    Onsets use the existing probe's hidden-to-detectable definition. Each event
    covers at most the three native frames starting at its onset. A subsequent
    qualifying onset owns its own frames, so overlapping three-frame windows
    never count the same target/view/frame twice. Invisible/absent frames and
    frames with no pixels of the reference instance contribute no support.

    No future motion/state/identity descriptor is passed to the predictor; the
    simulator instance ID below is an index for supervision masks only.
    """
    y = supervision
    instance, ids = y['future_instance'], y['instance_id']
    if instance.ndim != 4 or instance.shape[1] != 2:
        raise ValueError('Future instance masks must have shape [T,2,H,W]')
    if ids.ndim != 1 or len(torch.unique(ids)) != len(ids) or (ids <= 0).any():
        raise ValueError('Positive unique reference instance IDs required')
    expected = (*instance.shape[:2], len(ids))
    if y['visible_fraction'].shape != expected or y['projection_area'].shape != expected:
        raise ValueError('Future visibility and projection area must be [T,2,K]')
    present = y['present'].bool()
    visible = (y['visible_fraction'] >= .1) & (y['projection_area'] >= 144)
    last_visible = ((y['history_visible_fraction'][-1] >= .1)
                    & (y['history_projection_area'][-1] >= 144))
    _, starts = reappearance_frames(visible, y['projection_area'], last_visible,
                                    y['history_projection_area'][-1], present)
    events = []
    for target_index, instance_id in enumerate(ids.tolist()):
        for view in range(2):
            onsets = torch.where(starts[:, view, target_index])[0].tolist()
            for position, onset in enumerate(onsets):
                following = onsets[position + 1] if position + 1 < len(onsets) else len(instance)
                end = min(onset + 3, following, len(instance))
                frames = [frame for frame in range(onset, end)
                          if bool(visible[frame, view, target_index])
                          and bool(present[frame, target_index])
                          and bool((instance[frame, view] == instance_id).any())]
                events.append(dict(target_index=target_index, instance_id=int(instance_id),
                                   view=view, onset=int(onset), frames=frames))
    return events


def event_terms(events):
    return dict(reappearance_starts=len(events),
                reappearance_events=sum(bool(event['frames']) for event in events),
                reappearance_empty_events=sum(not event['frames'] for event in events),
                reappearance_event_frames=sum(len(event['frames']) for event in events))


def reappearance_mse(error_squared, supervision, events=None):
    """Mean of per-event means of independently normalized native-frame MSEs.

    A frame mask is zero at every other time/view, then padded and pooled into
    the unchanged future flow latent grid. Each frame is divided by its own
    pooled coverage mass times the latent channel count. Thus large targets or
    longer visible runs do not outweigh a short/small reappearance event.

    The VAE mixes time and pixels: this is fractional latent loss support, not
    decoded RGB error. In particular frames in one four-frame latent bin can
    share an error tensor; normalization does not create extra temporal detail.
    """
    if error_squared.ndim != 5 or error_squared.shape[0] != 2:
        raise ValueError('Squared future flow error must have shape [2,C,T,H,W]')
    events = reappearance_events(supervision) if events is None else events
    instance = supervision['future_instance']
    channels = error_squared.shape[1]
    if not channels:
        raise ValueError('A positive number of latent channels is required')
    losses = []
    for event in events:
        frames = []
        for frame in event['frames']:
            with torch.no_grad():
                mask = torch.zeros_like(instance, dtype=torch.bool)
                mask[frame, event['view']] = instance[frame, event['view']] == event['instance_id']
                region = pooled_region(mask, error_squared.shape[2:]).to(error_squared.device)
                denominator = region.sum() * channels
            if not bool(denominator > 0):
                raise ValueError('A nonempty native event frame lost all pooled support')
            frames.append((error_squared * region).sum() / denominator)
        if frames:
            losses.append(torch.stack(frames).mean())
    loss = torch.stack(losses).mean() if losses else error_squared.sum() * 0
    return loss, event_terms(events)


class ReappearanceSystem(Base48System):
    """Base48-compatible renderer with one supervised reappearance loss."""
    def __init__(self, paths, device, phase, variant=None, reappearance_weight=.05):
        value = float(reappearance_weight)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Reappearance weight must be finite and nonnegative')
        super().__init__(paths, device, phase, variant)
        self.reappearance_weight = value
        self.last_region_terms = {}

    def flow_objective(self, branch, condition, latent, generator):
        # Masks only enter loss accounting below, never self.wan's condition.
        # The control delegates to the original method with no added RNG calls,
        # forward calls, scalar arithmetic or autograd operations on its result.
        if self.reappearance_weight == 0:
            original = super().flow_objective(branch, condition, latent, generator)
            events = reappearance_events(branch['supervision'])
            self.last_region_terms = dict(event_terms(events), original_flow=float(original.detach()),
                                          reappearance_mse=None, reappearance_loss=0.,
                                          reappearance_weight=0.)
            return original

        # Exact Base48 s/noise/corruption/velocity/error and denominator. Keep
        # this separate from the fixed-t, independent-noise diagnostic probe.
        s = torch.randn((), device=latent.device, generator=generator).sigmoid()
        s = 5*s/(1+4*s)
        noise = torch.randn(latent[:, :, 1:].shape, device=latent.device,
                            dtype=latent.dtype, generator=generator)
        corrupted = torch.cat([latent[:, :, :1], (1-s)*latent[:, :, 1:]+s*noise], 2)
        velocity = self.wan(corrupted, s, condition)
        error = (velocity[:, :, 1:]-(noise-latent[:, :, 1:])).square()
        weight, domain = flow_weights(branch)
        original = (error*weight).sum() / (domain.sum()*48).clamp_min(1)
        event_loss, counts = reappearance_mse(error, branch['supervision'])
        additional = self.reappearance_weight * event_loss
        self.last_region_terms = dict(counts, original_flow=float(original.detach()),
                                      reappearance_mse=float(event_loss.detach()),
                                      reappearance_loss=float(additional.detach()),
                                      reappearance_weight=self.reappearance_weight)
        return original + additional
