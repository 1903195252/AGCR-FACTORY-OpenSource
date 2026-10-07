"""Losses with an explicit boundary between predictions and supervision.

No function in this module reads images, reference states, or simulator IDs to
construct a forecast. References are detached loss targets only.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def candidate_log_density(paths, mode_logits, references, scales, bandwidth,
                          log_survival=None):
    """Kernel-smoothed predictive density at two reference continuations.

    paths: [B,M,R,T,D], free-running draws in each scene mode.
    references: [B,2,T,D]; scales: [D], fitted on training data only.
    log_survival: [B,M,R], probability of the persistent-scene event.
    Returns [B,2]. This is a smoothed density, not exact trajectory likelihood
    or a calibrated probability that one of the two candidates must occur.
    """
    if paths.ndim != 5 or references.shape != (paths.shape[0], 2, *paths.shape[-2:]):
        raise ValueError('Expected paths [B,M,R,T,D], references [B,2,T,D]')
    if mode_logits.shape != paths.shape[:2] or paths.shape[2] < 1:
        raise ValueError('Invalid mixture/sample dimensions')
    scales = torch.as_tensor(scales, device=paths.device, dtype=paths.dtype).detach()
    if scales.shape != paths.shape[-1:] or not torch.isfinite(scales).all() or (scales <= 0).any() or bandwidth <= 0:
        raise ValueError('Positive finite training scales and bandwidth required')
    difference = (paths[:, None] - references.detach()[:, :, None, None]) / scales
    dimensions = paths.shape[-1] * paths.shape[-2]
    kernel = -.5 * (difference / bandwidth).square().sum((-1, -2))
    kernel = kernel - dimensions * math.log(bandwidth * math.sqrt(2 * math.pi))
    if log_survival is not None:
        if log_survival.shape != paths.shape[:3]:
            raise ValueError('Invalid survival dimensions')
        kernel = kernel + log_survival[:, None]
    log_weights = mode_logits.log_softmax(-1)[:, None, :, None]
    return torch.logsumexp((kernel + log_weights).flatten(2), -1) - math.log(paths.shape[2])


def evidence_preference(scores, labels):
    """Two-candidate cross entropy; apply only to evidence-informative pairs."""
    if scores.ndim != 2 or scores.shape[1] != 2 or labels.shape != scores.shape[:1]:
        raise ValueError('Expected [B,2] scores and [B] world labels')
    return F.cross_entropy(scores, labels.detach().long())


def dynamics_replay_loss(base_terms, ambiguous=False):
    """Retain the mixture likelihood, without mean collapse, on ambiguous data.

    The existing mean-per-mode L1 pulls every mode toward one realized future;
    it must not supervise the ambiguous cohort. Inputs are the loss dictionary
    returned by JointDynamics.objective, not forecast-conditioning labels.
    """
    nll=base_terms['dynamics_nll']
    return nll if ambiguous else nll+.1*base_terms['rollout_l1']


def masked_mean(values, reference_mask):
    weights = reference_mask.detach().to(values)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def camera_readout_consistency(centers, descriptors, reference_visible, scales):
    """Two camera interventions, arbitrary number of views.

    centers [B,2,V,T,K,3], descriptors [B,2,V,T,K,C]. Readouts must be
    dense history-query outputs, not thresholded detections: reference-visible
    misses cannot be dropped. Coordinates are in the same fixed world frame.
    Supervised visible-mask loss is separate and is required to prevent empty
    or mutually consistent but wrong outputs.
    """
    if centers.ndim != 6 or centers.shape[1] != 2 or centers.shape[-1] != 3:
        raise ValueError('Expected centers [B,2,V,T,K,3]')
    if reference_visible.shape != centers.shape[:-1] or descriptors.shape[:-1] != centers.shape[:-1]:
        raise ValueError('Readout/visibility shape mismatch')
    visible = reference_visible.detach().bool()
    joint = visible[:, 0] & visible[:, 1]
    delta = ((centers[:, 0] - centers[:, 1]) / scales).abs().mean(-1)
    identity = 1 - F.cosine_similarity(descriptors[:, 0], descriptors[:, 1], dim=-1)
    return masked_mean(delta + identity, joint)


def reappearance_feedback(centers, descriptors, mask_logits, target_centers,
                          history_descriptors, target_masks, reference_visible,
                          event_window, scales):
    """Dense frozen-reader loss on a pure-noise generated trajectory.

    Prefix dimensions are [B,view,T,K]. The event window includes the hidden
    interval and first three visible frames; mask loss includes hidden frames.
    No generated detection threshold chooses supervised elements.
    """
    window = event_window.detach().bool()
    active = window & reference_visible.detach().bool()
    center_error = ((centers - target_centers.detach()) / scales).abs().mean(-1)
    identity = 1 - F.cosine_similarity(descriptors, history_descriptors.detach(), dim=-1)
    mask_error = F.binary_cross_entropy_with_logits(mask_logits, target_masks.detach().to(mask_logits), reduction='none').mean((-1, -2))
    return masked_mean(center_error + identity, active) + masked_mean(mask_error, window)


def stochastic_feedback_surrogate(per_draw_loss, discrete_log_prob, baseline):
    """Pathwise gradient plus REINFORCE for mode/presence, without value shift.

    The baseline must be from earlier updates or independent samples. Do not
    compute it from the current single draw. Frozen reader/VAE parameters do
    not imply torch.no_grad() on their forward pass.
    """
    if per_draw_loss.shape != discrete_log_prob.shape:
        raise ValueError('One loss and discrete log probability per draw required')
    advantage = (per_draw_loss - torch.as_tensor(baseline, device=per_draw_loss.device)).detach()
    score = advantage * discrete_log_prob
    return (per_draw_loss + score - score.detach()).mean()


class OrderedAppearanceGate(nn.Module):
    """Four visibility-gated appearance slots -> one latent condition.

    features [...,4,C], masks [...,4] on the SAME token grid. Features must be
    built by a shared framewise map, before adding within-group time codes;
    partner appearance evidence belongs in those features, not a bypass.
    The mean control pools BOTH masked features and masks before time encoding
    and uses the same mixer parameters and dimensions as the ordered variant.
    Its complete mixer input is invariant to jointly permuting feature/mask
    pairs. This does not remove time information from other model pathways or
    from an upstream feature that already encodes order. All-hidden inputs
    produce exactly zero, including after training. No VAE changes are required.
    """
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.time = nn.Parameter(torch.randn(4, width) * .02)
        self.in_proj = nn.Linear(4 * (width + 1), width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)

    def condition_slots(self, features, masks, ordered=True):
        """Return the entire [...,4,C+1] input to the learned mixer.

        Mean control: g_bar = mean_r(m_r * f_r), m_bar = mean_r(m_r),
        q_r = [g_bar + m_bar * e_r, m_bar]. The fixed destination-slot codes
        e_r never receive the original frame-specific masks in this control.
        Exposing this boundary permits checks before learned projections can
        hide a leaking input (for example, at zero-initialized output weights).
        """
        if features.shape[-2:] != (4, self.width) or masks.shape != features.shape[:-1]:
            raise ValueError('Expected four feature and visibility slots')
        if not torch.isfinite(masks).all() or (masks < 0).any() or (masks > 1).any():
            raise ValueError('Visibility masks must be finite in [0,1]')
        gated = features * masks[..., None]
        if not ordered:
            gated = gated.mean(-2, keepdim=True).expand_as(gated)
            masks = masks.mean(-1, keepdim=True).expand_as(masks)
        return torch.cat((gated + masks[..., None] * self.time, masks[..., None]), -1)

    def forward(self, features, masks, ordered=True):
        encoded = self.condition_slots(features, masks, ordered=ordered)
        return self.out_proj(F.gelu(self.in_proj(encoded.flatten(-2))))
