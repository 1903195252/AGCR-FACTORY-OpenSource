"""Autograd-enabled adapters for the existing JointDynamics.forward interface.

The paused source package is left intact. These entrypoints deliberately do not
accept labels, future images, camera plans, or reference presence sequences.
"""
import math

import torch
from torch.nn import functional as F


def _initial(initial, covariance, generator):
    if covariance is None:
        return initial
    if covariance.shape != (*initial.shape[:-1], 6, 6):
        raise ValueError('Initial posterior covariance must have shape [B,K,6,6]')
    eye = torch.eye(6, device=initial.device, dtype=initial.dtype)
    L = torch.linalg.cholesky(covariance + 1e-6 * eye)
    noise = torch.randn(initial.shape[:-1] + (6,), device=initial.device, dtype=initial.dtype, generator=generator)
    state = initial[..., :6] + (L @ noise[..., None]).squeeze(-1)
    return torch.cat((state, initial[..., 6:]), -1)


def _step(model, old, out, generator, dt):
    am, scale = out['acc_mean'][:, 0, -1], out['acc_scale'][:, 0, -1]
    acceleration = (am + scale * torch.randn(am.shape, device=am.device, dtype=am.dtype, generator=generator)) * model.scale.acc_std + model.scale.acc_mean
    ym, ys = out['yaw_mean'][:, 0, -1], out['yaw_scale'][:, 0, -1]
    delta = ym + ys * torch.randn(ym.shape, device=ym.device, dtype=ym.dtype, generator=generator)
    return torch.cat((old[..., :3] + dt * old[..., 3:6] + .5 * dt**2 * acceleration,
                      old[..., 3:6] + dt * acceleration,
                      ((old[..., 6] + delta + math.pi) % (2 * math.pi) - math.pi)[..., None],
                      (delta / dt)[..., None], old[..., 8:]), -1)


def persistent_mode_rollouts(model, tokens, scene, initial, initial_presence_logits,
                             generator, covariance=None, draws=4, horizon=16, dt=.2):
    """Enumerate modes; sample continuous states conditional on persistence.

    Returns state [B,M,R,T,K,11], mode_logits [B,M], log_survival [B,M,R].
    The survival factor is the probability of ALL cohort targets remaining
    present, under each free-running continuation. It is part of the density;
    no ground-truth existence mask is used to control the rollout. Appropriate
    for the controlled persistent-target cohort, not an open-entry sampler.
    """
    if initial.ndim != 3 or not initial.shape[1] or not 1 <= horizon <= 16 or draws < 1:
        raise ValueError('Nonempty batched histories, 1..16 steps, >=1 draw required')
    batch, count = initial.shape[:2]
    logits = model(tokens, scene, initial[:, None])['mode_logits']
    modes = logits.shape[-1]
    repeat = modes * draws
    expand = lambda x: x.repeat_interleave(repeat, dim=0)
    mode_ids = torch.arange(modes, device=initial.device).repeat_interleave(draws).repeat(batch)
    state = _initial(expand(initial), None if covariance is None else expand(covariance), generator)
    states = [state]
    log_survival = F.logsigmoid(expand(initial_presence_logits)).sum(-1)
    for _ in range(horizon):
        out = model(expand(tokens), expand(scene), torch.stack(states, 1), mode_ids)
        log_survival = log_survival + F.logsigmoid(out['presence_logits'][:, 0, -1]).sum(-1)
        states.append(_step(model, states[-1], out, generator, dt))
    paths = torch.stack(states[1:], 1).reshape(batch, modes, draws, horizon, count, 11)
    return dict(state=paths, mode_logits=logits, log_survival=log_survival.reshape(batch, modes, draws))


def sample_shared_future(model, tokens, scene, initial, initial_presence_logits,
                         generator, covariance=None, horizon=16, dt=.2):
    """One discrete scene mode, pathwise continuous noise, absorbing absence.

    Share this returned object across both video branches. Use discrete_log_prob
    with stochastic_feedback_surrogate for gradients through discrete draws.
    """
    if initial.ndim != 3 or not initial.shape[1] or not 1 <= horizon <= 16:
        raise ValueError('Expected a nonempty batch and 1..16 future steps')
    state = _initial(initial, covariance, generator)
    logits = model(tokens, scene, state[:, None])['mode_logits']
    modes = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
    logp = logits.log_softmax(-1).gather(1, modes[:, None]).squeeze(-1)
    alive = torch.rand(initial_presence_logits.shape, device=state.device, generator=generator) < initial_presence_logits.sigmoid()
    logp = logp - F.binary_cross_entropy_with_logits(initial_presence_logits, alive.to(state), reduction='none').sum(-1)
    states, present = [state], []
    for _ in range(horizon):
        out = model(tokens, scene, torch.stack(states, 1), modes)
        hazard = out['presence_logits'][:, 0, -1]
        draw = torch.rand(hazard.shape, device=state.device, generator=generator) < hazard.sigmoid()
        logp = logp - (F.binary_cross_entropy_with_logits(hazard, draw.to(state), reduction='none') * alive).sum(-1)
        alive = alive & draw
        new = _step(model, states[-1], out, generator, dt)
        states.append(torch.where(alive[..., None], new, states[-1]))
        present.append(alive)
    return dict(state=torch.stack(states[1:], 1), present=torch.stack(present, 1),
                mode=modes, discrete_log_prob=logp)
