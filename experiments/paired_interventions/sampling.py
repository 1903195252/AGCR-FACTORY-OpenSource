"""Pure-noise rollout with a documented truncated gradient, not target noising."""
from contextlib import nullcontext

import torch


def pure_noise_euler(velocity, noise, anchor, condition, steps=50, grad_tail=2):
    """Velocity(z, time, condition) -> z-shaped flow; input time axis is dim 1.

    noise is a sampled future latent (must not be an encoded target clip).
    condition is a flat tensor dict built from HISTORY and predicted states.
    First steps-grad_tail updates are detached; the tail trains conditions,
    LoRA/adapters and upstream differentiable state. Frozen decoding afterward
    must retain autograd. Caller is responsible for pure-noise provenance.
    """
    if not 1 <= grad_tail <= steps or anchor.shape[1] != 1:
        raise ValueError('Invalid gradient tail or clean anchor')
    if noise.shape[0] != anchor.shape[0] or noise.shape[2:] != anchor.shape[2:]:
        raise ValueError('Anchor/future latent shapes disagree')
    if any(not isinstance(v, torch.Tensor) for v in condition.values()):
        raise TypeError('Conditions must be a flat tensor dictionary')
    rho = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=noise.dtype)
    times = 5 * rho / (1 + 4 * rho)
    z = torch.cat((anchor.detach(), noise.detach()), 1)
    for n in range(steps):
        keep_grad = n >= steps - grad_tail
        if n == steps - grad_tail:
            z = z.detach()
        values = condition if keep_grad else {k: v.detach() for k, v in condition.items()}
        with nullcontext() if keep_grad else torch.no_grad():
            flow = velocity(z, times[n], values)
            if flow.shape != z.shape:
                raise ValueError('Flow shape does not match latent shape')
            future = z[:, 1:] + (times[n+1] - times[n]) * flow[:, 1:]
            z = torch.cat((anchor.detach(), future), 1)
    return z
