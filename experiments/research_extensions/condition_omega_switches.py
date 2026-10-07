"""Default-off, independent angular-rate interventions for CPU diagnosis.

No runner installs this module automatically. Neither switch changes a world
state, rollout, weights, sampling seeds, or the other condition pathway.
New sampled outputs must record these switches and use a new method identity.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from .geometry_alignment_repair import instantaneous_geometry


def renderer_condition(condition, *, neutralize_omega=False):
    """Copy only renderer's encoded omega; keep the predicted world untouched."""
    if not neutralize_omega:
        return condition
    encoded = condition['normalized_state']
    if encoded.shape[-1] != 12:
        raise ValueError('Expected p,u,sin/cos yaw,omega,dimensions: 12 channels')
    normalized = torch.cat((encoded[..., :8], torch.zeros_like(encoded[..., 8:9]),
                            encoded[..., 9:]), -1)
    return dict(condition, normalized_state=normalized)


@contextmanager
def conditioning_switches(*, geometry=None, encoder=None,
                         neutralize_geometry_omega=False,
                         neutralize_renderer_omega=False):
    """Four explicit conditions: off, geometry-only, renderer-only, or both.

    Geometry-only intercepts its independent StateScale. Renderer-only changes
    a shallow condition copy at ConditionEncoder.forward; Z/P/M and the original
    normalized_state/world state remain unchanged. State uncertainty and all
    non-omega channels remain available. Context exit restores both modules.
    """
    if neutralize_geometry_omega and geometry is None:
        raise ValueError('Geometry intervention requires the explicit module')
    if neutralize_renderer_omega and encoder is None:
        raise ValueError('Renderer intervention requires the explicit encoder')
    with ExitStack() as stack:
        if neutralize_geometry_omega:
            stack.enter_context(instantaneous_geometry(geometry))
        if neutralize_renderer_omega:
            original = encoder.forward

            def forward(condition):
                return original(renderer_condition(condition, neutralize_omega=True))

            stack.enter_context(patch.object(encoder, 'forward', forward))
        yield dict(geometry_omega_neutralized=neutralize_geometry_omega,
                   renderer_omega_neutralized=neutralize_renderer_omega,
                   world_state_modified=False, training_updates=0)
