"""Historical identity adaptation, isolated from the immutable surface pilot.

The residual query projection starts at the identity map. All four experimental
arms have exactly the same architecture and trainable parameter set. The frozen
encoder cache contains observations only; labels enter objectives after tracking.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .history_surface_refiner import HistorySurfaceRefiner, install_history_surface
from .train_history_surface import measurement_objective, observations

TRACKER_MODULES = ('association', 'reliability', 'memory', 'token', 'presence',
                   'readout', 'appearance')


class HistoryIdentityRefiner(HistorySurfaceRefiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity_residual = nn.Linear(self.width, self.width)
        nn.init.zeros_(self.identity_residual.weight)
        nn.init.zeros_(self.identity_residual.bias)

    def forward(self, pair, *args):
        # Native support and measurement use unchanged parent queries. Adapted
        # identity tokens enter temporal/birth association and recurrent memory.
        refined, auxiliary = super().forward(pair, *args)
        refined['tokens'] = pair['tokens'] + self.identity_residual(pair['tokens'])
        return refined, auxiliary


def historical_parameters(system, refiner, enable=True):
    system.eval().requires_grad_(False)
    refiner.requires_grad_(enable)
    for name in TRACKER_MODULES:
        getattr(system.tracker, name).requires_grad_(enable)
    system.tracker.log_process.requires_grad_(enable)
    system.observation.association.requires_grad_(enable)
    values = list(refiner.parameters()) + [system.tracker.log_process]
    for name in TRACKER_MODULES:
        values.extend(getattr(system.tracker, name).parameters())
    values.extend(system.observation.association.parameters())
    if len({id(p) for p in values}) != len(values):
        raise ValueError('Unexpected shared trainable parameter')
    return values


def overlay_keys(system):
    prefixes = tuple('tracker.'+name+'.' for name in TRACKER_MODULES)
    return {key for key in system.state_dict()
            if key == 'tracker.log_process' or key.startswith(prefixes)
            or key.startswith('observation.association.')}


def overlay_tensors(system):
    return {key: value.detach().cpu().clone() for key, value in system.state_dict().items()
            if key in overlay_keys(system)}


def install_identity_candidate(system, refiner, overlay, record_aux=False):
    expected = overlay_keys(system)
    current = system.state_dict()
    if set(overlay) != expected:
        raise ValueError('Historical identity overlay tensor keys differ')
    for key, value in overlay.items():
        if (not isinstance(value, torch.Tensor) or value.shape != current[key].shape
                or value.dtype != current[key].dtype or not torch.isfinite(value).all()):
            raise ValueError('Invalid historical identity tensor: '+key)
    with torch.no_grad():
        for key, value in overlay.items():
            current[key].copy_(value.to(current[key]))
    install_history_surface(system, refiner, record_aux=record_aux)
    return system


def birth_association_loss(system, obs, labels):
    """Supervise same-frame cross-view births absent from temporal loss.

    Selection matches inference's >.5 predicted foreground confidence. Labels
    assign these already-predicted queries only for this post-forward BCE.
    """
    from experiments.paper_experiments.agcr_exp.temporal_state import historical_assignment
    mapping = historical_assignment(obs, labels)
    losses = []
    for t, pair in enumerate(obs):
        selected = [torch.where(pair['class_logits'][v].softmax(-1)[:, 1:].amax(-1).detach() > .5)[0]
                    for v in range(2)]
        if not all(len(ids) for ids in selected):
            continue
        sides = [{key: pair[key][v, selected[v]]
                  for key in ('tokens', 'center', 'covariance', 'class_logits')}
                 for v in range(2)]
        logits = system.observation.association_logits(*sides)
        a, g = [mapping[t][v][selected[v]] for v in range(2)]
        target = ((a[:, None] == g[None]) & (a[:, None] >= 0)).to(logits)
        losses.append(F.binary_cross_entropy_with_logits(logits, target))
    return torch.stack(losses).mean() if losses else obs[0]['tokens'].sum()*0


def identity_objective(system, refiner, row, objective, fit_tracker=True):
    from experiments.paper_experiments.agcr_exp.temporal_state import supervised_loss
    if objective not in ('posterior', 'full'):
        raise ValueError('Unknown historical objective')
    x, labels, pair = row['inputs'], row['labels'], row['pair']
    refined, auxiliary = refiner(pair, x['rgb'].flatten(0, 1), x['depth_mm'].flatten(0, 1),
        x['depth_valid'].flatten(0, 1), x['K'], x['camera_to_world'].flatten(0, 1))
    loss, parts, diagnostics = measurement_objective(refiner, pair, refined, auxiliary,
                                                     labels, system.tracker.scale.std[:3])
    zero = loss*0
    for key in ('association', 'posterior', 'state_readout', 'presence', 'reliability',
                'identity', 'appearance_identity', 'birth_association'):
        parts[key] = zero
    if fit_tracker:
        obs = observations(refined)
        result = system.tracker(obs, system.observation, row['encoded']['dino'], row['encoded']['vae'],
                                x['camera_to_world'], x['time_s'])
        tracking_loss, tracking_parts, ids = supervised_loss(system.tracker, result, obs, labels)
        parts.update(tracking_parts)
        parts['birth_association'] = birth_association_loss(system, obs, labels)
        loss = loss + (tracking_loss+parts['birth_association'] if objective == 'full' else parts['posterior'])
        known = ids >= 0
        diagnostics.update(predicted_slots=len(ids), matched_slots=int(known.sum()),
            matched_unique_reference_targets=len(set(ids[known].tolist())),
            unassociated_slots=int((~known).sum()),
            duplicate_slots=int(known.sum())-len(set(ids[known].tolist())),
            eligible_reference_targets=int(labels['history_detectable'].any(0).any(0).sum()))
        if known.any():
            delta = result['initial_state'][known, :6]-labels['history_state'][-1, ids[known], :6]
            diagnostics.update(posterior_position_error_m=float(delta[:, :3].norm(dim=-1).mean().detach()),
                posterior_velocity_error_m_s=float(delta[:, 3:6].norm(dim=-1).mean().detach()))
    diagnostics['optimized_tracking_terms'] = (['posterior'] if objective == 'posterior' else
        ['association', 'posterior', 'state_readout', 'presence', 'reliability', 'identity',
         'appearance_identity', 'birth_association']) if fit_tracker else []
    return loss, parts, diagnostics
