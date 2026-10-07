"""History-only connection to the existing StageTwo model, without checkpoint mutation."""
import torch

from .objectives import candidate_log_density, evidence_preference
from .rollout import persistent_mode_rollouts, sample_shared_future


def historical_posterior(system, inputs, encoded):
    """Run the real observation/filter path with autograd; no future labels."""
    result, observations, shared, separate = system.history(inputs,encoded)
    tracks = result['tracks']
    if not tracks:
        # Do not insert a ground-truth track into a missing predicted track set.
        return dict(result=result,observations=observations,shared=shared,separate=separate,empty=True)
    return dict(result=result,observations=observations,shared=shared,separate=separate,empty=False,
                tokens=result['tokens'][None],initial=result['initial_state'][None],
                covariance=torch.stack([item['covariance'] for item in tracks])[None],
                presence_logits=torch.stack([item['presence_logit'] for item in tracks])[None])


def forecast_from_history(system, inputs, encoded, generator):
    """Returns differentiable state and historical features for the video adapter.

    The complete Wan/depth/reader training path is in system.JointSystem.
    """
    history=historical_posterior(system,inputs,encoded)
    if history['empty']:
        return dict(history=history,draw=None)
    draw=sample_shared_future(system.dynamics,history['tokens'],history['shared'],history['initial'],
                              history['presence_logits'],generator,covariance=history['covariance'])
    return dict(history=history,draw=draw)


def evidence_pair_objective(system, inputs, encoded, history_track_indices,
                            reference_paths, scales, generator, bandwidth=.2, draws=4):
    """A complete differentiable evidence-loss call for an audited two-world pair.

    inputs/encoded each contain two HISTORICAL records; reference_paths [2,16,6]
    is used only after free rollout. Track indices come from separate historical
    matching. -1 means a missed reference target, never a placeholder GT track.
    Denominator remains two worlds; coverage is mandatory in reporting.
    """
    if len(inputs)!=2 or len(encoded)!=2 or reference_paths.shape!=(2,16,6):
        raise ValueError('Two worlds and two 16-frame p/u target trajectories required')
    losses=[]; scores=[]; covered=0
    for world in range(2):
        history=historical_posterior(system,inputs[world],encoded[world])
        k=history_track_indices[world]
        if history['empty'] or k<0:
            losses.append(history['shared'].sum()*0); scores.append(None)
            continue
        if k>=history['initial'].shape[1]:
            raise IndexError('Historical matching index exceeds predicted tracks')
        predicted=persistent_mode_rollouts(system.dynamics,history['tokens'],history['shared'],history['initial'],
                                          history['presence_logits'],generator,covariance=history['covariance'],draws=draws)
        log_density=candidate_log_density(predicted['state'][...,k,:6],predicted['mode_logits'],
                                         reference_paths[None],scales,bandwidth,predicted['log_survival'])
        losses.append(evidence_preference(log_density,torch.tensor([world],device=log_density.device)))
        scores.append(log_density.detach()); covered+=1
    return dict(loss=torch.stack(losses).sum()/2,covered_worlds=covered,total_worlds=2,scores=scores)
