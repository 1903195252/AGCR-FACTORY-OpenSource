"""Opt-in history-to-generated-video feedback for a separate joint candidate.

Not called by renderer-only training. Reuses the existing JointSystem.feedback
objective, including 50 pure-noise Euler steps, last-two-step autograd, frozen
differentiable RGB-D decoding/readout and discrete mode/presence score terms.
No future-reference-condition base forward is called here.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch


@contextmanager
def feedback_parameter_scope(model):
    """Caller must build optimizer and backward/step INSIDE this explicit scope.

    Frozen backbone/DINO/VAE flags remain unchanged. The historical/state model
    can learn; the auxiliary training reader is frozen. This is not the frozen
    formal RGB evaluator. Restore every parameter flag even after an exception.
    """
    saved = [(p,p.requires_grad) for p in model.parameters()]
    try:
        model.upstream.requires_grad_(True)
        model.reader.requires_grad_(False)
        groups = model.trainable_groups()
        if not any(group.get('name') == 'upstream' for group in groups):
            raise ValueError('Joint feedback needs an explicit upstream optimizer group')
        reader = {id(p) for p in model.reader.parameters()}
        if any(id(p) in reader for group in groups for p in group['params']):
            raise ValueError('The auxiliary readout must remain frozen')
        yield groups
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


def history_feedback_objective(model, family, generator, update=0, baseline=0.,
                               evidence_weight=.1, camera_consistency=True):
    """Loss-only extension; preserve original flow/depth and replay separately.

    Neither this helper nor JointSystem.feedback calls Base48.infer_condition
    (which is no_grad) or JointSystem.reference_condition (future GT). A real
    joint trainer must retain supervised state replay to prevent state drift.
    Baseline is the previous-update scalar, not this draw's own loss.
    """
    from experiments.paired_interventions.system import JointSystem
    if family.get('metadata',{}).get('split') != 'train':
        raise ValueError('Joint feedback training accepts TRAIN families only')
    if set(family['branches']) != {(0,0),(0,1),(1,0),(1,1)}:
        raise ValueError('Use a complete audited two-world/two-camera family')
    if not any(p.requires_grad for p in model.upstream.parameters()):
        raise ValueError('Frozen upstream cannot support a joint-feedback claim')
    if any(p.requires_grad for p in model.reader.parameters()):
        raise ValueError('Generated-video auxiliary reader must be frozen')
    if not torch.is_grad_enabled():
        raise ValueError('Feedback cannot run under no_grad/inference mode')
    if evidence_weight < 0:
        raise ValueError('Evidence weight must be nonnegative')
    # The existing feedback uses prepared(..., grad=True), so history/filter,
    # continuous rollout, geometry and both camera paths share one graph.
    with patch.dict(model.variant,dict(detach_decoded_feedback_state=False,camera=bool(camera_consistency))):
        value, details = JointSystem.feedback(model,family,generator,(update//8)%2,baseline)
        evidence, coverage = model.evidence(family,generator)
    total = value + evidence_weight*evidence
    return total,dict(details,**coverage,paired_evidence=evidence,
        gradient_route='history -> sampled state -> conditions -> last 2 of 50 pure-noise Euler steps -> decoded RGB-D -> frozen auxiliary reader -> loss',
        continuous_state_pathwise=True,discrete_mode_presence_score_function=True,
        future_truth_conditions=False,formal_evaluation_reader_used=False,
        implementation_scope='prepared objective; no joint training run or native-Wan gradient success claimed')


def feedback_gradient_report(model):
    """Call after backward, before zero_grad; no success inferred from test doubles."""
    groups = {}
    for prefix in ('upstream.observation.','upstream.tracker.','upstream.dynamics.',
                   'upstream.geometry.','condition.','wan.adapters.','depth.'):
        grads = [p.grad.detach().float() for name,p in model.named_parameters()
                 if name.startswith(prefix) and p.grad is not None]
        groups[prefix] = dict(parameters_with_gradient=len(grads),
            norm=float(torch.sqrt(sum((g.square().sum() for g in grads),torch.tensor(0.,device=grads[0].device))).cpu()) if grads else 0.)
    return dict(tensor_hooks=dict(model.feedback_gradient_norms),parameter_groups=groups,
                interpretation='Only native full-model nonzero finite hooks demonstrate the actual route; CPU toy tests verify algebra/control flow only')
