"""Event metrics with reference-defined denominators and eight-draw reporting."""
import numpy as np


def branch_diagnostics(assignments, reference_prior):
    """Assignments [families,8] in {0,1,-1}; -1 is missing/other.

    Do not renormalize over detected targets. Expected Brier includes a
    missing/other category whose reference probability is zero. For a known
    prior q it equals ||p-q||^2 + 1-||q||^2 (not just squared calibration error).
    Pooling across families is
    needed to assess calibration: eight samples need not split evenly.
    """
    assignments = np.asarray(assignments)
    prior = np.asarray(reference_prior,dtype=float)
    if assignments.ndim != 2 or assignments.shape[1] != 8 or not np.isin(assignments,[-1,0,1]).all():
        raise ValueError('Eight fixed-seed draws in {0,1,-1} required')
    if prior.shape != (len(assignments),2) or (prior < 0).any() or not np.allclose(prior.sum(1),1):
        raise ValueError('Known two-branch prior required')
    probability = np.stack([(assignments == k).mean(1) for k in (0,1,-1)],-1)
    target = np.concatenate([prior,np.zeros((len(prior),1))],-1)
    probability_error=((probability-target)**2).sum(1)
    return dict(probability=probability, brier=probability_error+1-(target**2).sum(1),
                probability_error=probability_error,
                both_branch_coverage=((probability[:,:2]>0)|(prior==0)).all(1),
                missing_or_other=probability[:,2],
                pooled_calibration_gap=(probability.mean(0)-target.mean(0)))


def evidence_effect(world0_assignments, world1_assignments):
    zeros = np.zeros((len(world0_assignments),2)); zeros[:,0]=1
    ones = zeros[:,::-1].copy()
    a = branch_diagnostics(world0_assignments,zeros)['probability']
    b = branch_diagnostics(world1_assignments,ones)['probability']
    return dict(correct_mass=(a[:,0]+b[:,1])/2,
                aligned_effect=((a[:,0]-b[:,0])+(b[:,1]-a[:,1]))/2,
                paired_majority_correct=(a[:,0]>.5)&(b[:,1]>.5),
                missing_or_other=(a[:,2]+b[:,2])/2)


def first_visible(bits, start=0, consecutive=3):
    bits = np.asarray(bits,dtype=bool)
    for index in range(start,len(bits)-consecutive+1):
        if bits[index:index+consecutive].all():
            return index
    return None


def reappearance_timing(reference, predicted, start=0, dt=.2):
    """Reappearance after an annotated hidden interval; missing penalty = F*dt.

    Caller supplies start at the hidden event boundary, avoiding an earlier
    historical/visible run being mistaken for the reappearance.
    """
    if len(reference) != len(predicted):
        raise ValueError('Reference and prediction horizons differ')
    ref = first_visible(reference,start)
    if ref is None:
        raise ValueError('No reference reappearance event with three visible frames')
    pred = first_visible(predicted,start)
    missing = pred is None
    signed = None if missing else (pred-ref)*dt
    return dict(signed_seconds=signed, absolute_seconds=len(reference)*dt if missing else abs(signed),
                missing=missing, early=bool(not missing and pred<ref),
                early_duration_seconds=0 if missing else max(0,(ref-pred)*dt))


def occlusion_violations(predicted_mask, full_projection, reference_visible_mask):
    """Area-weighted false appearance on fully occluded, in-frame targets."""
    prediction = np.asarray(predicted_mask,dtype=bool)
    projection = np.asarray(full_projection,dtype=bool)
    visible = np.asarray(reference_visible_mask,dtype=bool)
    if not (prediction.shape == projection.shape == visible.shape):
        raise ValueError('Mask grids must match')
    eligible = (~visible.any((-1,-2))) & projection.any((-1,-2))
    selected = projection & eligible[...,None,None]
    denom = int(selected.sum())
    return dict(false_pixel_fraction=None if not denom else float((prediction&selected).sum()/denom),
                reference_projection_pixels=denom, eligible_targets_frames=int(eligible.sum()))


def transition_errors(reference_visible, predicted_visible, event_frame, kind, window=4):
    """Reference-defined four-frame neighborhood of one visibility transition.

    Early appearance counts visible predictions on hidden frames immediately
    BEFORE reference reappearance. Late disappearance counts them immediately
    AFTER reference disappearance. Denominators include missed predictions.
    """
    reference=np.asarray(reference_visible,dtype=bool)
    predicted=np.asarray(predicted_visible,dtype=bool)
    if reference.shape!=predicted.shape or reference.ndim!=1 or not 0<=event_frame<len(reference) or window<1:
        raise ValueError('Invalid transition sequence, frame, or window')
    if kind=='reappearance':
        if not reference[event_frame]:
            raise ValueError('Reference reappearance must be visible')
        indices=np.arange(max(0,event_frame-window),event_frame)
    elif kind=='disappearance':
        if reference[event_frame]:
            raise ValueError('Reference disappearance must be hidden')
        indices=np.arange(event_frame,min(len(reference),event_frame+window))
    else:
        raise ValueError('Unknown transition type')
    hidden=indices[~reference[indices]]
    return dict(false_visible_frames=int(predicted[hidden].sum()),reference_hidden_frames=len(hidden))
