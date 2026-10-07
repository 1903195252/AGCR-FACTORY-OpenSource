"""Reference-only state and camera-visibility diagnostics from saved predictions."""
import numpy as np


def ratio(a, b):
    return None if not b else float(a / b)


def reference_metrics(a, ref, focal):
    """Post-inference alignment; masks/labels never construct predictions.

    Report availability together with conditional ADE. Missing reference slots
    remain in denominators; no infinite/sentinel position error is invented.
    Geometry F1 is a separate low-resolution condition diagnostic, never an
    RGB-reader or generated-video score.
    """
    from experiments.formal_evaluation.metrics import aligned_state
    state = a['internal_state']; present = a['state_present'].astype(bool)
    ids = a['historical_reference_ids']; selected = a['historical_selected_slot_mask'].astype(bool)
    eligible = a['history_eligible_reference'].astype(bool)
    truth = ref['world_state'][6:]; truth_present = ref['present'][6:].astype(bool)
    if len(eligible) != truth.shape[1] or not np.array_equal(ref['instance_id'], np.arange(1, len(eligible)+1)):
        raise ValueError('Reference IDs/population changed')
    if not np.allclose(a['camera_to_world'], ref['camera_to_world'][6:], rtol=0, atol=2e-6):
        raise ValueError('Future cameras are not aligned at the six-frame boundary')
    aligned = np.full(truth.shape, np.nan, np.float64)
    available = np.zeros(truth_present.shape, bool)
    # One geometry cell corresponds to 8x8 pixels. geometry_intrinsics adds one
    # left padding column, and the native image has exactly 720/8 = 90 columns.
    P = a['predicted_full_layers'][..., 1:91].sum((-1, -2), dtype=np.float64) * 64
    M = a['predicted_visible_layers'][..., 1:91].sum((-1, -2), dtype=np.float64) * 64
    effective = np.zeros((16, 2, len(eligible)), bool)
    fractions = np.zeros(effective.shape, np.float64)
    projected = np.zeros(effective.shape, np.float64)
    for slot in np.flatnonzero(selected):
        k = int(ids[slot])
        if not 0 <= k < len(eligible) or np.isfinite(aligned[:, k]).any():
            raise ValueError('Duplicate/nonreference selected historical slot')
        aligned[:, k] = state[:, slot]
        available[:, k] = present[:, slot]
        projected[:, :, k] = P[:, :, slot+1]
        fractions[:, :, k] = np.divide(M[:, :, slot+1], P[:, :, slot+1],
            out=np.zeros_like(P[:, :, slot+1]), where=P[:, :, slot+1] > 0)
        effective[:, :, k] = ((fractions[:, :, k] >= .1) & (P[:, :, slot+1] >= 144)
            & present[:, slot, None])
    ref_effective = ((ref['visible_fraction'][6:] >= .1) & (ref['projection_area'][6:] >= 144)
        & truth_present[:, None])
    result = {}
    for name, use in [('all_history_eligible', eligible), ('focal_history_eligible', eligible & (np.arange(len(eligible)) == focal))]:
        pop = truth_present & use[None]
        good = pop & available & np.isfinite(aligned).all(-1)
        vp = np.broadcast_to(pop[:, None], effective.shape)
        counts = {k: int(v.sum()) for k, v in dict(tp=effective & ref_effective & vp,
            fp=effective & ~ref_effective & vp, fn=~effective & ref_effective & vp,
            tn=~effective & ~ref_effective & vp).items()}
        result[name] = dict(state=aligned_state(aligned, available, truth, truth_present, use),
            state_reference_steps=int(pop.sum()), state_available_steps=int(good.sum()),
            state_coverage=ratio(good.sum(), pop.sum()),
            geometry_visibility_counts=counts,
            geometry_visibility_F1=ratio(2*counts['tp'], 2*counts['tp']+counts['fp']+counts['fn']),
            predicted_geometry_support_steps=int(((projected >= 144) & vp).sum()),
            geometry_reference_steps=int(vp.sum()))
    return result, effective, ref_effective


def aggregate_reference(rows, population):
    if any(not r['reference_available'] for r in rows):
        return dict(complete=False, expected_clips=len(rows), available_clips=sum(r['reference_available'] for r in rows))
    rr=[r['metrics'][population] for r in rows]
    totals={k:sum(r[k] for r in rr) for k in ('state_reference_steps','state_available_steps','predicted_geometry_support_steps','geometry_reference_steps')}
    c={k:sum(r['geometry_visibility_counts'][k] for r in rr) for k in ('tp','fp','fn','tn')}
    values=[r['state']['ADE_m'] for r in rr]
    return dict(complete=True,clips=len(rr),**totals,
        state_ADE_clip_mean_m=None if any(v is None for v in values) else float(np.mean(values)),
        state_ADE_defined_clips=sum(v is not None for v in values),
        state_coverage=ratio(totals['state_available_steps'],totals['state_reference_steps']),
        geometry_visibility_counts=c,geometry_visibility_F1=ratio(2*c['tp'],2*c['tp']+c['fp']+c['fn']))
