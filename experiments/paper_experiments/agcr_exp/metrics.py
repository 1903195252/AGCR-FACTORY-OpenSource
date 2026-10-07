"""Reference-defined metric populations and additive sufficient statistics.

No invalid prediction may shrink a reference-defined denominator. Missing state
errors are reported separately; missing depth incurs the sensor-range penalty.
"""
import numpy as np


def ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def depth_metrics(pred_mm, validity_probability, reference_mm, reference_valid):
    p, q, r, v = map(np.asarray, (pred_mm, validity_probability, reference_mm, reference_valid))
    if not (p.shape == q.shape == r.shape == v.shape):
        raise ValueError('Depth tensors must have identical shapes')
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Validity probabilities must be finite in [0,1]')
    v = v.astype(bool)
    if np.any(v & (~np.isfinite(r) | (r < 100) | (r > 60000))):
        raise ValueError('Reference valid depth outside declared sensor range')
    finite = np.isfinite(p) & (p >= 100) & (p <= 60000)
    error = np.where(finite, np.abs(p - r), 59900.)[v]
    # Predicted invalidity does not remove errors from the reference population.
    q = q.clip(1e-7, 1-1e-7)
    bce = -(v*np.log(q) + (~v)*np.log1p(-q))
    n = int(v.sum())
    return dict(depth_MAE_mm=ratio(error.sum(), n),
                depth_RMSE_mm=None if n == 0 else float(np.sqrt(np.square(error).mean())),
                depth_missing_rate=ratio((v & ~finite).sum(), n),
                validity_BCE=float(bce.mean()),
                sums=dict(depth_abs=float(error.sum()), depth_square=float(np.square(error).sum()),
                          depth_count=n, missing_depth=int((v & ~finite).sum()),
                          validity_bce=float(bce.sum()), validity_count=int(v.size)))


def validity_auc(probability, reference):
    """Exact rank AUC with average ranks for ties; undefined for single class."""
    from scipy.stats import rankdata
    p = np.asarray(probability).ravel(); y = np.asarray(reference, bool).ravel()
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError('Invalid AUC population')
    n1, n0 = int(y.sum()), int((~y).sum())
    return None if min(n1,n0) == 0 else float((rankdata(p)[y].sum()-n1*(n1+1)/2)/(n1*n0))


def state_metrics(predicted, reference, present, eligible, predicted_present=None):
    """Aligned track states [time,track,11]; association must be evaluated separately.

    Population: history-eligible AND reference-present. Finite aligned errors
    alone are not a full-cohort score; always retain state_missing_rate.
    """
    p, r = np.asarray(predicted), np.asarray(reference)
    if p.shape != r.shape or p.shape[-1] != 11:
        raise ValueError('Expected aligned [T,N,11] box states')
    population = np.asarray(present, bool) & np.asarray(eligible, bool)[None]
    available = np.isfinite(p).all(-1)
    if predicted_present is not None: available &= np.asarray(predicted_present, bool)
    good = population & available
    fields = {'position_error_m':np.linalg.norm(p[...,:3]-r[...,:3],axis=-1),
              'velocity_error_mps':np.linalg.norm(p[...,3:6]-r[...,3:6],axis=-1),
              'heading_error_rad':np.abs((p[...,6]-r[...,6]+np.pi)%(2*np.pi)-np.pi),
              'size_error_m':np.abs(p[...,8:11]-r[...,8:11]).mean(-1)}
    out = {k:None if not good.any() else float(e[good].mean()) for k,e in fields.items()}
    out.update(state_missing_rate=ratio((population & ~available).sum(), population.sum()),
               reference_count=int(population.sum()), matched_count=int(good.sum()),
               sums={k:dict(total=float(e[good].sum()),count=int(good.sum())) for k,e in fields.items()})
    return out


def binary_counts(prediction, reference, population=None):
    p, r = np.asarray(prediction,bool), np.asarray(reference,bool)
    if p.shape != r.shape: raise ValueError('Binary tensors must match')
    m = np.ones_like(r) if population is None else np.broadcast_to(population,r.shape)
    return dict(tp=int((m&p&r).sum()),fp=int((m&p&~r).sum()),fn=int((m&~p&r).sum()),tn=int((m&~p&~r).sum()))


def binary_scores(counts):
    t,f,n = counts['tp'],counts['fp'],counts['fn']
    return dict(F1=ratio(2*t,2*t+f+n),IoU=ratio(t,t+f+n),precision=ratio(t,t+f),recall=ratio(t,t+n))


def false_appearance(predicted_target_mask, isolated_depth_mm, reference_depth_mm, fully_occluded):
    """Target pixels rendered in front of a reference occluder, never an RGB heuristic.

    predicted_target_mask MUST be supplied by the independent evaluation reader.
    Target support from reference isolated rendering defines the denominator.
    """
    p = np.asarray(predicted_target_mask,bool); z = np.asarray(isolated_depth_mm)
    d = np.asarray(reference_depth_mm)
    if d.ndim == z.ndim-1: d = np.expand_dims(d,-3)
    hidden = np.asarray(fully_occluded,bool)[...,None,None]
    support = (z > 0) & (d > 0) & (d + 20 < z) & hidden
    n = int(support.sum()); wrong = int((p & support).sum())
    events = support.any(axis=(-2,-1))
    return dict(false_appearance_rate=ratio(wrong,n), false_appearance_pixels=wrong,
                occluded_support_pixels=n, event_false_appearance_rate=ratio(((p&support).any((-2,-1))&events).sum(),events.sum()))


def paired_samples(records, expected=8):
    """Exactly eight predeclared indices; primary mean and one LPIPS-selected draw."""
    ids=[r['sample_index'] for r in records]
    if sorted(ids)!=list(range(expected)): raise ValueError('Missing/duplicated paired sample index')
    records=sorted(records,key=lambda r:r['sample_index'])
    keys=set(records[0]['metrics'])
    if any(set(r['metrics'])!=keys for r in records):raise ValueError('Sample metric schema changed')
    primary={}
    for k in keys:
        vals=[r['metrics'][k] for r in records]
        # Missing metric is not silently dropped from the eight-sample average.
        primary[k]=None if any(v is None for v in vals) else float(np.mean(vals))
    lp=[r['metrics'].get('paired_LPIPS') for r in records]
    if any(x is None or not np.isfinite(x) for x in lp):best=None
    else:
        j=int(np.argmin(lp));best=dict(sample_index=j,metrics=records[j]['metrics'])
    return dict(primary=primary,best_of_8=best)


def frechet_distance(reference_features, generated_features):
    """Fréchet feature distance; call FVD ONLY for a verified I3D feature backend.

    Evaluate independently for each view and sample index over the full corpus.
    """
    from scipy.linalg import sqrtm
    x,y=map(lambda a:np.asarray(a,np.float64),(reference_features,generated_features))
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y))<2:
        raise ValueError('At least two video features per population are required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite video features')
    a,b=np.cov(x,rowvar=False),np.cov(y,rowvar=False)
    root=sqrtm(a@b)
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag))>1e-5:raise ValueError('Unstable Fréchet covariance root')
        root=root.real
    return max(0.,float(np.square(x.mean(0)-y.mean(0)).sum()+np.trace(a+b-2*root)))


def probabilistic_state_metrics(mean,covariance,reference,presence_probability,present,population):
    """Gaussian 6-D state diagnostic; not a replacement for joint mixture NLL."""
    from scipy.stats import chi2
    mean,covariance,reference=map(np.asarray,(mean,covariance,reference))
    pop=np.asarray(population,bool);exists=np.asarray(present,bool)
    q=np.asarray(presence_probability,float)
    if mean.shape!=reference.shape or covariance.shape!=mean.shape[:-1]+(mean.shape[-1],mean.shape[-1]):
        raise ValueError('State Gaussian shape mismatch')
    if not np.isfinite(q).all() or np.any((q<0)|(q>1)):raise ValueError('Invalid presence probability')
    good=pop&exists;delta=reference[good]-mean[good];cov=covariance[good]
    if not np.isfinite(cov).all() or not np.isfinite(delta).all():raise ValueError('Nonfinite predicted state distribution')
    # Cholesky both validates positive-definiteness and avoids forming an inverse.
    L=np.linalg.cholesky(cov);z=np.linalg.solve(L,delta[...,None])[...,0]
    maha=np.square(z).sum(-1);dim=mean.shape[-1]
    nll=.5*(maha+2*np.log(np.diagonal(L,axis1=-2,axis2=-1)).sum(-1)+dim*np.log(2*np.pi))
    return dict(gaussian_state_NLL=None if not len(nll) else float(nll.mean()),
                state_coverage_95=ratio((maha<=chi2.ppf(.95,dim)).sum(),len(maha)),
                presence_Brier=None if not pop.any() else float(np.square(q[pop]-exists[pop]).mean()),
                reference_count=int(good.sum()),presence_count=int(pop.sum()))
