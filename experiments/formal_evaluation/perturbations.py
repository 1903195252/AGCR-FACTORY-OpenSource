"""Predeclared observation degradations; future references are not accepted."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch

ALLOWED_HISTORY_KEYS={'rgb','depth_mm','depth_valid','K','camera_to_world','time_s','planned_camera_to_world','future_time_s','observation_available'}


def degrade_history(inputs,config,seed,partner=0,alternate_history=None):
    """Return a fresh input dictionary with the explicitly selected degradation.

    Missing observations are explicitly marked in observation_available[T,2].
    A frontend MUST honor that mask (not interpret black pixels as observations).
    Delay removes newer frames rather than shifting them onto false timestamps.
    Calibration bias changes both historical and planned camera transforms.
    All random factors use a recorded seed, shared across paired comparisons.
    """
    if set(inputs)-ALLOWED_HISTORY_KEYS:raise ValueError('Supervision/unknown keys must not enter history degradations')
    expected=config.get('implementation_sha256')
    if expected and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=expected:
        raise ValueError('Perturbation implementation differs from the registered configuration')
    kind=config.get('kind','clean');out={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in inputs.items()}
    T,V=out['rgb'].shape[:2];rng=np.random.default_rng(seed)
    if V!=2 or partner not in (0,1):raise ValueError('Two-platform history required')
    if 'observation_available' not in out:out['observation_available']=torch.ones(T,V,dtype=torch.bool,device=out['rgb'].device)
    if kind=='clean':return out
    if kind in ('partner_remove','partner_delay'):
        latest=out['time_s'][-1];threshold=latest-float(config.get('seconds',0.))
        mask=torch.ones(T,dtype=torch.bool,device=latest.device) if kind=='partner_remove' else out['time_s']>threshold+1e-6
        out['observation_available'][mask,partner]=False
        out['rgb'][mask,partner]=0;out['depth_mm'][mask,partner]=0;out['depth_valid'][mask,partner]=False
    elif kind=='partner_shuffle':
        permutation=torch.as_tensor(rng.permutation(T),device=out['rgb'].device)
        # Intentionally mismatched content/time negative control; retain the
        # original declared timestamps and poses and record the permutation.
        for key in ('rgb','depth_mm','depth_valid'):out[key][:,partner]=out[key][permutation,partner]
    elif kind=='partner_time_misalignment':
        # Unknown positive latency: older pixels are incorrectly attached to
        # the current declared pose/time. Unlike partner_delay, this does not
        # expose true old timestamps or discard just the unavailable tail.
        frames=int(config.get('frames',1))
        if frames not in (1,3,5):raise ValueError('Unregistered timing offset')
        source=(torch.arange(T,device=out['rgb'].device)-frames).clamp_min(0)
        for key in ('rgb','depth_mm','depth_valid'):out[key][:,partner]=out[key][source,partner]
    elif kind=='partner_mismatch':
        if alternate_history is None:raise ValueError('Requires registered paired alternate HISTORY, never future data')
        if set(alternate_history)-ALLOWED_HISTORY_KEYS:raise ValueError('Leaking alternate history keys')
        for key in ('time_s','camera_to_world','K'):
            if not torch.equal(inputs[key],alternate_history[key]):raise ValueError('Mismatch control requires identical camera path/time')
        for key in ('rgb','depth_mm','depth_valid'):out[key][:,partner]=alternate_history[key][:,partner]
    elif kind=='depth_noise':
        beta=float(config['beta'])
        if beta not in (0.,.005,.02):raise ValueError('Unregistered depth noise level')
        noise=torch.as_tensor(rng.normal(size=tuple(out['depth_mm'].shape)),device=out['depth_mm'].device,dtype=out['depth_mm'].dtype)
        value=out['depth_mm']*(1+beta*noise);valid=out['depth_valid']
        out['depth_mm']=torch.where(valid,value.clamp(100,60000),out['depth_mm'])
    elif kind=='depth_missing':
        fraction=float(config['fraction'])
        if fraction not in (0.,.1,.3,.5):raise ValueError('Unregistered missing-depth fraction')
        # A permuted sequence of rectangles followed by a final rectangular
        # pixel strip reaches the exact requested count of initially valid
        # pixels; the denominator never includes initially invalid pixels.
        for t in range(T):
            for v in range(V):
                valid=out['depth_valid'][t,v].cpu().numpy().copy();need=int(round(valid.sum()*fraction));removed=np.zeros_like(valid)
                H,W=valid.shape;attempt=0
                while need and attempt<10000:
                    h=int(rng.integers(1,max(2,H//4)));w=int(rng.integers(1,max(2,W//4)));y=int(rng.integers(0,H-h+1));x=int(rng.integers(0,W-w+1));attempt+=1
                    yy,xx=np.where(valid[y:y+h,x:x+w]&~removed[y:y+h,x:x+w]);yy+=y;xx+=x
                    take=min(need,len(yy));removed[yy[:take],xx[:take]]=True;need-=take
                if need:raise RuntimeError('Could not construct requested depth mask')
                m=torch.as_tensor(removed,device=out['depth_valid'].device);out['depth_valid'][t,v][m]=False;out['depth_mm'][t,v][m]=0
    elif kind=='depth_range':
        maximum=float(config.get('maximum_mm',15000))
        if maximum!=15000:raise ValueError('Unregistered depth range')
        out['depth_valid'] &= out['depth_mm']<=maximum
        out['depth_mm']=torch.where(out['depth_valid'],out['depth_mm'],torch.zeros_like(out['depth_mm']))
    elif kind=='extrinsic_calibration':
        translation=float(config.get('translation_std_m',.02));rotation=float(config.get('rotation_std_deg',1.))
        if translation not in (0.,.02,.05) or rotation not in (0.,1.,2.):raise ValueError('Unregistered calibration error')
        like=out['camera_to_world']
        for view in range(V):
            axis=rng.normal(size=3)*np.deg2rad(rotation);theta=float(np.linalg.norm(axis))
            delta=torch.eye(4,device=like.device,dtype=like.dtype)
            if theta>0:
                x,y,z=axis/theta
                skew=like.new_tensor([[0,-z,y],[z,0,-x],[-y,x,0]])
                delta[:3,:3]+=np.sin(theta)*skew+(1-np.cos(theta))*(skew@skew)
            delta[:3,3]=like.new_tensor(rng.normal(size=3)*translation)
            # One fixed calibration bias per camera, shared over its history
            # and known planned path. Images and reference geometry stay clean.
            for key in ('camera_to_world','planned_camera_to_world'):
                out[key][:,view]=out[key][:,view]@delta
    elif kind=='rgb_only':out['depth_mm'].zero_();out['depth_valid'].zero_()
    else:raise ValueError('Unknown degradation '+kind)
    return out
