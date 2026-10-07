"""Reference-defined video supervision and dense readout feedback."""
import torch
from torch.nn import functional as F
from experiments.paper_experiments.agcr_exp.geometry import normalize_depth,correspondences,backproject
from .objectives import masked_mean


def depth_loss(log_depth,validity_logits,reference,valid):
    truth,valid=normalize_depth(reference,valid)
    error=log_depth-truth
    value=masked_mean(error.abs(),valid)
    for axis in (-2,-1):
        d=error.diff(dim=axis)
        a=[slice(None)]*error.ndim;b=list(a);a[axis]=slice(1,None);b[axis]=slice(None,-1)
        value=value+.5*masked_mean(d.abs(),valid[tuple(a)]&valid[tuple(b)])
    return value+.1*F.binary_cross_entropy_with_logits(validity_logits,valid.float())


def reference_for_queries(branch,ids):
    y=branch['supervision'];known=ids>=0;safe=ids.clamp_min(0)
    if len(ids)==0:
        like=y['state']
        return dict(mask=like.new_empty(16,2,0,24,36),visible=torch.empty(16,2,0,device=like.device,dtype=torch.bool),
                    centers=like.new_empty(16,2,0,3),known=known)
    mask=(y['future_instance'][:,:,None]==y['instance_id'][safe][None,None,:,None,None])&known[None,None,:,None,None]
    # Loss targets only; not appearance gates or forecast geometry.
    small=F.avg_pool2d(mask.flatten(0,2)[:,None].float(),20,20).reshape(16,2,len(ids),24,36)
    visible=(y['visible_fraction'][:,:,safe]>=.1)&(y['projection_area'][:,:,safe]>=144)&known
    return dict(mask=small,visible=visible,centers=y['state'][:,safe,:3][:,None].expand(-1,2,-1,-1),known=known)


def readout_loss(readout,reference,descriptors,scales,event=None,use_dice=True):
    known=reference['known'][None,None].expand_as(reference['visible'])
    window=known if event is None else known&event
    visible=reference['visible']&window
    center=masked_mean(((readout['center']-reference['centers'])/scales).abs().mean(-1),visible)
    identity=masked_mean(1-F.cosine_similarity(readout['descriptor'],descriptors[None,None].detach(),dim=-1),visible)
    logits=readout['mask_logits'];target=reference['mask']
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean((-2,-1))
    p=logits.sigmoid();dice=1-(2*(p*target).sum((-2,-1))+1)/(p.sum((-2,-1))+target.sum((-2,-1))+1)
    mask=masked_mean(bce+dice if use_dice else bce,window)
    return center+mask,identity


def reappearance_window(visible):
    """The last hidden interval and first three reappearance frames per view/ID."""
    window=torch.zeros_like(visible,dtype=torch.bool)
    for view in range(2):
        for k in range(visible.shape[-1]):
            start=None
            for t in range(16):
                if not visible[t,view,k] and start is None: start=t
                if visible[t,view,k] and start is not None:
                    window[start:min(16,t+3),view,k]=True;start=None
            if start is not None: window[start:,view,k]=True
    return window


def paired_readout_loss(readout,reference,scales):
    joint=reference['visible'][:,0]&reference['visible'][:,1]
    return masked_mean(((readout['center'][:,0]-readout['center'][:,1])/scales).abs().mean(-1),joint)


def surface_loss(depth,c):
    # Geometry is the supervised depth target of this term, not optimized by
    # moving the reference layers to fit the generated image.
    z,m=c['Z'].detach(),c['M'].detach()
    pred=F.interpolate(F.pad(depth.flatten(0,1)[:,None],(8,8,0,0)),(60,92),mode='area').reshape(16,2,60,92)
    log,_=normalize_depth(pred);ref,_=normalize_depth(z)
    return ((log[:,:,None]-ref).abs()*m).sum()/m.sum().clamp_min(1)


def correspondence_loss(depth,branch,scales):
    x,y=branch['inputs'],branch['supervision'];total=depth.sum()*0;count=0
    for t in range(16):
        with torch.no_grad():
            ij=correspondences(y['depth_mm'][t,0],y['valid'][t,0],y['depth_mm'][t,1],y['valid'][t,1],
                x['K'],x['planned_camera_to_world'][t,0],x['planned_camera_to_world'][t,1])
        if not len(ij): continue
        points=backproject(depth[t],x['K'][None].expand(2,-1,-1),x['planned_camera_to_world'][t])
        delta=(points[0,ij[:,0],ij[:,1]]-points[1,ij[:,2],ij[:,3]])/scales
        total=total+delta.abs().mean(-1).sum();count+=len(ij)
    return total/max(count,1)


def flow_weights(branch):
    support=branch['supervision']['isolated_support'].any(2).float().permute(1,0,2,3)[:,None]
    domain=torch.ones_like(support)
    weight=F.adaptive_avg_pool3d(F.pad(1+2*support,(8,8,0,0,0,0)),(4,30,46))
    coverage=F.adaptive_avg_pool3d(F.pad(domain,(8,8,0,0,0,0)),(4,30,46))
    return weight,coverage
