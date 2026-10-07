"""Reference-defined populations; internal states and visual readouts stay separate."""
import numpy as np
from experiments.paper_experiments.agcr_exp.metrics import depth_metrics,validity_auc,binary_counts,binary_scores,state_metrics
from experiments.paired_interventions.metrics import branch_diagnostics,evidence_effect,reappearance_timing,transition_errors,occlusion_violations


def ratio(n,d):return None if d==0 else float(n/d)

def metric_position(detection):
    """Return a measured/predicted 3-D center, never a floor proxy or null."""
    if detection.get('localization_scope') in ('floor_projection','unavailable'):return None
    position=detection.get('position_world_m')
    if position is None:return None
    value=np.asarray(position,float)
    return value if value.shape==(3,) and np.isfinite(value).all() else None

def box_iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    tl=np.maximum(a[:,None,:2],b[None,:,:2]);br=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(br-tl,0).prod(-1)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(-1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def matched_indices(reference,prediction,threshold=.5):
    from scipy.optimize import linear_sum_assignment
    iou=box_iou([d['box_xyxy'] for d in reference],[d['box_xyxy'] for d in prediction])
    if not len(reference) or not len(prediction):return [],iou
    classes=np.equal.outer([d['class_id'] for d in reference],[d['class_id'] for d in prediction])
    iou*=classes
    # Favor cardinality before IoU; do not let one high IoU suppress two valid matches.
    # A single extra valid match must dominate the complete sum of IoUs,
    # including dense scenes with more than two targets.
    score=(iou>=threshold).astype(float)*(min(iou.shape)+1)+iou
    rr,pp=linear_sum_assignment(-score)
    return [(int(r),int(p)) for r,p in zip(rr,pp) if iou[r,p]>=threshold],iou

def visibility_groups(visible):
    """[T,2,N] reference effective visibility; no predicted-mask stratification."""
    v=np.asarray(visible,bool)
    if v.ndim!=3 or v.shape[1]!=2:raise ValueError('Expected [T,2,N]')
    a,g=v[:,0],v[:,1]
    return {'both_visible':a&g,'aerial_only':a&~g,'ground_only':~a&g,'neither_visible':~a&~g}

def depth_all(pred_mm,probability,ref_mm,ref_valid):
    out=depth_metrics(pred_mm,probability,ref_mm,ref_valid)
    c=binary_counts(np.asarray(probability)>=.5,ref_valid);out['validity_counts']=c
    out.update({'validity_'+k:v for k,v in binary_scores(c).items()})
    out['validity_AUC']=validity_auc(probability,ref_valid)
    return out

def aligned_state(predicted,pred_present,reference,ref_present,eligible,visible=None):
    p=np.asarray(predicted,float);r=np.asarray(reference,float);present=np.asarray(ref_present,bool)
    eligible=np.asarray(eligible,bool);available=np.asarray(pred_present,bool)&np.isfinite(p).all(-1)
    out=state_metrics(p,r,present,eligible,available)
    pop=present&eligible[None];good=pop&available
    final=pop[-1];fg=final&available[-1]
    out.update(ADE_m=out['position_error_m'],FDE_m=ratio(np.linalg.norm(p[-1,:,:3]-r[-1,:,:3],axis=-1)[fg].sum(),fg.sum()),
        FDE_coverage=ratio(fg.sum(),final.sum()),sample_presence_squared_error=ratio(np.square(available.astype(float)-present)[np.broadcast_to(eligible,present.shape)].sum(),np.broadcast_to(eligible,present.shape).sum()),
        false_present_rate=ratio((available&~present&eligible[None]).sum(),((~present)&eligible[None]).sum()))
    if visible is not None:
        out['visibility_strata']={name:state_metrics(p,r,pop&mask,np.ones_like(eligible),available) for name,mask in visibility_groups(visible).items()}
    return out

def decode_rle(rle):
    """JSON COCO RLE from the actual reader's learned mask head."""
    from pycocotools import mask as maskutil
    r=dict(rle)
    if isinstance(r['counts'],str):r['counts']=r['counts'].encode('ascii')
    return maskutil.decode(r).astype(bool)

def load_detection_mask(detection,base,canvas):
    from pathlib import Path
    if 'mask_rle' in detection:mask=decode_rle(detection['mask_rle'])
    elif 'mask_file' in detection:
        path=Path(base)/detection['mask_file']
        with np.load(path,allow_pickle=False) as z:mask=z[detection.get('mask_key','mask')]
        if 'mask_index' in detection:mask=mask[int(detection['mask_index'])]
        if mask.dtype!=bool:mask=mask>=float(detection.get('mask_threshold',.5))
    else:raise ValueError('Independent pixel masks are required; boxes cannot substitute')
    if tuple(mask.shape)!=tuple(canvas):
        # Only a reader-declared low-resolution mask may be resized to the
        # original export grid. Nearest preserves hard pixel membership.
        import torch
        import torch.nn.functional as F
        mask=F.interpolate(torch.as_tensor(mask,dtype=torch.float32)[None,None],size=tuple(canvas),mode='nearest')[0,0].numpy()>0.5
    return mask

def world_points(depth_mm,K,C):
    d=np.asarray(depth_mm,float);h,w=d.shape
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    ray=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    cam=ray*d[...,None]/1000.
    return cam@C[:3,:3].T+C[:3,3]

def reference_correspondence_error(pred_depth,ref_depth,ref_valid,K,camera_to_world,tolerance_mm=20.,max_points_per_frame=None):
    """Fixed reference-visible 3D correspondences, bidirectional, at native pixels.

    Reference depth and geometry select correspondences before predictions are
    read. Invalid predictions receive the full sensor-range error (59.9 m).
    No predicted-validity or predicted-depth agreement filters are allowed.
    """
    pred=np.asarray(pred_depth,float);ref=np.asarray(ref_depth,float);valid=np.asarray(ref_valid,bool)
    if pred.shape!=ref.shape or ref.shape!=valid.shape or ref.ndim!=4 or ref.shape[1]!=2:raise ValueError('Expected [T,2,H,W]')
    if max_points_per_frame is not None:raise ValueError('Formal correspondence denominator uses every reference pixel')
    K=np.asarray(K);C=np.asarray(camera_to_world);total=0.;count=0;missing=0
    h,w=ref.shape[-2:]
    for t in range(len(ref)):
        for a,b in ((0,1),(1,0)):
            ka=K if K.ndim==2 else K[t,a];kb=K if K.ndim==2 else K[t,b]
            world=world_points(ref[t,a],ka,C[t,a]);other=(world-C[t,b,:3,3])@C[t,b,:3,:3]
            uv=other@kb.T;z=uv[...,2];positive=z>0
            x=np.rint(uv[...,0]/np.maximum(z,1e-12)).astype(np.int64);y=np.rint(uv[...,1]/np.maximum(z,1e-12)).astype(np.int64)
            inside=positive&(x>=1)&(x<w-1)&(y>=1)&(y<h-1)&valid[t,a]
            inside[[0,-1],:]=False;inside[:,[0,-1]]=False
            yy,xx=np.where(inside);u=x[yy,xx];v=y[yy,xx]
            good=valid[t,b,v,u]&(np.abs(ref[t,b,v,u]-other[yy,xx,2]*1000)<=tolerance_mm)
            # Exclude reference depth boundaries using fixed four-neighbors.
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                good&=valid[t,b,v+dy,u+dx]&(np.abs(ref[t,b,v+dy,u+dx]-ref[t,b,v,u])<=tolerance_mm)
                good&=valid[t,a,yy+dy,xx+dx]&(np.abs(ref[t,a,yy+dy,xx+dx]-ref[t,a,yy,xx])<=tolerance_mm)
            yy,xx,u,v=yy[good],xx[good],u[good],v[good]
            if not len(u):continue
            pa=world_points(pred[t,a],ka,C[t,a])[yy,xx];pb=world_points(pred[t,b],kb,C[t,b])[v,u]
            okay=np.isfinite(pa).all(-1)&np.isfinite(pb).all(-1)&(pred[t,a,yy,xx]>=100)&(pred[t,a,yy,xx]<=60000)&(pred[t,b,v,u]>=100)&(pred[t,b,v,u]<=60000)
            errors=np.where(okay,np.linalg.norm(pa-pb,axis=-1),59.9)
            total+=float(errors.sum());count+=len(errors);missing+=int((~okay).sum())
    return dict(surface_correspondence_error_m=ratio(total,count),reference_correspondence_count=count,
        surface_error_sum_m=total,invalid_prediction_count=missing,coverage=ratio(count-missing,count))

def exit_assignment(position,regions):
    """Regions are fixed validation-calibrated world AABBs; no nearest-GT future."""
    if position is None:return -1
    p=np.asarray(position,float)
    if p.shape!=(3,) or not np.isfinite(p).all():return -1
    hits=[]
    for j,region in enumerate(regions):
        if np.all(p>=np.asarray(region['min_xyz_m'])) and np.all(p<=np.asarray(region['max_xyz_m'])):hits.append(j)
    return hits[0] if len(hits)==1 else -1

def paired_camera_error(first,second,reference_jointly_visible):
    """World coordinates from independently decoded/observed RGB-D, never S."""
    a,b=np.asarray(first,float),np.asarray(second,float);pop=np.asarray(reference_jointly_visible,bool)
    if a.shape!=b.shape or a.shape[:-1]!=pop.shape or a.shape[-1]!=3:raise ValueError('Camera readout shape mismatch')
    good=pop&np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
    return dict(world_disagreement_m=ratio(np.linalg.norm(a-b,axis=-1)[good].sum(),good.sum()),
        matched_count=int(good.sum()),reference_count=int(pop.sum()),coverage=ratio(good.sum(),pop.sum()))

def reappearance_events(visible,projection_area,history_frames=6,area_min=144):
    """Reference-defined occlusions with a pre-occlusion observable identity."""
    visible=np.asarray(visible,bool);area=np.asarray(projection_area)
    events=[]
    for view in range(visible.shape[1]):
        for target in range(visible.shape[2]):
            seq=visible[:,view,target]
            for h in range(max(1,history_frames),len(seq)-2):
                if seq[h:h+3].all() and not seq[h-1] and seq[:h].any():
                    start=h-1
                    while start>0 and not seq[start-1]:start-=1
                    if np.all(area[start:h,view,target]>=area_min):
                        events.append(dict(view=view,target=target,hidden_start=start,reappearance=h,
                            prior_visible=int(np.where(seq[:start])[0][-1]),hidden_frames=h-start))
    return events


def projected_exit_assignment(point,regions,K,camera_to_world):
    """Image-only endpoint diagnostic from fixed validation 3-D exit regions.

    Camera parameters are known conditions. Overlapping projected regions are
    unresolved, not assigned using the reference future branch or its depth.
    """
    if point is None:return -1
    p=np.asarray(point,float);C=np.asarray(camera_to_world);hits=[]
    for index,region in enumerate(regions):
        lo,hi=np.asarray(region['min_xyz_m']),np.asarray(region['max_xyz_m'])
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        cam=(corners-C[:3,3])@C[:3,:3]
        if (cam[:,2]<=0).any():continue
        pixels=cam@np.asarray(K).T;uv=pixels[:,:2]/pixels[:,2:]
        if np.all(p>=uv.min(0)) and np.all(p<=uv.max(0)):hits.append(index)
    return hits[0] if len(hits)==1 else -1
