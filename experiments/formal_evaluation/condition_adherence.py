"""CPU diagnosis of adherence to the generator's own unfiltered soft conditions.

Internal state/geometry define intended conditions, never image measurements or
ground truth. Image boxes come exclusively from the independent reader. No GT
class, box, ID, existence or visibility selects a slot or enters matching.
"""
import argparse,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import read_json,read_jsonl,write_json,sha256,frame_index
from .metrics import box_iou,ratio

VISIBILITY_RATIOS=(.05,.10,.20)
AREA_PX=144


def support_boxes(support):
    """60x90 cropped condition grid -> native 480x720 pixel-edge boxes."""
    boxes=np.zeros((len(support),4),float);nonempty=np.zeros(len(support),bool)
    for k,mask in enumerate(support):
        yy,xx=np.where(mask>0)
        if len(xx):boxes[k]=[8*xx.min(),8*yy.min(),8*(xx.max()+1),8*(yy.max()+1)];nonempty[k]=True
    return boxes,nonempty


def soft_box_iou(masks,boxes):
    """Continuous mask mass against exact box area on the condition grid."""
    masks=np.asarray(masks,float);boxes=np.asarray(boxes,float).reshape(-1,4)
    if not len(boxes):return np.zeros((len(masks),0))
    x=np.arange(90)*8;y=np.arange(60)*8
    width=np.maximum(0,np.minimum(x[None]+8,boxes[:,2,None])-np.maximum(x[None],boxes[:,0,None]))
    height=np.maximum(0,np.minimum(y[None]+8,boxes[:,3,None])-np.maximum(y[None],boxes[:,1,None]))
    occupancy=height[:,:,None]*width[:,None,:]/64
    intersection=np.einsum('khw,dhw->kd',masks,occupancy)*64
    union=masks.sum((1,2))[:,None]*64+occupancy.sum((1,2))[None]*64-intersection
    return np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)


def assignment_values(scores,slot_classes,det_classes,eligible):
    """Every eligible slot is retained, including unassigned zero-score slots."""
    values=np.zeros(len(eligible),float);assignment=np.full(len(eligible),-1,int)
    rows=np.flatnonzero(eligible)
    if not len(rows) or not len(det_classes):return values,assignment
    compatible=np.asarray(slot_classes)[rows,None]==np.asarray(det_classes)[None]
    gain=np.where(compatible,scores[rows],0)
    i,j=linear_sum_assignment(-gain)
    for a,b in zip(i,j):
        if compatible[a,b] and gain[a,b]>0:values[rows[a]]=gain[a,b];assignment[rows[a]]=b
    return values,assignment


def analyze_frame(P,M,Z,envelope,present,state_valid,classes,detections):
    P=P[...,1:-1];M=M[...,1:-1];Z=Z[...,1:-1];envelope=envelope[...,1:-1]
    boxes,full=support_boxes(P);area=P.sum((1,2))*64;visible=M.sum((1,2))*64
    fractions=np.divide(visible,area,out=np.zeros_like(visible),where=area>0)
    detboxes=np.array([d['box_xyxy'] for d in detections],float).reshape(-1,4);detclasses=[d['class_id'] for d in detections]
    full_score=box_iou(boxes,detboxes);visible_score=soft_box_iou(M,detboxes)
    eligible=np.asarray(present)&state_valid&full
    pvals,passign=assignment_values(full_score,classes,detclasses,eligible)
    mvals,massign=assignment_values(visible_score,classes,detclasses,eligible&(visible>0))
    overlap=box_iou(boxes,boxes);same=np.asarray(classes)[:,None]==np.asarray(classes)[None]
    conflict=np.triu((overlap>=.5)&same&eligible[:,None]&eligible[None],1)
    support=P>0;finite=np.isfinite(Z)
    counts=dict(all_slot_views=len(classes),present_slot_views=int(np.sum(present)),invalid_state_slot_views=int((present&~state_valid).sum()),
        present_no_full_support_slot_views=int((present&state_valid&~full).sum()),supported_slot_views=int(eligible.sum()),
        supported_zero_visible_mass_slot_views=int((eligible&(visible<=0)).sum()),independent_detections=len(detections),
        supported_slots_exceeding_detection_count=max(int(eligible.sum())-len(detections),0),
        same_class_projected_box_overlap_pairs=int(conflict.sum()),
        projected_P_mass=float(P.sum()),projected_P_mass_outside_own_state_box=float((P*~envelope).sum()),
        supported_depth_pixels=int(support.sum()),supported_nonfinite_depth_pixels=int((support&~finite).sum()),
        supported_depth_outside_100_60000mm_pixels=int((support&finite&((Z<100)|(Z>60000))).sum()),
        supported_depth_outside_range_beyond_1mm_pixels=int((support&finite&((Z<99)|(Z>60001))).sum()),
        full_box_IoU50_hits_all_supported=int(((pvals>=.5)&eligible).sum()),
        full_box_IoU_sum_all_supported=float(pvals[eligible].sum()),visible_soft_IoU_sum_all_supported=float(mvals[eligible].sum()))
    sensitivity={}
    for cutoff in VISIBILITY_RATIOS:
        selected=eligible&(area>=AREA_PX)&(fractions>=cutoff)
        p,_=assignment_values(full_score,classes,detclasses,selected);m,_=assignment_values(visible_score,classes,detclasses,selected)
        sensitivity[str(cutoff)]=dict(condition_visible_slot_views=int(selected.sum()),full_box_IoU50_hits=int(((p>=.5)&selected).sum()),
            full_box_IoU_sum=float(p[selected].sum()),visible_soft_IoU_sum=float(m[selected].sum()))
    slots=[dict(slot=k,class_id=int(classes[k]),present=bool(present[k]),state_valid=bool(state_valid[k]),
        full_support_nonempty=bool(full[k]),full_soft_area_px=float(area[k]),visible_soft_area_px=float(visible[k]),
        visible_to_full_mass_ratio=float(fractions[k]),full_support_box=boxes[k].tolist(),
        matched_full_box_IoU=float(pvals[k]),matched_visible_soft_IoU=float(mvals[k]),
        full_box_detection_index=int(passign[k]),visible_support_detection_index=int(massign[k])) for k in range(len(classes))]
    return dict(counts=counts,sensitivity=sensitivity,slots=slots)


def aggregate(rows):
    counts=Counter();sensitivity={str(x):Counter() for x in VISIBILITY_RATIOS}
    for row in rows:
        counts.update(row['counts'])
        for key in sensitivity:sensitivity[key].update(row['sensitivity'][key])
    result=dict(counts=dict(counts),full_box_IoU50_fraction_all_supported=ratio(counts['full_box_IoU50_hits_all_supported'],counts['supported_slot_views']),
        mean_full_box_IoU_all_supported=ratio(counts['full_box_IoU_sum_all_supported'],counts['supported_slot_views']),
        mean_visible_soft_IoU_all_supported=ratio(counts['visible_soft_IoU_sum_all_supported'],counts['supported_slot_views']),sensitivity={})
    for key,c in sensitivity.items():
        result['sensitivity'][key]=dict(counts=dict(c),full_box_IoU50_fraction=ratio(c['full_box_IoU50_hits'],c['condition_visible_slot_views']),
            mean_full_box_IoU=ratio(c['full_box_IoU_sum'],c['condition_visible_slot_views']),mean_visible_soft_IoU=ratio(c['visible_soft_IoU_sum'],c['condition_visible_slot_views']))
    return result


def example_figure(path,native_rgb,generated,reader,family):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,4,figsize=(16,6.6));t=15
    for v,view in enumerate(('air','ground')):
        p=generated['predicted_full_layers'][t,v,1:,:,1:-1];m=generated['predicted_visible_layers'][t,v,1:,:,1:-1]
        axes[v,0].imshow(native_rgb[6+t,v]);axes[v,1].imshow(generated['rgb'][t,v])
        image=np.zeros((60,90,3));image[...,1]=p.max(0) if len(p) else 0;image[...,0]=m.max(0) if len(m) else 0
        axes[v,2].imshow(np.repeat(np.repeat(image,8,0),8,1));axes[v,3].imshow(generated['rgb'][t,v])
        boxes,valid=support_boxes(p)
        for k in np.flatnonzero(valid):
            b=boxes[k];axes[v,3].add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#31e4b5',linewidth=.9))
            axes[v,3].text(b[0],b[1],f's{k}',color='#31e4b5',fontsize=6,backgroundcolor='#172020')
        for d in reader[view]:
            b=d['box_xyxy'];axes[v,3].add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#f568cf',linewidth=1,linestyle='--'))
            axes[v,3].text(b[0],b[3],f'd{d["track_id"]}',color='#f568cf',fontsize=6,backgroundcolor='#251725')
        for j,title in enumerate(('Reference pixels (display only)','Unchanged generated pixels','Actual P: green; M: red/yellow','Own P boxes: green; reader boxes: magenta')):
            axes[v,j].set_title((view+' | ' if j==0 else '')+title,fontsize=9);axes[v,j].axis('off')
    fig.suptitle(f'{family["family_id"]} | world 0, camera 0, draw 0, future frame 16\nAll slots retained; model conditions are not visibility ground truth',fontsize=10)
    fig.tight_layout(rect=(0,0,.995,.92));fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','predictions','readouts','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Use a new versioned output directory')
    out.mkdir(parents=True)
    root=Path(a.predictions);readouts=Path(a.readouts);manifest=read_json(readouts/'manifest.json');index=read_json(a.index)
    if not read_json(readouts/'completion.json').get('completed') or manifest['generated_draw']!=0:raise ValueError('Require complete independent draw-zero readouts')
    if manifest['index_sha256']!=sha256(a.index) or manifest['generated_manifest_sha256']!=sha256(root/'manifest.json'):raise ValueError('Changed conditions or corpus')
    families={f['family_id']:f for f in index['families']};selected=[families[f] for f in manifest['family_ids']]
    examples=[]
    for cohort,receiver in (('evidence',0),('evidence',1),('ambiguous',0)):
        choices=[f for f in selected if f['cohort']==cohort and f['receiver']==receiver]
        if choices:examples.append(min(choices,key=lambda f:hashlib.sha256(('condition-adherence-example-v1:'+f['family_id']).encode()).hexdigest())['family_id'])
    import torch
    from experiments.paper_experiments.agcr_exp.geometry import box_intervals
    from experiments.paper_experiments.agcr_exp.future_prediction import geometry_intrinsics
    torch.set_num_threads(4);cells=[];figures=[]
    keys=('internal_state','state_present','class_id','K','camera_to_world','time_s','predicted_full_layers','predicted_visible_layers','predicted_surface_depth_mm')
    for family in selected:
        for branch in family['records']:
            w,c=branch['world'],branch['camera'];ep='%s_w%d_c%d'%(family['family_id'],w,c)
            path=root/family['family_id']/('world%d_camera%d_sample00.npz'%(w,c));cached=read_json(readouts/'cells'/ep/'metrics.json')
            if sha256(path)!=cached['generated_source_sha256']:raise ValueError('Changed generated condition/pixel sample')
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in keys}
            state=arrays['internal_state'];present=arrays['state_present'];classes=arrays['class_id'];N=len(classes)
            if state.shape!=(16,N,11) or present.shape!=(16,N) or arrays['predicted_full_layers'].shape!=(16,2,N+1,60,92):raise ValueError('Slot/layer ordering mismatch')
            if not np.isfinite(arrays['predicted_full_layers']).all() or not np.isfinite(arrays['predicted_visible_layers']).all():raise ValueError('Nonfinite actual support')
            good=np.isfinite(state).all(-1)&(state[...,8:11]>0).all(-1)
            safe=np.where(good[...,None],state,np.zeros_like(state));s=torch.from_numpy(safe[:,None].repeat(2,1).reshape(32,N,11))
            K=geometry_intrinsics(torch.from_numpy(arrays['K']));C=torch.from_numpy(arrays['camera_to_world'].reshape(32,4,4))
            with torch.no_grad():
                envelope=torch.stack([box_intervals(s[j,:,:3],s[j,:,8:11],s[j,:,6],K,C[j])[2] for j in range(32)]).numpy().reshape(16,2,N,60,92)
            predpath=readouts/'cells'/ep/'generated_prediction.jsonl'
            if sha256(predpath)!=cached['generated_prediction_sha256']:raise ValueError('Changed independent image boxes')
            predictions=read_jsonl(predpath);by={(r['frame'],r['view']):r['detections'] for r in predictions};rows=[]
            for t in range(16):
                for v,view in enumerate(('air','ground')):
                    frame=analyze_frame(arrays['predicted_full_layers'][t,v,1:],arrays['predicted_visible_layers'][t,v,1:],
                        arrays['predicted_surface_depth_mm'][t,v,1:],envelope[t,v],present[t],good[t],classes,by[t+6,view])
                    rows.append(dict(frame=t+6,view=view,**frame))
            cell=dict(family_id=family['family_id'],world=w,camera=c,cohort=family['cohort'],slots=N,source_sha256=sha256(path),
                independent_readout_sha256=sha256(predpath),static_depth_nonfinite_pixels=int((~np.isfinite(arrays['predicted_surface_depth_mm'][:,:,0])).sum()),frames=rows)
            cells.append(cell)
            if w==0 and c==0 and family['family_id'] in examples:
                native=Path(index['root'])/branch['path']
                if sha256(native)!=branch['sha256']:raise ValueError('Changed display reference')
                with np.load(native,allow_pickle=False) as z:native_rgb=z['rgb']
                with np.load(path,allow_pickle=False) as z:arrays['rgb']=z['rgb']
                figure=out/(family['family_id']+'.png');example_figure(figure,native_rgb,arrays,{v:by[21,v] for v in ('air','ground')},family)
                figures.append(dict(path=figure.name,sha256=sha256(figure),family_id=family['family_id'],selection='fixed SHA first per registered cohort/receiver stratum; world0 camera0 draw0 frame16'))
    cohorts={name:aggregate([f for cell in cells if name=='all' or cell['cohort']==name for f in cell['frames']]) for name in ('all','evidence','ambiguous')}
    write_json(out/'report.json',dict(schema='agcr-self-condition-adherence-diagnostic-1',diagnostic_only=True,complete=True,
        new_image_inference=False,internal_state_is_not_image_measurement=True,ground_truth_used_for_population_or_matching=False,
        all_slots_retained=True,ground_truth_history_assignment_masks_ignored=True,draw=0,not_mean_of_eight=True,
        condition_visibility_not_ground_truth=True,geometry_grid=dict(height=60,width=92,pad_columns=1,native_scale=8,native_shape=[480,720]),
        sensitivity_protocol=dict(soft_full_area_min_px=AREA_PX,visible_to_full_soft_mass_ratios=VISIBILITY_RATIOS,
            numeric_cutoffs_from_reference_protocol_are_not_the_same_estimand=True,test_threshold_selection_performed=False),
        interpretation='Full-P-box scores include occluded slots and are not detection recall. M is learned soft intended visibility, not physical visibility truth. Visibility-stratified scores measure adherence to own conditions only; invalid/extra/overlapping slots remain reported. Box and soft-support assignments are separate class-compatible one-to-one matchings.',
        source_sha256=sha256(__file__),independent_manifest_sha256=sha256(readouts/'manifest.json'),generator_manifest_sha256=sha256(root/'manifest.json'),
        cohorts=cohorts,cells=cells,figures=figures))


if __name__=='__main__':main()
