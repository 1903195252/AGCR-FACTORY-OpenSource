"""The independent reader API. Camera calibration is the only geometry input."""
import numpy as np
import torch
from torch.nn import functional as F
from .model import CooperativeDetector,xyxy
from .data import camera_K,pack_images,move
from .association import CooperativeTracker
from experiments.formal_evaluation import predicted_box_mask

POSTPROCESSING_VERSION='retain-2d-box-mask-and-exclude-floor-proxy-v4'


def reader_implementation():
    from pathlib import Path
    from .data import sha256
    source={name:sha256(Path(__file__).with_name(name))
            for name in ('inference.py','association.py','model.py','data.py')}
    source['formal_evaluation/predicted_box_mask.py']=sha256(Path(predicted_box_mask.__file__))
    return dict(reader_postprocessing=POSTPROCESSING_VERSION,reader_source_sha256=source)


def finite_position(detection):
    value=detection.get('position_world_m')
    return value is not None and np.shape(value)==(3,) and np.isfinite(value).all()


def load_reader(checkpoint,assets,device='cuda:0',required_role=None,allow_development=False):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    identity=saved['identity']
    if required_role and identity['role']!=required_role:raise ValueError('Independent reader role mismatch')
    if not allow_development and (identity['development_only'] or not saved['completed']):
        raise ValueError('Formal inference needs a completed formal independent checkpoint')
    model=CooperativeDetector(assets,identity['modality'],pretrained=False).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    return model,saved


@torch.no_grad()
def predict_pair(model,rgb,depth_mm,valid,K,camera_to_world,confidence=.5,floor_height_m=0.):
    """Returns raw detections (with actual predicted masks); no GT can enter here.

    rgb [2,H,W,3], metric depth/valid [2,H,W], K [3,3] or[2,3,3], poses[2,4,4].
    Uses float masks clipped only to each detection's own predicted box;
    caller stores thresholded masks separately. Empty masks retain detections.
    """
    inputs=dict(rgb=np.asarray(rgb)[None],depth_mm=np.asarray(depth_mm)[None],valid=np.asarray(valid)[None],K=np.asarray(K))
    device=next(model.parameters()).device
    output=model(**move(pack_images(inputs),device));H,W=rgb.shape[1:3];result={}
    for v,view in enumerate(('air','ground')):
        scores,classes=output['logits'][v].sigmoid().max(-1);keep=torch.where(scores>=confidence)[0]
        boxes=xyxy(output['boxes'][v,keep]).clamp(0,1)*torch.tensor([W,H,W,H],device=device)
        masks=F.interpolate(output['mask_logits'][v,keep,None],size=(H,W),mode='bilinear',align_corners=False)[:,0].sigmoid() if len(keep) else torch.empty(0,H,W,device=device)
        masks=predicted_box_mask.clip_torch(masks,boxes)
        detections=[];T=np.asarray(camera_to_world[v]);R=T[:3,:3];origin=T[:3,3];kv=camera_K(np.asarray(K),0,v)
        for j,q in enumerate(keep.tolist()):
            contact=output['contact'][v,q].cpu().numpy()*[W,H]
            d=dict(class_id=int(classes[q])+1,score=float(scores[q]),box_xyxy=boxes[j].cpu().tolist(),
                identity_embedding=output['identity'][v,q].cpu().tolist(),contact_xy=contact.tolist(),mask=masks[j].cpu().numpy())
            if model.modality=='rgbd':
                p=output['center_camera_m'][v,q].cpu().numpy();var=output['center_var_m2'][v,q].cpu().numpy()
                d.update(position_world_m=(R@p+origin).tolist(),center_covariance_world_m2=(R@np.diag(var)@R.T).tolist(),
                         dimensions_m=output['dimensions_m'][v,q].cpu().tolist())
                direction=R@output['heading_camera'][v,q].cpu().numpy()
                d['yaw_rad']=float(np.arctan2(direction[1],direction[0]))
                d['localization_scope']='predicted_center_3d'
            else:
                direction=R@np.linalg.solve(kv,np.r_[contact,1.]);scale=(floor_height_m-origin[2])/direction[2] if abs(direction[2])>1e-6 else -1
                # A flying UAV has no ground contact. A ground-class ray may
                # also miss the floor. Neither invalidates its image detection.
                if d['class_id']!=4 and scale>0 and np.isfinite(scale):
                    d.update(position_world_m=(origin+scale*direction).tolist(),localization_scope='floor_projection',
                        center_covariance_world_m2=(np.eye(3)*.25).tolist())
                else:
                    d.update(position_world_m=None,center_covariance_world_m2=None,localization_scope='unavailable',
                        geometry_missing_reason='airborne_class_has_no_floor_contact' if d['class_id']==4 else 'no_positive_floor_intersection')
            if not finite_position(d):
                d.update(position_world_m=None,center_covariance_world_m2=None,localization_scope='unavailable')
                d.setdefault('geometry_missing_reason','nonfinite_predicted_geometry')
            d['reader_postprocessing']=POSTPROCESSING_VERSION
            detections.append(d)
        result[view]=detections
    return result


class IndependentReader:
    def __init__(self,model,association=None,floor_height_m=0.):
        self.model=model;self.tracker=CooperativeTracker(association);self.floor_height_m=float(floor_height_m)
    def step(self,rgb,depth_mm,valid,K,camera_to_world,time_s):
        detected=predict_pair(self.model,rgb,depth_mm,valid,K,camera_to_world,self.tracker.config['confidence'],self.floor_height_m)
        return self.tracker.update(detected,time_s)
