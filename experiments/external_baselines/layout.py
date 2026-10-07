"""Shared predicted layouts for layout-conditioned baselines (no future labels).

Keep this comparison separate from raw-history SynCamMaster/ShareVerse and the
fully joint method. A common frozen forecaster supplies the same sampled world
for every layout-conditioned method and both planned camera conditions.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .conditions import strict_inputs


class CommonForecaster:
    def __init__(self,paths,checkpoint,device='cuda',development=False):
        from experiments.paper_experiments.agcr_exp.stage2_system import StageTwo,FrozenImageEncoders
        from experiments.paired_interventions.video_modules import WeightedAppearanceState
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not development and (not payload.get('completed') or payload.get('identity',{}).get('development_only',False)):
            raise ValueError('Common forecaster must be a completed formal training artifact')
        state=payload['model'];prefix='upstream.'
        if any(k.startswith(prefix) for k in state):state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        stats=json.loads(Path(paths['statistics']).read_text())
        obs={k[len('observation.'):]:v for k,v in state.items() if k.startswith('observation.')}
        self.model=StageTwo(stats,obs,execution_policy='optimized_v1')
        self.model.tracker=WeightedAppearanceState(stats,self.model.observation.association)
        self.model.load_state_dict(state,strict=True);self.model.to(device).eval().requires_grad_(False)
        self.encoders=FrozenImageEncoders(paths.get('dino_code',paths.get('dino_source')),paths['dino_weights'],str(Path(paths['wan'])/'vae'),device)
        self.device=device

    @torch.no_grad()
    def predict(self,inputs,seed):
        from experiments.paired_interventions.world import predict_world
        from experiments.paper_experiments.agcr_exp.future_prediction import static_voxels
        strict_inputs(inputs)
        history,draw=predict_world(self.model,inputs,self.encoders.history(inputs['rgb']),torch.Generator(device=self.device).manual_seed(seed))
        tracks=history['result']['tracks']
        classes=torch.stack([t['class_prob'][1:].argmax()+1 for t in tracks]) if tracks else torch.empty(0,device=self.device,dtype=torch.long)
        return dict(initial_state=history['result']['initial_state'],state=draw['state'][0],present=draw['present'][0],class_id=classes,
                    static_voxels=static_voxels(inputs,history['observations'],packed=True))


def box_corners(state):
    """Physical OBB corners from [p,u,yaw,yaw_rate,length,width,height]."""
    signs=state.new_tensor([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
    local=state[...,None,8:11]*signs
    c,s=state[...,6].cos()[...,None],state[...,6].sin()[...,None]
    x,y=local[...,0],local[...,1]
    return torch.stack((c*x-s*y,s*x+c*y,local[...,2]),-1)+state[...,None,:3]


@torch.no_grad()
def historical_map(voxels,poses,K,h,w):
    """Three explicit geometry channels from historically observed static voxels.

    Occupancy, normalized elevation and floor membership replace driving road
    semantics; no hidden factory CAD map or future depth is consulted.
    """
    out=poses.new_zeros(*poses.shape[:2],3,h,w)
    for t in range(len(poses)):
        for v in range(2):
            if not len(voxels):continue
            p=(voxels-poses[t,v,:3,3])@poses[t,v,:3,:3]
            uv=p@K.T;uv=uv[:,:2]/p[:,2:].clamp_min(1e-6)
            x,y=uv.round().long().unbind(-1);valid=(p[:,2]>.1)&(x>=0)&(x<w)&(y>=0)&(y<h)
            ids=torch.where(valid)[0];pixels=y[ids]*w+x[ids]
            nearest=p.new_full((h*w,),float('inf'))
            nearest.scatter_reduce_(0,pixels,p[ids,2],reduce='amin',include_self=True)
            front=p[ids,2]==nearest[pixels]
            owner=torch.full((h*w,),len(voxels),device=p.device,dtype=torch.long)
            owner.scatter_reduce_(0,pixels[front],ids[front],reduce='amin',include_self=True)
            good=owner<len(voxels);flat=out[t,v].reshape(3,-1);chosen=owner[good]
            flat[0,good]=1
            flat[1,good]=(voxels[chosen,2]/12).clamp(0,1)
            flat[2,good]=(voxels[chosen,2]<.2).float()
    return out


@torch.no_grad()
def panacea_layout(state,present,classes,voxels,poses,K,height=480,width=720):
    """19-channel published control format, four factory classes in ten slots.

    Channels 0..9: per-class front depth, 10..12: box boundary/class color,
    13..15: historical static geometry, 16..18: normalized world ray direction.
    """
    from experiments.paper_experiments.agcr_exp.geometry import box_intervals
    if state.shape[:2]!=present.shape or len(classes)!=state.shape[1]:raise ValueError('Bad predicted scene dimensions')
    result=state.new_zeros(len(state),2,19,height,width)
    yy,xx=torch.meshgrid(torch.arange(height,device=state.device),torch.arange(width,device=state.device),indexing='ij')
    rays=torch.stack((xx,yy,torch.ones_like(xx)),-1).float()@torch.linalg.inv(K).T
    colors=state.new_tensor([[.9,.6,.1],[.8,.1,.5],[.1,.8,.5],[.2,.5,.95]])
    for t in range(len(state)):
        for v in range(2):
            for k in range(state.shape[1]):
                if not present[t,k]:continue
                category=int(classes[k])-1
                if not 0<=category<4:raise ValueError('Unknown factory category')
                entry,_,mask=box_intervals(state[t,k:k+1,:3],state[t,k:k+1,8:11],state[t,k:k+1,6],K,poses[t,v],h=height,w=width)
                mask=mask[0];value=torch.where(mask,1-entry[0].clamp(0,60000)/60000,0.)
                result[t,v,category]=torch.maximum(result[t,v,category],value)
                expanded=F.max_pool2d(mask[None,None].float(),3,1,1)[0,0]
                eroded=-F.max_pool2d(-mask[None,None].float(),3,1,1)[0,0]
                edge=expanded-eroded
                result[t,v,10:13]=torch.maximum(result[t,v,10:13],colors[category,:,None,None]*edge)
            result[t,v,16:19]=F.normalize(rays@poses[t,v,:3,:3].T,dim=-1).permute(2,0,1)
    result[:,:,13:16]=historical_map(voxels,poses,K,height,width)
    return result
