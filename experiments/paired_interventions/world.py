"""History-only shared-world construction and a dense historical-query reader."""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .bridge import historical_posterior
from .rollout import sample_shared_future
from .video_modules import WeightedAppearanceState
from .upstream_runtime import RuntimeStageTwo
from contextlib import nullcontext
from experiments.paper_experiments.agcr_exp.stage2_system import StageTwo,FrozenImageEncoders
from experiments.paper_experiments.agcr_exp.future_prediction import geometry_intrinsics
from experiments.paper_experiments.agcr_exp.geometry import normalize_depth,backproject
from experiments.paper_experiments.agcr_exp.temporal_state import historical_assignment


def load_upstream(checkpoint_path,statistics_path,device):
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    statistics=json.loads(Path(statistics_path).read_text())
    weights=payload['model']
    observation={k[len('observation.'):]:v for k,v in weights.items() if k.startswith('observation.')}
    model=RuntimeStageTwo(statistics,observation,execution_policy='optimized_v1')
    model.tracker=WeightedAppearanceState(statistics,model.observation.association)
    model.load_state_dict(weights,strict=True)
    return model.to(device),payload['step']


def query_labels(history,labels):
    """Used for losses AFTER inference; never supplies tracks to prediction."""
    maps=historical_assignment(history['observations'],labels)
    ids=[int(maps[t][v][q]) for t,v,q in (track['birth'] for track in history['result']['tracks'])]
    return torch.tensor(ids,device=history['shared'].device,dtype=torch.long)


def predict_world(system,inputs,encoded,generator,draw_grad=True):
    history=historical_posterior(system,inputs,encoded)
    if history['empty']:
        like=history['shared']
        draw=dict(state=like.new_empty(1,16,0,11),present=torch.empty(1,16,0,device=like.device,dtype=torch.bool),
                  discrete_log_prob=like.sum((1,2))*0)
    else:
        with nullcontext() if draw_grad else torch.no_grad():
            draw=sample_shared_future(system.dynamics,history['tokens'],history['shared'],history['initial'],
                history['presence_logits'],generator,covariance=history['covariance'])
    return history,draw


def world_conditions(system,history,draw,inputs,detach_state=False):
    """One draw; two camera paths can independently call this function.

    Neither future RGB-D nor reference state/masks/IDs are accepted.
    """
    tracks=history['result']['tracks'];tokens=history['result']['tokens']
    state,present=draw['state'][0],draw['present'][0]
    if detach_state: state=state.detach();tokens=tokens.detach()
    cameras=inputs['planned_camera_to_world'].flatten(0,1)
    K=geometry_intrinsics(inputs['K'])
    with torch.no_grad(): _,projection=system.project_history(inputs,history['observations'],K,cameras)
    repeated=state[:,None].expand(-1,2,-1,-1).flatten(0,1)
    alive=present[:,None].expand(-1,2,-1).flatten(0,1)
    geometry=[]
    for first in range(0,32,2):
        geometry.append(system.geometry(history['shared'],history['separate'],tokens,repeated[first:first+2],
            alive[first:first+2],projection[first:first+2],K,cameras[first:first+2],
            inputs['future_time_s'].repeat_interleave(2)[first:first+2],torch.arange(2,device=K.device)))
    c={name:torch.cat([g[name] for g in geometry]).reshape(16,2,*geometry[0][name].shape[1:]) for name in geometry[0]}
    max_bank=max([sum(len(a) for a in track['appearance']) for track in tracks]+[1])
    banks,weights=[],[]
    for track in tracks:
        a=torch.cat(track['appearance']) if track['appearance'] else tokens.new_empty(0,256)
        w=torch.cat(track['appearance_weights']) if track['appearance_weights'] else tokens.new_empty(0)
        if len(a)!=len(w): raise ValueError('Historical bank weights lost alignment')
        banks.append(F.pad(a,(0,0,0,max_bank-len(a))));weights.append(F.pad(w,(0,max_bank-len(w))))
    identity=torch.stack([track['descriptor'] for track in tracks]) if tracks else tokens.new_empty(0,256)
    uncertainty=torch.stack([track['covariance'].diagonal().clamp_min(0).sqrt() for track in tracks]) if tracks else tokens.new_empty(0,6)
    c.update(state=state,normalized_state=system.tracker.scale.encode(state),present=present,cameras=inputs['planned_camera_to_world'],K=inputs['K'],times=inputs['future_time_s'],
        identity=identity,uncertainty=uncertainty/system.tracker.scale.std[:6],bank=torch.stack(banks) if banks else tokens.new_empty(0,1,256),
        bank_weight=torch.stack(weights) if weights else tokens.new_empty(0,1),
        background=history['shared'],view_background=history['separate'])
    return c


class HistoryQueryReader(nn.Module):
    """Dense K outputs, including invisible and missed generated targets.

    Historical descriptors select queries. Generated RGB-D, calibration and
    time are the only image inputs. Future boxes/masks are supervision only.
    This is separately pretrained on rendered TRAIN images before freezing.
    """
    def __init__(self,observation_head):
        super().__init__()
        self.head=copy.deepcopy(observation_head)
        self.query=nn.Linear(256,256)
        nn.init.eye_(self.query.weight);nn.init.zeros_(self.query.bias)
        self.identity=nn.Linear(256,256)

    def forward(self,dino,depth_mm,validity,K,T,time_s,view,history_descriptors):
        h=self.head;b=dino.shape[0];n=history_descriptors.shape[-2]
        if n==0:
            return dict(center=dino.new_empty(b,0,3),descriptor=dino.new_empty(b,0,256),mask_logits=dino.new_empty(b,0,24,36))
        # Continuous validity probabilities retain decoder gradients.
        log,_=normalize_depth(depth_mm)
        valid=validity.float().clamp(0,1)
        value=F.interpolate(torch.stack([log*valid,valid],1),(336,504),mode='nearest')
        fused=h.fuse(torch.cat([dino,h.depth(value).flatten(2).transpose(1,2)],-1))
        points=backproject(depth_mm,K,T)
        weight=F.avg_pool2d(valid[:,None],20,20)
        world=F.avg_pool2d((points*valid[...,None]).permute(0,3,1,2),20,20)/weight.clamp_min(1e-8)
        world=world.flatten(2).transpose(1,2)
        yy,xx=torch.meshgrid(torch.arange(24,device=dino.device),torch.arange(36,device=dino.device),indexing='ij')
        pixels=torch.stack([(xx+.5)*20-.5,(yy+.5)*20-.5,torch.ones_like(xx)],-1).float().reshape(864,3)
        rays=F.normalize(torch.einsum('bij,bnj->bni',T[:,:3,:3],torch.einsum('bij,nj->bni',torch.linalg.inv(K),pixels)),dim=-1)
        uv=(pixels[:,:2]/pixels.new_tensor([720,480])*2-1)[None].expand(b,-1,-1)
        memory=fused+h.position(torch.cat([world/20,uv,rays,time_s[:,None,None].expand(-1,864,1)],-1))+h.platform(view)[:,None]
        # Queries identify historical targets, but are not an optimization
        # shortcut around the generated RGB-D. Decoded feedback must return
        # through the image/depth inputs, not through changes to this query.
        q=self.query(history_descriptors.detach()).expand(b,-1,-1)
        for layer in h.blocks: q=checkpoint(layer,q,memory,use_reentrant=False) if torch.is_grad_enabled() else layer(q,memory)
        P=torch.einsum('bkc,bnc->bkn',h.full(q),fused).div(16).sigmoid()
        M=P*torch.einsum('bkc,bnc->bkn',h.visible(q),fused).div(16).sigmoid()
        support=M*weight.flatten(2)
        center=torch.einsum('bkn,bnc->bkc',support,world)/support.sum(-1,keepdim=True).clamp_min(1e-8)+h.center(q)
        return dict(center=center,descriptor=self.identity(q),mask_logits=torch.logit(M.clamp(1e-6,1-1e-6)).reshape(b,n,24,36))


def read_generated(reader,dino,rgb,depth,validity,inputs,history_descriptors):
    """RGB [16,2,3,480,720] in [-1,1]; no no_grad around frozen DINO."""
    output=[]
    for t in range(16):
        def features(x):
            x=F.interpolate((x.float()+1)/2,(336,504),mode='bilinear',align_corners=False)
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            return dino.forward_features(x)['x_norm_patchtokens']
        feat=checkpoint(features,rgb[t],use_reentrant=False) if torch.is_grad_enabled() else features(rgb[t])
        output.append(reader(feat,depth[t],validity[t],inputs['K'][None].expand(2,-1,-1),
            inputs['planned_camera_to_world'][t],inputs['future_time_s'][t].expand(2),
            torch.arange(2,device=rgb.device),history_descriptors))
    return {key:torch.stack([r[key] for r in output]) for key in output[0]}
