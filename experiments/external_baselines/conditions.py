"""Common historical interface. Future RGB, depth and states are never read here."""
import torch
from torch import nn
from torch.nn import functional as F

ADMISSIBLE={'rgb','depth_mm','depth_valid','K','camera_to_world','planned_camera_to_world','time_s','future_time_s'}


def strict_inputs(x):
    if set(x)-{'observation_available'}!=ADMISSIBLE: raise ValueError('Historical interface mismatch: '+str(set(x)^ADMISSIBLE))
    if x['rgb'].shape[:2]!=(6,2) or x['planned_camera_to_world'].shape[:2]!=(16,2):
        raise ValueError('Expected six paired history frames and sixteen planned poses')
    if not torch.all(x['time_s']<=0) or not torch.all(x['future_time_s']>0):
        raise ValueError('Invalid history/future time boundary')
    return x


def relative_cameras(x):
    """Native SynCamMaster pair-relative extrinsics at latent frame centers."""
    poses=torch.cat([x['camera_to_world'][-1:],x['planned_camera_to_world']],0)[::4].float()
    relative=torch.linalg.inv(poses[:,:1])@poses
    return relative[:,:,:3].permute(1,0,2,3).flatten(2)


def retrieve_memory(x):
    """ShareVerse nearest-camera spatial retrieval, candidates history only.

    Four future camera queries select from the partner's six historical frames.
    The same rule is used during training and sampling; returns indices for audit.
    """
    pose=x['camera_to_world'];plan=x['planned_camera_to_world'];keys=[3,7,11,15]
    indices=[];rgb=[];cams=[]
    for view in range(2):
        distances=torch.cdist(plan[keys,view,:3,3].float(),pose[:,1-view,:3,3].float())
        available=x.get('observation_available',torch.ones(6,2,device=pose.device,dtype=torch.bool))[:,1-view].bool()
        distances=distances.masked_fill(~available[None],float('inf'))
        ids=distances.argmin(-1);indices.append(ids if available.any() else torch.full_like(ids,-1))
        rgb.append(x['rgb'][ids,1-view] if available.any() else torch.zeros_like(x['rgb'][ids,1-view]))
        cams.append(pose[ids,1-view] if available.any() else torch.zeros_like(pose[ids,1-view]))
    return torch.stack(rgb,1),torch.stack(cams,1),torch.stack(indices,1)


def raymaps(poses,K,height,width):
    """Official ShareVerse world direction + camera origin (meters), stride 8."""
    y,z=torch.meshgrid(torch.arange(height,device=poses.device),torch.arange(width,device=poses.device),indexing='ij')
    pixels=torch.stack((z,y,torch.ones_like(y)),-1).float()
    directions=pixels@torch.linalg.inv(K.float()).T
    world=torch.einsum('fvij,hwj->fvihw',poses[...,:3,:3].float(),directions)
    world=F.interpolate(world.flatten(0,1),size=(height//8,width//8),mode='bilinear',align_corners=False)
    world=world.reshape(*poses.shape[:2],3,height//8,width//8)
    origin=poses[...,:3,3,None,None].expand_as(world)
    return torch.cat((world,origin),2).permute(1,0,2,3,4).contiguous()


class HistoryAdapter(nn.Module):
    """Width-256 RGB(-D), calibrated camera and time adapter; 648 history tokens."""
    def __init__(self,width,modality='rgbd'):
        super().__init__();self.modality=modality
        self.visual=nn.Sequential(nn.Conv2d(5 if modality=='rgbd' else 3,64,5,2,2),nn.SiLU(),nn.Conv2d(64,256,3,2,1),nn.SiLU())
        self.camera=nn.Linear(26,256);self.query=nn.Linear(width,256)
        self.attn=nn.MultiheadAttention(256,8,batch_first=True)
        self.out=nn.Linear(256,width);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def encode(self,x):
        strict_inputs(x)
        rgb=x['rgb'].permute(0,1,4,2,3).flatten(0,1).float()/127.5-1
        # Pool RGB/depth to a fixed observation grid before convolution. This is
        # the disclosed new adapter, not a replacement for the native backbone.
        if self.modality=='rgbd':
            valid=x['depth_valid'].flatten(0,1).float()[:,None]
            depth=torch.log1p(x['depth_mm'].flatten(0,1).float()[:,None].clamp(0,100000))/12
            rgb=torch.cat([rgb,depth*valid,valid],1)
        rgb=F.adaptive_avg_pool2d(rgb,(48,72))
        tokens=F.adaptive_avg_pool2d(self.visual(rgb),(6,9)).flatten(2).transpose(1,2)
        K=x['K'].reshape(1,9).expand(12,-1).float().clone();K[:,[0,2]]/=x['rgb'].shape[3];K[:,[4,5]]/=x['rgb'].shape[2]
        C=x['camera_to_world'].flatten(0,1).flatten(1).float().clone();C[:,[3,7,11]]/=20
        time=x['time_s'][:,None].expand(-1,2).reshape(12,1).float()
        condition=torch.cat([K,C,time],-1)
        tokens=(tokens+self.camera(condition)[:,None]).reshape(1,648,256)
        available=x.get('observation_available',torch.ones(6,2,device=rgb.device,dtype=torch.bool)).bool()
        padding=(~available).reshape(1,12,1).expand(-1,-1,54).reshape(1,648)
        if padding.all():raise ValueError('No observation in either history')
        return tokens,padding

    def forward(self,query,history):
        tokens,padding=history
        keys=tokens.expand(query.shape[0],-1,-1)
        attended=self.attn(self.query(query.float()),keys.float(),keys.float(),key_padding_mask=padding.expand(query.shape[0],-1),need_weights=False)[0]
        return query+self.out(attended).to(query.dtype)
