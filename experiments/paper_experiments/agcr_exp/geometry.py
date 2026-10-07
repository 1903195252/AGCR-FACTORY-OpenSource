"""Calibrated metric geometry. Depth is optical Z in mm, points in meters."""
import torch
from torch.nn import functional as F

def backproject(depth_mm,K,T):
    h,w=depth_mm.shape[-2:];y,x=torch.meshgrid(torch.arange(h,device=depth_mm.device),torch.arange(w,device=depth_mm.device),indexing='ij')
    pixels=torch.stack([x,y,torch.ones_like(x)],-1).to(depth_mm.dtype)
    rays=torch.einsum('...ij,hwj->...hwi',torch.linalg.inv(K),pixels)
    points=rays*depth_mm[...,None]*.001
    return torch.einsum('...ij,...hwj->...hwi',T[...,:3,:3],points)+T[...,:3,3][...,None,None,:]

def normalize_depth(d,valid=None):
    ok=torch.isfinite(d)&(d>=100)&(d<=60000)
    if valid is not None:ok &= valid.bool()
    log=2*(d.clamp(100,60000).log()-torch.log(d.new_tensor(100.)))/torch.log(d.new_tensor(600.))-1
    return torch.where(ok,log,0.),ok

def inverse_depth(log):return torch.exp((log+1)*.5*torch.log(log.new_tensor(600.))+torch.log(log.new_tensor(100.)))

def soft_order(depth,support,mode='soft',tolerance_mm=20.,temperature_mm=50.):
    # [..., layers, height, width]; simultaneous products, not a recursive order.
    if mode=='none':return support
    if mode not in ('soft','hard'):raise ValueError(mode)
    result=[]
    for l in range(depth.shape[-3]):
        terms=[]
        for j in range(depth.shape[-3]):
            if l==j:continue
            delta=depth[...,l,:,:]-depth[...,j,:,:]-tolerance_mm
            nearer=(delta>0).to(depth.dtype) if mode=='hard' else torch.sigmoid(delta/temperature_mm)
            terms.append(1-support[...,j,:,:]*nearer)
        product=torch.stack(terms).prod(0) if terms else torch.ones_like(support[...,l,:,:])
        result.append(support[...,l,:,:]*product)
    return torch.stack(result,-3)

def box_intervals(center,dimensions,yaw,K,T,h=60,w=92,near_mm=100.,far_mm=60000.):
    """One camera, N upright boxes; K already refers to this image grid."""
    y,x=torch.meshgrid(torch.arange(h,device=center.device),torch.arange(w,device=center.device),indexing='ij')
    pixels=torch.stack([x,y,torch.ones_like(x)],-1).to(center.dtype)
    ray=pixels@torch.linalg.inv(K).T@T[:3,:3].T # parameter is optical-axis Z in meters
    c,s=yaw.cos(),yaw.sin();R=torch.zeros((len(center),3,3),device=center.device,dtype=center.dtype)
    R[:,0,0]=c;R[:,0,1]=-s;R[:,1,0]=s;R[:,1,1]=c;R[:,2,2]=1
    origin=torch.einsum('ki,kij->kj',T[:3,3]-center,R)
    direction=torch.einsum('hwi,kij->khwj',ray,R)
    parallel=direction.abs()<1e-9
    safe=torch.where(parallel,torch.ones_like(direction),direction)
    low=(-dimensions[:,None,None]/2-origin[:,None,None])/safe
    high=(dimensions[:,None,None]/2-origin[:,None,None])/safe
    lo=torch.minimum(low,high);hi=torch.maximum(low,high)
    lo=torch.where(parallel,torch.full_like(lo,-torch.inf),lo)
    hi=torch.where(parallel,torch.full_like(hi,torch.inf),hi)
    outside=parallel & (origin[:,None,None].abs()>dimensions[:,None,None]/2)
    entry=lo.amax(-1).clamp_min(near_mm*.001);exit=hi.amin(-1).clamp_max(far_mm*.001)
    valid=(exit>=entry)&~outside.any(-1)
    return torch.where(valid,entry*1000,far_mm),torch.where(valid,exit*1000,far_mm),valid

def correspondences(depth_a,valid_a,depth_g,valid_g,K,T_a,T_g,tolerance_mm=20.,edge_mm=50.):
    """Reference-only correspondences. Both reference depth edges are excluded."""
    P=backproject(depth_a,K,T_a);cam=(P-T_g[:3,3])@T_g[:3,:3]
    uv=cam@K.T;uv=uv[...,:2]/cam[...,2:].clamp_min(1e-8)
    x=uv[...,0].round().long();y=uv[...,1].round().long();h,w=depth_a.shape
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)&(cam[...,2]>0)
    xx=x.clamp(0,w-1);yy=y.clamp(0,h-1)
    def interior(d,v):
        m=v.clone().bool();m[[0,-1],:]=False;m[:,[0,-1]]=False
        for axis in (0,1):
            for sign in (-1,1):m &= (d-d.roll(sign,axis)).abs()<edge_mm;m &= v.roll(sign,axis)
        return m
    ok=inside & interior(depth_a,valid_a)&interior(depth_g,valid_g)[yy,xx]
    ok &= (cam[...,2]*1000-depth_g[yy,xx]).abs()<=tolerance_mm
    ya,xa=torch.where(ok)
    return torch.stack([ya,xa,yy[ok],xx[ok]],-1)
