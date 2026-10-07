"""Simple cooperative forecasting baselines using the same frozen observations.

Concat-GRU replaces the future decoder: each tracked target's historical token
is concatenated with the separately pooled aerial and ground scene features.
The historical association/filter remains shared with PeerDream. A second
baseline uses precision fusion with equal evidence weights and constant
velocity. No future labels enter either baseline's prediction interface.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.paper_experiments.agcr_exp.future_prediction import JointDynamics
from experiments.paper_experiments.agcr_exp.temporal_state import StateScale


class ConcatGRU(nn.Module):
    objective = JointDynamics.objective

    def __init__(self, statistics):
        super().__init__()
        self.scale=StateScale(statistics)
        self.concat=nn.Sequential(nn.Linear(256*3+12,256),nn.LayerNorm(256),nn.GELU())
        self.recurrent=nn.GRU(256,256,num_layers=2,batch_first=True)
        self.modes=nn.Parameter(torch.randn(8,256)*.02)
        self.times=nn.Parameter(torch.randn(16,256)*.02)
        self.mixture=nn.Linear(256*3,8)
        self.acceleration=nn.Linear(256,6)
        self.yaw=nn.Linear(256,2)
        self.presence=nn.Linear(256,1)
        self._view_context=None
        self.incremental=True
        self._recurrent_cache=None

    def condition(self, history):
        # History-only pooled views; the shared visibility-weighted scene token
        # is not used by this future decoder. Target tracking is held common.
        separate=history['separate']
        if separate.ndim!=3 or separate.shape[0]!=2 or separate.shape[-1]!=256:
            raise ValueError('Two historical per-view scene features required')
        self._view_context=separate.mean(1).reshape(1,512)
        self._recurrent_cache=None

    def forward(self,tokens,scene,previous,mode_indices=None):
        if self._view_context is None:raise ValueError('Set historical view context before prediction')
        b,t,n,_=previous.shape
        if not n or not 1<=t<=16:raise ValueError('Nonempty track set and original horizon required')
        view=self._view_context.expand(b,-1)
        cached=self._recurrent_cache
        append=(self.incremental and not torch.is_grad_enabled() and cached is not None and cached['shape']==(b,t-1,n)
                and torch.equal(cached['previous'],previous[:,:-1])
                and torch.equal(cached['tokens'],tokens))
        prefix=previous[:,-1:]if append else previous
        length=prefix.shape[1]
        encoded=self.scale.encode(prefix)
        token=tokens[:,None].expand(-1,length,-1,-1)
        views=view[:,None,None].expand(-1,length,n,-1)
        fused=self.concat(torch.cat((token,views,encoded),-1))
        x,hidden=self.recurrent(fused.permute(0,2,1,3).reshape(b*n,length,256),
                                cached['hidden']if append else None)
        x=x.reshape(b,n,length,256).transpose(1,2)
        if append:x=torch.cat((cached['output'],x),1)
        self._recurrent_cache=(dict(shape=(b,t,n),previous=previous,tokens=tokens,output=x,hidden=hidden)
                               if not torch.is_grad_enabled()else None)
        modes=self.modes[None].expand(b,-1,-1) if mode_indices is None else self.modes[mode_indices,None]
        x=x[:,None]+modes[:,:,None,None]+self.times[None,None,:t,None]
        acc,yaw=self.acceleration(x),self.yaw(x)
        return dict(mode_logits=self.mixture(torch.cat((tokens.mean(1),view),-1)),
            acc_mean=acc[...,:3],acc_scale=F.softplus(acc[...,3:])+1e-4,
            yaw_mean=math.pi*torch.tanh(yaw[...,0]),
            yaw_scale=(F.softplus(yaw[...,1])+1e-4).clamp_max(math.pi),
            presence_logits=self.presence(x).squeeze(-1))


def constant_velocity(history,generator,horizon=16,dt=.2):
    """Sample the same initial posterior; retain its existence thereafter."""
    from experiments.paired_interventions.rollout import _initial
    state=_initial(history['initial'],history['covariance'],generator)
    alive=torch.rand(history['presence_logits'].shape,device=state.device,generator=generator)<history['presence_logits'].sigmoid()
    states=[]
    for step in range(1,horizon+1):
        new=state.clone();new[...,:3]=state[...,:3]+dt*step*state[...,3:6]
        states.append(torch.where(alive[...,None],new,state))
    return dict(state=torch.stack(states,1),present=alive[:,None].expand(-1,horizon,-1),
                mode=torch.zeros(len(state),dtype=torch.long,device=state.device))


def equal_precision_update(tracker, original_update, track, evidence, initial=False):
    """Standard covariance-based fusion of associated observations, without v/r.

    Association and detection thresholds stay fixed. Equal scalar weights leave
    the measurement covariance to determine precision in the Kalman update.
    """
    items=[dict(item,weight=item['weight'].new_ones(())) for item in evidence]
    return original_update(track,items,initial=initial)
