"""Trainable mechanism kernels, independent of reference labels and figure code.

These are composable components, not a claim of an integrated video generator.
All lengths/velocities here use meters; image geometry uses millimeters.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EvidenceFusion(nn.Module):
    def __init__(self,width=256,heads=8):
        super().__init__()
        self.association=nn.Sequential(nn.Linear(2*width+3,width),nn.GELU(),nn.Linear(width,1))
        self.reliability=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.concat=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width))
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True)

    def score(self,track,observation,mahalanobis,class_compatibility):
        # [...,tracks,observations,width] with caller-provided pairwise broadcasting.
        cosine=F.cosine_similarity(track,observation,dim=-1)
        x=torch.cat([track,observation,mahalanobis[...,None],class_compatibility[...,None],cosine[...,None]],-1)
        return self.association(x).sigmoid().squeeze(-1)

    def forward(self,query,evidence,association,visibility,available,flags,generator=None):
        # query [B,N,C], evidence [B,N,2,C]; matched observations only.
        reliability=self.reliability(evidence).sigmoid().squeeze(-1)
        w=available.to(evidence.dtype)
        if flags.get('weight_control')=='random':
            w=w*torch.rand(w.shape,device=w.device,dtype=w.dtype,generator=generator)
        elif flags.get('fusion')!='uniform':
            for enabled,value in [('use_association_weight',association),('use_visibility_weight',visibility),('use_reliability_weight',reliability)]:
                if flags.get(enabled,True):w=w*value
        fusion=flags.get('fusion','weighted')
        e=evidence*available[...,None]
        if fusion=='concat':fused=self.concat(e.flatten(-2))
        elif fusion=='attention':
            shape=query.shape;missing=~available.flatten(0,1)
            # Null memory prevents NaNs when neither source supplies evidence.
            memory=e.flatten(0,1);memory=torch.cat([memory,torch.zeros_like(memory[:,:1])],1)
            padding=torch.cat([missing,torch.zeros_like(missing[:,:1])],1)
            fused=self.attention(query.reshape(-1,1,shape[-1]),memory,memory,key_padding_mask=padding,need_weights=False)[0].reshape(shape)
        elif fusion in ('weighted','uniform'):fused=(w[...,None]*evidence).sum(-2)/w.sum(-1,keepdim=True).clamp_min(1e-8)
        else:raise ValueError(fusion)
        if fusion in ('concat','attention'):w=available.to(evidence.dtype) # ordinary fusion has no α/v/r gate
        return fused,w,reliability


def hungarian_association(probability,threshold=.5):
    """At most one observation per track PER VIEW, with unmatched dummy columns."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    a=probability.detach().cpu().numpy() if torch.is_tensor(probability) else np.asarray(probability)
    if a.ndim!=2 or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError('Association must be a finite [tracks,observations] probability matrix')
    n,m=a.shape
    if n==0 or m==0:return []
    cost=np.concatenate([-np.log(a.clip(1e-8,1)),np.full((n,n),-math.log(threshold))],axis=1)
    rows,cols=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(rows,cols) if j<m and a[i,j]>threshold]


class GaussianMemory(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.log_process=nn.Parameter(torch.full((6,),-5.))
        self.memory=nn.GRUCell(width,width)
        self.token=nn.Sequential(nn.Linear(width+6+36+1,width),nn.GELU(),nn.Linear(width,width))

    def forward(self,mean,covariance,memory,age,centers,measurement_cov,weights,features,dt):
        # [...,N,6], [...,N,6,6], observations [...,N,views,3].
        # Caller normalizes physical state/covariance using TRAIN-only scales.
        eye=torch.eye(6,device=mean.device,dtype=mean.dtype)
        Fcv=eye.clone();Fcv[:3,3:]=torch.eye(3,device=mean.device,dtype=mean.dtype)*dt
        prior=mean@Fcv.T
        C=Fcv@covariance@Fcv.T+torch.diag(F.softplus(self.log_process)+1e-4)*dt
        J=torch.linalg.inv(C+eye*1e-6);eta=(J@prior[...,None]).squeeze(-1)
        invR=torch.linalg.inv(measurement_cov+torch.eye(3,device=mean.device,dtype=mean.dtype)*1e-6)
        weighted=weights[...,None,None]*invR
        J=J+F.pad(weighted.sum(-3),(0,3,0,3))
        eta=eta+F.pad((weighted@centers[...,None]).squeeze(-1).sum(-2),(0,3))
        postC=torch.linalg.inv(J);post=(postC@eta[...,None]).squeeze(-1)
        total=weights.sum(-1,keepdim=True);gate=1-torch.exp(-total)
        oldshape=memory.shape
        proposal=self.memory(features.reshape(-1,oldshape[-1]),memory.reshape(-1,oldshape[-1])).reshape(oldshape)
        mem=memory+gate*(proposal-memory)
        newage=torch.where(total>0,torch.zeros_like(age),age+dt)
        token=self.token(torch.cat([post,postC.flatten(-2),newage,mem],-1))
        return dict(mean=post,covariance=postC,memory=mem,age=newage,token=token)


class CrossBlock(nn.Module):
    def __init__(self,width=256,heads=8):
        super().__init__();self.norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.self_attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.cross_attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,x,context,causal=None,padding=None,context_padding=None):
        q=self.norm[0](x);x=x+self.self_attn(q,q,q,attn_mask=causal,key_padding_mask=padding,need_weights=False)[0]
        q=self.norm[1](x);x=x+self.cross_attn(q,context,context,key_padding_mask=context_padding,need_weights=False)[0]
        return x+self.ff(self.norm[2](x))


class SceneDynamics(nn.Module):
    """Variable target count, 8 JOINT scene modes and causal autoregressive rollout."""
    def __init__(self,width=256,heads=8,layers=6,modes=8,horizon=16):
        super().__init__();self.modes=modes;self.horizon=horizon
        self.state=nn.Linear(11,width);self.mode=nn.Parameter(torch.randn(modes,width)*.02)
        self.time=nn.Parameter(torch.randn(horizon,width)*.02)
        self.blocks=nn.ModuleList([CrossBlock(width,heads) for _ in range(layers)])
        self.mixture=nn.Linear(width,modes);self.acc=nn.Linear(width,6)
        self.yaw=nn.Linear(width,2);self.exists=nn.Linear(width,1)

    def forward(self,track_tokens,scene_tokens,previous,track_valid):
        # previous: [B,T,N,11], teacher-forced ONLY during supervised training.
        b,t,n,_=previous.shape
        if t>self.horizon or (~track_valid.any(-1)).any():raise ValueError('Empty or overlength dynamics context')
        context=torch.cat([track_tokens*track_valid[...,None],scene_tokens],1)
        pooled=(track_tokens*track_valid[...,None]).sum(1)/track_valid.sum(1,keepdim=True)
        logits=self.mixture(pooled)
        x=self.state(previous)[:,None]+track_tokens[:,None,None]+self.mode[None,:,None,None]+self.time[None,None,:t,None]
        x=x.reshape(b*self.modes,t*n,-1)
        context_padding=torch.cat([~track_valid,torch.zeros(scene_tokens.shape[:2],device=track_valid.device,dtype=torch.bool)],1)
        context_padding=context_padding[:,None].expand(-1,self.modes,-1).flatten(0,1)
        context=context[:,None].expand(-1,self.modes,-1,-1).flatten(0,1)
        times=torch.arange(t,device=x.device).repeat_interleave(n)
        causal=times[None,:]>times[:,None]
        padding=(~track_valid)[:,None,None].expand(-1,self.modes,t,-1).reshape(b*self.modes,t*n)
        for block in self.blocks:x=block(x,context,causal,padding,context_padding)
        x=x.reshape(b,self.modes,t,n,-1);acc=self.acc(x);yaw=self.yaw(x)
        return dict(mode_logits=logits,acc_mean=acc[...,:3],acc_scale=F.softplus(acc[...,3:])+.01,
                    yaw_mean=math.pi*torch.tanh(yaw[...,0]),yaw_scale=(F.softplus(yaw[...,1])+.01).clamp_max(math.pi),
                    presence_logits=self.exists(x).squeeze(-1))

    def sample(self,track_tokens,scene_tokens,initial_state,initial_presence,track_valid,generator,dt=.2,mode_policy='shared'):
        b,n,_=initial_state.shape;states=[initial_state];present=initial_presence.bool();survival=[];mode=None
        for _ in range(self.horizon):
            out=self(track_tokens,scene_tokens,torch.stack(states,1),track_valid)
            if mode is None:
                if mode_policy=='shared':mode=torch.multinomial(out['mode_logits'].softmax(-1),1,generator=generator).expand(-1,n)
                elif mode_policy=='independent_targets':mode=torch.multinomial(out['mode_logits'].softmax(-1),n,replacement=True,generator=generator)
                else:raise ValueError(mode_policy)
            def choose(key):
                value=out[key][:,:,-1]
                return value[torch.arange(b,device=value.device)[:,None],mode,torch.arange(n,device=value.device)[None]]
            acc=choose('acc_mean')+choose('acc_scale')*torch.randn((b,n,3),device=initial_state.device,dtype=initial_state.dtype,generator=generator)
            yaw=choose('yaw_mean')+choose('yaw_scale')*torch.randn((b,n),device=initial_state.device,dtype=initial_state.dtype,generator=generator)
            present=present & (torch.rand((b,n),device=initial_state.device,generator=generator)<choose('presence_logits').sigmoid()) & track_valid
            old=states[-1];new=old.clone()
            new[...,:3]=old[...,:3]+dt*old[...,3:6]+.5*dt*dt*acc;new[...,3:6]=old[...,3:6]+dt*acc
            new[...,6]=(old[...,6]+yaw+math.pi)%(2*math.pi)-math.pi;new[...,7]=yaw/dt
            states.append(new);survival.append(present)
        return dict(state=torch.stack(states[1:],1),present=torch.stack(survival,1),scene_mode=mode)


def joint_dynamics_nll(output,previous,reference,present,eligible,previous_present,dt=.2,independent_modes=False):
    """Sum evidence over tracks and horizon BEFORE shared-mode marginalization."""
    a=(reference[...,3:6]-previous[...,3:6])/dt
    da=(a[:,None]-output['acc_mean'])/output['acc_scale']
    logp=(-.5*da.square()-output['acc_scale'].log()-.5*math.log(2*math.pi)).sum(-1)
    dy=(reference[...,6]-previous[...,6]+math.pi)%(2*math.pi)-math.pi
    shifts=torch.arange(-4,5,device=dy.device,dtype=dy.dtype)*2*math.pi
    diff=(dy[:,None,...,None]-output['yaw_mean'][...,None]+shifts)/output['yaw_scale'][...,None]
    logp=logp+torch.logsumexp(-.5*diff.square()-output['yaw_scale'][...,None].log()-.5*math.log(2*math.pi),-1)
    mask=present.bool()&eligible[:,None]
    logp=torch.where(mask[:,None],logp,torch.zeros_like(logp))
    target=present[:,None].float().expand_as(output['presence_logits'])
    # Absorbing absence: survival hazard supervised only when previously present.
    hazard_mask=previous_present.bool()&eligible[:,None]
    survival=-F.binary_cross_entropy_with_logits(output['presence_logits'],target,reduction='none')*hazard_mask[:,None]
    by_track=(logp+survival).sum(2)
    logpi=output['mode_logits'].log_softmax(-1)
    if independent_modes:loss=-torch.logsumexp(logpi[...,None]+by_track,1).sum(-1)
    else:loss=-torch.logsumexp(logpi+by_track.sum(-1),1)
    return loss.mean()


class ConditionAdapter(nn.Module):
    """Width 256 adapter. All four residual branches project to the Wan width.

    Caller supplies geometry-aligned token maps and simultaneously computed
    partner tokens; this layer never derives visibility from future RGB.
    """
    def __init__(self,wan_width=3072,width=256,heads=8):
        super().__init__();self.down=nn.Linear(wan_width,width)
        self.cross=nn.ModuleDict({name:nn.MultiheadAttention(width,heads,batch_first=True) for name in ['background','state','geometry','partner']})
        self.appearance_q=nn.Linear(width,width,bias=False);self.appearance_k=nn.Linear(width,width,bias=False)
        self.appearance_v=nn.Linear(width,width,bias=False);self.appearance_out=nn.Linear(width,width,bias=False)
        self.heads=heads;self.width=width
        self.output=nn.Linear(width,wan_width);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,hidden,conditions,appearance_gate,partner=None):
        q=self.down(hidden);result=torch.zeros_like(q)
        for name in ['background','state','geometry','appearance']:
            value=conditions[name]
            if value.shape[1]==0:continue
            if name=='appearance':
                b,n,c=q.shape;k=value.shape[1]
                def heads(x):return x.reshape(b,-1,self.heads,c//self.heads).transpose(1,2)
                score=heads(self.appearance_q(q))@heads(self.appearance_k(value)).transpose(-2,-1)/(c//self.heads)**.5
                # [B, query, bank-token] masks keep each target's own visibility.
                # Four tokens belonging to a target repeat that target's mask.
                weights=score.softmax(-1)*appearance_gate[:,None]
                residual=(weights@heads(self.appearance_v(value))).transpose(1,2).reshape(b,n,c)
                residual=self.appearance_out(residual)
            else:residual=self.cross[name](q,value,value,need_weights=False)[0]
            result=result+residual
        if partner is not None:result=result+self.cross['partner'](q,partner,partner,need_weights=False)[0]
        return hidden+self.output(result)


def install_wan_lora(transformer,rank=16,alpha=16):
    from peft import LoraConfig
    if len(transformer.blocks)!=30:raise ValueError('Expected official 30-block Wan2.2-TI2V-5B')
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.,init_lora_weights=True,
                                      target_modules=['to_q','to_k','to_v','to_out.0']))
    return [p for p in transformer.parameters() if p.requires_grad]
