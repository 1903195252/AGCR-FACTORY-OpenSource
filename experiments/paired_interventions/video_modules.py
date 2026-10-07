"""Native Wan RGB-D generation and visibility-ordered paired adapters.

The official transformer/causal VAE are retained. Conditions are tensors
constructed from the historical posterior and a single sampled world state.
No labels or simulator IDs are accepted by these modules.
"""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .objectives import OrderedAppearanceGate
from experiments.paper_experiments.agcr_exp.geometry import inverse_depth
from experiments.paper_experiments.agcr_exp.temporal_state import TemporalState


class WeightedAppearanceState(TemporalState):
    """Preserve source evidence weights, without changing existing parameters."""
    def update(self,track,evidence,initial=False):
        track=dict(track,appearance=list(track.get('appearance',[])),
                   appearance_weights=list(track.get('appearance_weights',[])))
        out=super().update(track,evidence,initial)
        out['appearance_weights'].extend([item['weight'].expand(4) for item in evidence])
        return out


def resize_video(x,size):
    return F.interpolate(x,size=size,mode='trilinear',align_corners=True)


class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.state=nn.Sequential(nn.Linear(12+16+1+1+6,256),nn.GELU(),nn.Linear(256,256))
        self.state_group=nn.Linear(4*256,256)
        self.geometry=nn.Linear(4*8,256)
        self.kind=nn.Embedding(2,256)
        self.depth_geometry=nn.Conv3d(4,32,1)
        self.platform=nn.Embedding(2,256)

    def forward(self,c):
        # states [16,K,11], matrices [16,2,4,4], layers [16,2,K+1,60,92].
        s,C=c['state'],c['cameras'];k=s.shape[1]
        state=c['normalized_state'][:,None].expand(-1,2,-1,-1)
        camera=C.flatten(-2)[:,:,None].expand(-1,-1,k,-1)
        t=c['times'][:,None,None,None].expand(16,2,k,1)
        present=c['present'][:,None,:,None].expand(-1,2,-1,-1).float()
        uncertainty=c['uncertainty'][None,None].expand(16,2,-1,-1)
        state=self.state(torch.cat([state,camera,t,present,uncertainty],-1))
        state=state.permute(1,0,2,3).reshape(2,4,4,k,256).permute(0,1,3,2,4).flatten(-2)
        state=self.state_group(state)+c['identity'][None,None]
        z,p,m=(F.adaptive_avg_pool2d(c[name].flatten(0,2),(15,23)).reshape(16,2,k+1,345)
               for name in ('Z','P','M'))
        logz=2*(z.clamp_min(100).log()-math.log(100))/math.log(600)-1
        yy,xx=torch.meshgrid(torch.arange(15,device=z.device),torch.arange(23,device=z.device),indexing='ij')
        pixels=torch.stack([(xx+.5)*32-8-.5,(yy+.5)*32-.5,torch.ones_like(xx)],-1).float().reshape(345,3)
        rays=F.normalize(torch.einsum('tvij,nj->tvni',C[:,:,:3,:3],pixels@torch.linalg.inv(c['K']).T),dim=-1)
        coverage=F.adaptive_avg_pool2d(c['coverage'].flatten(0,1)[:,None],(15,23)).reshape(16,2,1,345)
        coverage=torch.cat([coverage,coverage.new_zeros(16,2,k,345)],2)
        tuples=torch.cat([logz[...,None],p[...,None],m[...,None],rays[:,:,None].expand(-1,-1,k+1,-1,-1),
            c['times'][:,None,None,None,None].expand(16,2,k+1,345,1),coverage[...,None]],-1)
        tuples=tuples.permute(1,0,3,2,4).reshape(2,4,4,345,k+1,8).permute(0,1,3,4,2,5).flatten(-2)
        layer=self.geometry(tuples)
        identity=torch.cat([self.kind.weight[:1],c['identity']+self.kind.weight[1]],0)
        layer=layer+identity[None,None,None]
        support=p.permute(1,0,3,2).reshape(2,4,4,345,k+1).sum(2)>0
        support[...,0]=True
        mask=m.permute(1,0,2,3).reshape(2,4,4,k+1,345)
        full=p.permute(1,0,2,3).reshape(2,4,4,k+1,345)
        direction=F.normalize(s[:,None,:,:3]-C[:,:,:3,3][:,:,None],dim=-1)
        direction=direction.permute(1,0,2,3).reshape(2,4,4,k,3)
        return dict(state=state,layer=layer,layer_support=support,mask=mask,full=full,direction=direction,
            bank=c['bank'],bank_weight=c['bank_weight'],background=c['background'],view_background=c['view_background'])

    def depth_features(self,c):
        # Static and target surfaces stay separate in the transformer. The
        # depth pyramid additionally gets their visible front-surface summary.
        z=c['Z'].clamp_min(100)
        logz=2*(z.log()-math.log(100))/math.log(600)-1
        weight=c['M'];den=weight.sum(2).clamp_min(1e-6)
        visible=(logz*weight).sum(2)/den
        v=weight.sum(2).clamp(0,1)
        full=c['P'][:, :, 1:].sum(2).clamp(0,1)
        x=torch.stack([visible,v,full,c['coverage']],2).permute(1,2,0,3,4)
        x=torch.cat([x[:,:,:1],x],2)
        return self.depth_geometry(x)


class PairedAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(3072);self.query=nn.Linear(3072,256)
        self.state_attn=nn.MultiheadAttention(256,8,batch_first=True)
        self.background_attn=nn.MultiheadAttention(256,8,batch_first=True)
        self.geo_query=nn.Linear(256,256);self.geo_key=nn.Linear(256,256);self.geo_value=nn.Linear(256,256)
        self.app_query=nn.Linear(256+3,256);self.app_key=nn.Linear(256,256);self.app_value=nn.Linear(256,256)
        self.partner=nn.Linear(256,256,bias=False)
        self.ff=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1024),nn.GELU(),nn.Linear(1024,256))
        self.output=nn.Linear(256,3072,bias=False);nn.init.zeros_(self.output.weight)

    def appearance(self,q,c,gate,ordered,slot_permutation=None):
        """All partner appearance is inside the four gated packets.

        q [2,4,345,256] is held fixed for control-invariance checks. A joint
        slot permutation moves masks, viewing directions and partner pooling
        together, before the only within-group time code (gate.time).
        """
        m,p=c['mask'][:,:,:,1:],c['full'][:,:,:,1:]
        directions=c['direction']
        if slot_permutation is not None:
            m=m[:,:,slot_permutation];p=p[:,:,slot_permutation];directions=directions[:,:,slot_permutation]
        total=torch.zeros_like(q)
        for k in range(m.shape[3]):
            # Both branches pool the unmodified q, with four distinct masks.
            source=m[:,:,:,k]
            pooled=torch.einsum('vgrn,vgnc->vgrc',source,q)/source.sum(-1,keepdim=True).clamp_min(1e-6)
            ratio=source.sum(-1)/p[:,:,:,k].sum(-1).clamp_min(1e-6)
            partner=self.partner(pooled.flip(0))
            weight=ratio.flip(0);partner=partner*torch.where(weight>=.05,weight,0)[...,None]
            query=self.app_query(torch.cat([q[:,:,None].expand(-1,-1,4,-1,-1),
                directions[:,:,:,k,None].expand(-1,-1,-1,345,-1)],-1))
            bank=c['bank'][k];logw=c['bank_weight'][k].clamp_min(1e-12).log()
            valid=c['bank_weight'][k]>0
            if valid.any():
                key=self.app_key(bank[valid]).unflatten(-1,(8,32))
                value=self.app_value(bank[valid]).unflatten(-1,(8,32))
                logits=torch.einsum('vgrnhc,lhc->vgrnlh',query.unflatten(-1,(8,32)),key)/math.sqrt(32)+logw[valid,None]
                hist=torch.einsum('vgrnlh,lhc->vgrnhc',logits.softmax(-2),value).flatten(-2)
            else: hist=query*0
            features=(hist+partner[:,:,:,None]).permute(0,1,3,2,4)
            masks=source.permute(0,1,3,2).clamp(0,1)
            total=total+gate(features,masks,ordered)
        return total

    def forward(self,H,c,gate,ordered=True):
        q=self.query(self.norm(H.float())).reshape(2,5,345,256)[:,1:]
        bg=torch.cat([c['background'].expand(2,-1,-1),c['view_background']],1)
        b=self.background_attn(q.flatten(1,2),bg,bg,need_weights=False)[0].reshape_as(q)
        s=c['state']; state=torch.zeros_like(q)
        if s.shape[2]:
            state=self.state_attn(q.flatten(0,1),s.flatten(0,1),s.flatten(0,1),need_weights=False)[0].reshape_as(q)
        layer=c['layer'];key=self.geo_key(layer).unflatten(-1,(8,32))
        logits=(self.geo_query(q).unflatten(-1,(8,32))[...,None,:,:]*key).sum(-1)/math.sqrt(32)
        logits=logits.masked_fill(~c['layer_support'][...,None],-torch.inf)
        geo=(logits.softmax(-2)[...,None]*self.geo_value(layer).unflatten(-1,(8,32))).sum(-3).flatten(-2)
        app=self.appearance(q,c,gate,ordered)
        residual=b+state+geo+app
        residual=residual+self.ff(residual)
        output=self.output(residual).to(H.dtype)
        output=torch.cat([output.new_zeros(2,1,345,3072),output],1).reshape_as(H)
        return H+output


class PairedWan(nn.Module):
    def __init__(self,path,empty_text,device):
        super().__init__()
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig
        self.backbone=WanTransformer3DModel.from_pretrained(Path(path)/'transformer',torch_dtype=torch.bfloat16,local_files_only=True).to(device)
        b=self.backbone
        if len(b.blocks)!=30 or b.config.num_attention_heads!=24 or b.config.attention_head_dim!=128:
            raise ValueError('Expected official Wan2.2-TI2V-5B architecture')
        b.requires_grad_(False)
        b.add_adapter(LoraConfig(r=16,lora_alpha=16,lora_dropout=0,target_modules=['to_q','to_k','to_v','to_out.0']))
        # Keep trainable adapters in fp32; frozen transformer matmuls use bf16.
        for p in b.parameters():
            if p.requires_grad: p.data=p.data.float()
        self.encoder=ConditionEncoder().to(device)
        self.adapters=nn.ModuleList([PairedAdapter() for _ in range(10)]).to(device)
        self.gate=OrderedAppearanceGate().to(device)
        text=torch.load(empty_text,map_location=device,weights_only=True)
        if isinstance(text,dict): text=text['embedding']
        if text.shape!=(1,512,4096) or not text.abs().any(): raise ValueError('Official cached empty-string embedding required')
        self.register_buffer('empty_text',text.to(torch.bfloat16),persistent=False)
        self.ordered=True;self.use_checkpoint=True

    def forward(self,z,s,c):
        if z.shape!=(2,48,5,30,46): raise ValueError('Native paired 17-frame clip required')
        b=self.backbone;original=z.dtype
        conditions=self.encoder(c)
        z=z.to(torch.bfloat16);rope=b.rope(z)
        H=b.patch_embedding(z).flatten(2).transpose(1,2).contiguous()
        timestep=torch.ones((2,1725),device=z.device,dtype=torch.float32)*s*1000
        timestep[:,:345]=0
        temb,tproj,text,_=b.condition_embedder(timestep.flatten(),self.empty_text.expand(2,-1,-1),timestep_seq_len=1725)
        tproj=tproj.unflatten(2,(6,-1))
        for index,block in enumerate(b.blocks):
            if self.use_checkpoint and torch.is_grad_enabled():
                H=checkpoint(block,H,text,tproj,rope,use_reentrant=False)
            else: H=block(H,text,tproj,rope)
            if (index+1)%3==0:
                adapter=self.adapters[index//3]
                if self.use_checkpoint and torch.is_grad_enabled():
                    H=checkpoint(adapter,H,conditions,self.gate,self.ordered,use_reentrant=False)
                else: H=adapter(H,conditions,self.gate,self.ordered)
        shift,scale=(b.scale_shift_table.unsqueeze(0)+temb.unsqueeze(2)).chunk(2,dim=2)
        H=(b.norm_out(H.float())*(1+scale.squeeze(2))+shift.squeeze(2)).to(H.dtype)
        H=b.proj_out(H)
        H=H.reshape(2,5,15,23,1,2,2,48).permute(0,7,1,4,2,5,3,6).flatten(6,7).flatten(4,5).flatten(2,3)
        return H.to(original)


class CheckpointCausalResidual(nn.Module):
    """Functionalize the two native cache writes for safe recomputation.

    Passing the mutable global VAE cache directly to checkpoint would replay
    later temporal state on backward. Each call instead owns its two original
    cache tensors and returns the updated tensors without detaching them.
    """
    def __init__(self,block): super().__init__();self.block=block

    def forward(self,x,feat_cache=None,feat_idx=None):
        if not torch.is_grad_enabled(): return self.block(x,feat_cache,feat_idx)
        if feat_cache is None: return checkpoint(self.block,x,use_reentrant=False)
        start=feat_idx[0]
        def functional(value,cache0,cache1):
            local=[cache0,cache1];index=[0]
            result=self.block(value,local,index)
            if index[0]!=2: raise ValueError('Native causal residual cache contract changed')
            return result,*local
        result,c0,c1=checkpoint(functional,x,*feat_cache[start:start+2],use_reentrant=False)
        feat_cache[start:start+2]=[c0,c1];feat_idx[0]=start+2
        return result


class NativeVAE:
    """Wrap the existing frozen official VAE; input gradients remain enabled."""
    def __init__(self,encoders):
        self.vae,self.mean,self.std=encoders.vae,encoders.mean,encoders.std
        self.vae.disable_tiling()
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock
        def wrap(parent):
            for name,child in list(parent.named_children()):
                if isinstance(child,WanResidualBlock): setattr(parent,name,CheckpointCausalResidual(child))
                else: wrap(child)
        wrap(self.vae.decoder)

    @torch.no_grad()
    def encode(self,rgb):
        # uint8 [17,2,480,720,3] -> [2,48,5,30,46]
        x=rgb.permute(1,4,0,2,3).float()/127.5-1
        x=F.pad(x,(8,8,0,0,0,0),mode='replicate')
        return torch.cat([(self.vae.encode(v[None]).latent_dist.mode()-self.mean)/self.std for v in x])

    def _decode(self,z):
        features=[[] for _ in range(5)];handles=[]
        layers=[self.vae.decoder.mid_block,*self.vae.decoder.up_blocks]
        for i,layer in enumerate(layers):
            handles.append(layer.register_forward_hook(lambda module,args,out,i=i:features[i].append(out)))
        try:
            rgb=self.vae.decode(z*self.std+self.mean).sample
            values=[torch.cat(f,2) for f in features]
            if [f.shape[1] for f in values]!=[1024,1024,1024,512,256]:
                raise ValueError('Unexpected official VAE decoder feature contract')
            return (rgb,*values)
        finally:
            for handle in handles: handle.remove()
            self.vae.clear_cache()

    def decode(self,z):
        return self._decode(z)


class DepthScale(nn.Module):
    def __init__(self,input_channels,width):
        super().__init__()
        self.project=nn.Conv3d(input_channels,width,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv3d(width,width,3,padding=1)) for _ in range(2)])

    def forward(self,x):
        x=self.project(x)
        for block in self.blocks: x=x+block(x)
        return x


class RGBDDepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        widths=[256,256,128,64,64];native=[1024,1024,1024,512,256]
        self.rgb=nn.ModuleList([nn.Conv3d(a,b,1) for a,b in zip(native,widths)])
        self.blocks=nn.ModuleList([DepthScale((48 if i==0 else widths[i-1])+w+32,w) for i,w in enumerate(widths)])
        self.platform=nn.Embedding(2,48)
        self.output=nn.Conv3d(64,2,1)

    def forward(self,z,features,geometry,view):
        x=z+self.platform.weight[view][None,:,None,None,None]
        for index,(project,block,feature) in enumerate(zip(self.rgb,self.blocks,features)):
            size=(17,30*2**index,46*2**index)
            x=resize_video(x,size)
            # Project at native resolution before interpolation.
            rgb=resize_video(project(feature),size)
            g=resize_video(geometry,size)
            if torch.is_grad_enabled():
                x=checkpoint(lambda a,b,c,block=block:block(torch.cat([a,b,c],1)),x,rgb,g,use_reentrant=False)
            else: x=block(torch.cat([x,rgb,g],1))
        # Millimeter/log-depth heads must not quantize their output to bf16.
        # Mixed precision is confined to hidden convolutions; the continuous
        # final projection, sigmoid and inverse logarithm stay in fp32.
        with torch.autocast('cuda',enabled=False): value=self.output(x.float())
        log=2*value[:,0].sigmoid()-1
        return dict(log_depth=log,depth_mm=inverse_depth(log),validity_logits=value[:,1])
