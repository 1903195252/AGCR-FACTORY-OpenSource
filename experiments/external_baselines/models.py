"""Actual upstream SynCamMaster and ShareVerse models with factory interfaces.

Both start from their own published pretrained backbone. Adaptation weights are
trained on factory data; an untrained adaptation is never a benchmark result.
"""
import json
from pathlib import Path
import types
import torch
from torch import nn
from torch.nn import functional as F
from .official import syncam_module,shareverse_module,module
from .conditions import HistoryAdapter,strict_inputs,relative_cameras,retrieve_memory,raymaps


def rgb_uint8(video):
    if not torch.isfinite(video).all():raise FloatingPointError('Non-finite native decoded RGB')
    return ((video.clamp(-1,1)+1)*127.5).round().byte()


def weights(path):
    from safetensors.torch import load_file
    result={}
    for shard in sorted(Path(path).glob('*.safetensors')):
        part=load_file(str(shard))
        if set(result)&set(part): raise ValueError('Duplicate weight keys')
        result.update(part)
    if not result: raise FileNotFoundError(path)
    return result


class LoRALinear(nn.Module):
    def __init__(self,base,rank=16):
        super().__init__();self.base=base.requires_grad_(False);self.rank=rank
        self.down=nn.Linear(base.in_features,rank,bias=False);self.up=nn.Linear(rank,base.out_features,bias=False)
        nn.init.normal_(self.down.weight,std=.01);nn.init.zeros_(self.up.weight)
    def forward(self,x):
        return self.base(x)+self.up(self.down(x.float())).to(x.dtype)


def install_shareverse_lora(dit):
    # Official CogVideoX LoRA targets; newly introduced camera/cross-agent
    # modules remain fully trainable, since this factory adaptation initializes
    # them from the base model rather than from authors' CARLA checkpoint.
    for block in dit.transformer_blocks:
        for name in ('to_q','to_k','to_v'):
            setattr(block.attn1,name,LoRALinear(getattr(block.attn1,name)))
        block.attn1.to_out[0]=LoRALinear(block.attn1.to_out[0])


def install_syncam(base,mod):
    for block in base.blocks:
        dim=base.dim
        block.cam_encoder=nn.Linear(12,dim);block.projector=nn.Linear(dim,dim)
        nn.init.zeros_(block.cam_encoder.weight);nn.init.zeros_(block.cam_encoder.bias)
        nn.init.zeros_(block.projector.weight);nn.init.zeros_(block.projector.bias)
        block.norm_mvs=nn.LayerNorm(dim,eps=block.norm1.eps,elementwise_affine=False)
        block.modulation_mvs=nn.Parameter(block.modulation[:,:3].detach().clone())
        block.mvs_attn=mod.SelfAttention(dim,block.self_attn.num_heads,block.self_attn.norm_q.eps)
        block.mvs_attn.load_state_dict(block.self_attn.state_dict(),strict=True)


class SynCamMaster(nn.Module):
    def __init__(self,source,backbone,text_path,device='cuda',modality='rgbd',tiny=False):
        super().__init__();mod=syncam_module(source);self.device_name=device;self.tiny=tiny
        if tiny:
            cfg=dict(dim=96,in_dim=4,ffn_dim=128,out_dim=4,text_dim=32,freq_dim=32,eps=1e-6,patch_size=(1,2,2),num_heads=4,num_layers=2,has_image_input=False)
        else:
            raw=json.loads((Path(backbone)/'transformer/config.json').read_text())
            cfg=dict(dim=raw['num_attention_heads']*raw['attention_head_dim'],in_dim=raw['in_channels'],ffn_dim=raw['ffn_dim'],out_dim=raw['out_channels'],text_dim=raw['text_dim'],freq_dim=raw['freq_dim'],eps=raw['eps'],patch_size=tuple(raw['patch_size']),num_heads=raw['num_attention_heads'],num_layers=raw['num_layers'],has_image_input=False)
        self.dit=mod.WanModel(**cfg)
        if not tiny:
            converted,_=mod.WanModelStateDictConverter().from_diffusers(weights(Path(backbone)/'transformer'))
            self.dit.load_state_dict(converted,strict=True)
        install_syncam(self.dit,mod)
        self.dit.requires_grad_(False)
        for name,p in self.dit.named_parameters():
            if any(t in name for t in ['cam_encoder','projector','mvs_attn','modulation_mvs']):p.requires_grad_(True)
        self.history=HistoryAdapter(cfg['dim'],modality)
        self._history=None
        original=self.dit.patchify
        def patchify(d,x):
            tokens,grid=original(x)
            for block in d.blocks:block.factory_grid=grid[1:]
            if self._history is not None:tokens=self.history(tokens,self._history)
            return tokens,grid
        self.dit.patchify=types.MethodType(patchify,self.dit)
        self.to(device)
        if not tiny:
            from diffusers import AutoencoderKLWan
            self.vae=AutoencoderKLWan.from_pretrained(backbone,subfolder='vae',torch_dtype=torch.float32).to(device).requires_grad_(False).eval()
            scheduler_module=module('_agcr_syncam.flow_match',Path(source)/'diffsynth/schedulers/flow_match.py')
            self.scheduler=scheduler_module.FlowMatchScheduler(shift=5,sigma_min=0.,extra_one_step=True)
            self.scheduler.set_timesteps(1000,training=True)
            data=torch.load(text_path,map_location='cpu',weights_only=True)
            if isinstance(data,dict):data=data.get('embedding',data.get('prompt_embeds',data.get('context')))
            if data is None:raise ValueError('Missing official empty-text tensor')
            self.register_buffer('text',data.to(device),persistent=False)
            # Frozen pretrained parameters bf16; new modules and optimizers FP32.
            for p in self.dit.parameters():
                if not p.requires_grad:p.data=p.data.bfloat16()

    def forward(self,z,t,x):
        self._history=self.history.encode(x)
        context=self.text.expand(2,-1,-1) if not self.tiny else z.new_zeros((2,8,32))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=z.is_cuda):
            out=self.dit(z,timestep=t,cam_emb=relative_cameras(x),context=context,use_gradient_checkpointing=self.training)
        self._history=None
        return out

    @torch.no_grad()
    def encode(self,rgb):
        # T,V,H,W,C -> V,C,T,H,W. Wan2.1 spatial stride 8; 720 is valid.
        v=rgb.permute(1,4,0,2,3).float()/127.5-1
        z=self.vae.encode(v).latent_dist.mode()
        mu=z.new_tensor(self.vae.config.latents_mean)[None,:,None,None,None]
        sd=z.new_tensor(self.vae.config.latents_std)[None,:,None,None,None]
        return (z-mu)/sd

    @torch.no_grad()
    def decode(self,z):
        mu=z.new_tensor(self.vae.config.latents_mean)[None,:,None,None,None]
        sd=z.new_tensor(self.vae.config.latents_std)[None,:,None,None,None]
        out=self.vae.decode(z.float()*sd+mu).sample
        return rgb_uint8(out[:,:,1:17].permute(2,0,3,4,1))

    def loss(self,branch,generator):
        x=strict_inputs(branch['inputs']);target=branch['supervision']['rgb']
        clean=self.encode(torch.cat([x['rgb'][-1:],target],0));anchor=self.encode(x['rgb'][-1:])
        noise=torch.randn(clean.shape,device=clean.device,generator=generator)
        # Exact released timestep distribution and native training weighting.
        index=int(torch.randint(0,1000,(1,),device=clean.device,generator=generator))
        t=self.scheduler.timesteps[index:index+1].to(clean.device)
        noisy=self.scheduler.add_noise(clean,noise,t);noisy[:,:,0:1]=anchor
        out=self(noisy,t,x)
        return F.mse_loss(out[:,:,1:].float(),(noise-clean)[:,:,1:].float())*self.scheduler.training_weight(t).to(clean.device)

    @torch.no_grad()
    def sample(self,x,seed,steps=50):
        strict_inputs(x);anchor=self.encode(x['rgb'][-1:]);generator=torch.Generator(device=anchor.device).manual_seed(seed)
        z=torch.randn((2,anchor.shape[1],5,*anchor.shape[-2:]),device=anchor.device,generator=generator)
        self.scheduler.set_timesteps(steps)
        for t in self.scheduler.timesteps.to(z.device):
            present=x.get('observation_available',torch.ones(6,2,device=z.device,dtype=torch.bool))[-1].bool()
            z[present,:,0:1]=anchor[present];prediction=self(z,t.reshape(1),x)
            z=self.scheduler.step(prediction,t,z)
        z[present,:,0:1]=anchor[present]
        return self.decode(z)


class ShareVerse(nn.Module):
    def __init__(self,source,backbone,text_path,device='cuda',modality='rgbd',tiny=False):
        super().__init__();self.tiny=tiny;self.device_name=device
        if tiny:
            cfg=dict(num_attention_heads=4,attention_head_dim=16,in_channels=8,out_channels=4,num_layers=2,text_embed_dim=32,time_embed_dim=32,sample_height=4,sample_width=4,sample_frames=21,max_text_seq_length=8,use_rotary_positional_embeddings=False,use_learned_positional_embeddings=False)
            dim=64;grid=(2,2)
        else:
            cfg=json.loads((Path(backbone)/'transformer/config.json').read_text());cfg={k:v for k,v in cfg.items() if not k.startswith('_')}
            dim=cfg['num_attention_heads']*cfg['attention_head_dim'];grid=(30,45)
        mod=shareverse_module(source,6,*grid,dim=dim,layers=cfg['num_layers'])
        self.dit=mod.CogVideoXTransformer3DModel(**cfg)
        if not tiny:
            base=weights(Path(backbone)/'transformer');missing,extra=self.dit.load_state_dict(base,strict=False)
            if extra or any(not k.startswith(('camera_modules.','cross_view_modules.')) for k in missing):raise RuntimeError((missing,extra))
        with torch.no_grad():
            for camera in self.dit.camera_modules:
                camera.raymap_proj.weight.zero_();camera.raymap_proj.bias.zero_()
            for i,cross in enumerate(self.dit.cross_view_modules):
                source_state=self.dit.transformer_blocks[i*2].state_dict()
                for name,p in cross.named_parameters():
                    if name.startswith('projector.'):p.zero_()
                    else:
                        origin=source_state[name]
                        if name.startswith('norm1.linear.'):origin=origin[:3*dim]
                        p.copy_(origin)
        self.dit.requires_grad_(False)
        for name,p in self.dit.named_parameters():
            if name.startswith(('camera_modules.','cross_view_modules.')):p.requires_grad_(True)
        install_shareverse_lora(self.dit)
        self.history=HistoryAdapter(dim,modality);self._history=None
        def patch_hook(_module,args,out):
            text_length=args[0].shape[1]
            return torch.cat([out[:,:text_length],self.history(out[:,text_length:],self._history)],1) if self._history is not None else out
        self.dit.patch_embed.register_forward_hook(patch_hook)
        self.to(device)
        if not tiny:
            from diffusers import AutoencoderKLCogVideoX,CogVideoXDDIMScheduler
            self.vae=AutoencoderKLCogVideoX.from_pretrained(backbone,subfolder='vae',torch_dtype=torch.float32).to(device).requires_grad_(False).eval()
            self.scheduler=CogVideoXDDIMScheduler.from_pretrained(backbone,subfolder='scheduler')
            data=torch.load(text_path,map_location='cpu',weights_only=True)
            if isinstance(data,dict):data=data.get('embedding',data.get('prompt_embeds'))
            if data is None or data.shape[-2:]!=(226,4096):raise ValueError('Need CogVideoX T5 empty prompt [1,226,4096]; Wan UMT5 is incompatible')
            self.register_buffer('text',data.to(device),persistent=False)
            for p in self.dit.parameters():
                if not p.requires_grad:p.data=p.data.bfloat16()
            self.dit.enable_gradient_checkpointing()

    def forward(self,z,t,x,anchor,memory,memory_cameras):
        self._history=self.history.encode(x)
        # z: V,5,C,H,W; memory is a clean independent historical latent.
        all_latents=torch.cat((z,memory),1)
        condition=torch.zeros_like(all_latents);condition[:,:1]=anchor
        combined=torch.cat((all_latents,condition),2)
        poses=torch.cat((x['camera_to_world'][-1:],x['planned_camera_to_world'],memory_cameras),0)
        cameras=raymaps(poses,x['K'],x['rgb'].shape[2],x['rgb'].shape[3])
        rotary=rotary_cv=None
        if self.dit.config.use_rotary_positional_embeddings:
            from diffusers.models.embeddings import get_3d_rotary_pos_embed
            h,w=z.shape[-2]//2,z.shape[-1]//2
            rotary=get_3d_rotary_pos_embed(self.dit.config.attention_head_dim,None,(h,w),6,grid_type='slice',max_size=(h,w),device=z.device)
            rotary_cv=get_3d_rotary_pos_embed(self.dit.config.attention_head_dim,None,(h,w),12,grid_type='slice',max_size=(h,w),device=z.device)
        text=self.text if not self.tiny else z.new_zeros((1,8,32))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=z.is_cuda):
            a,g=self.dit(hidden_states_1=combined[:1],hidden_states_2=combined[1:],camera_1=cameras[:1],camera_2=cameras[1:],encoder_hidden_states=text,timestep=t,zero_timestep=torch.zeros_like(t),image_rotary_emb=rotary,image_rotary_emb_cv=rotary_cv,return_dict=False)
        self._history=None
        return torch.cat((a[:,:5],g[:,:5]),0)

    @torch.no_grad()
    def encode(self,rgb):
        v=rgb.permute(1,4,0,2,3).float()/127.5-1
        return (self.vae.encode(v).latent_dist.mode()*self.vae.config.scaling_factor).permute(0,2,1,3,4)

    @torch.no_grad()
    def conditioning(self,x):
        rgb,cameras,ids=retrieve_memory(x)
        # Encode memory separately: causal VAE cannot import target futures into memory.
        memory=self.encode(torch.cat((rgb[:1],rgb),0))[:,-1:]
        anchor=self.encode(x['rgb'][-1:])
        if 'observation_available' in x:
            anchor=anchor*x['observation_available'][-1,:,None,None,None,None].to(anchor.dtype)
            memory=memory*ids.ge(0).any(0)[:,None,None,None,None].to(memory.dtype)
        return anchor,memory,cameras,ids

    def loss(self,branch,generator):
        x=strict_inputs(branch['inputs']);clean=self.encode(torch.cat((x['rgb'][-1:],branch['supervision']['rgb']),0))
        anchor,memory,cameras,_=self.conditioning(x)
        noise=torch.randn(clean.shape,device=clean.device,generator=generator)
        t=torch.randint(0,self.scheduler.config.num_train_timesteps,(1,),device=clean.device,generator=generator)
        noisy=self.scheduler.add_noise(clean,noise,t)
        prediction=self(noisy,t,x,anchor,memory,cameras)
        recovered=self.scheduler.get_velocity(prediction,noisy,t)
        alpha=self.scheduler.alphas_cumprod.to(clean.device)[t]
        return ((recovered.float()-clean.float()).square()/(1-alpha)).mean()

    @torch.no_grad()
    def sample(self,x,seed,steps=50):
        strict_inputs(x);anchor,memory,cameras,_=self.conditioning(x)
        generator=torch.Generator(device=anchor.device).manual_seed(seed)
        z=torch.randn((2,5,*anchor.shape[2:]),device=anchor.device,generator=generator)
        self.scheduler.set_timesteps(steps,device=z.device)
        for t in self.scheduler.timesteps:
            prediction=self(z,t.reshape(1),x,anchor,memory,cameras)
            z=self.scheduler.step(prediction,t,z,eta=0,generator=generator).prev_sample
        out=self.vae.decode(z.permute(0,2,1,3,4).float()/self.vae.config.scaling_factor).sample
        return rgb_uint8(out[:,:,1:17].permute(2,0,3,4,1))


def create(method,**kwargs):
    if method=='syncammaster':return SynCamMaster(**kwargs)
    if method=='shareverse':return ShareVerse(**kwargs)
    raise RuntimeError('No accepted native implementation for '+method+'; see experiments/README.md')
