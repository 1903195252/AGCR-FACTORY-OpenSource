"""LPIPS and FVD from primary implementations, never feature-distance proxies.

FVD weights: StyleGAN-V authors' TorchScript conversion of DeepMind TFHub
Kinetics-400 I3D used by Google's FVD. Preprocessing reproduces the reference
TF1 resize_bilinear coordinates, not PyTorch's half-pixel resize default.
"""
from pathlib import Path
import numpy as np
from .common import sha256,read_json

I3D_URL='https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
I3D_SOURCE='https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py'
FVD_REFERENCE='https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py'
LPIPS_SOURCE='https://github.com/richzhang/PerceptualSimilarity'


def reference_resize224(videos):
    """[B,T,H,W,3] uint8 -> [B,3,T,224,224], TF1 align_corners=False.

    TF1 legacy resize samples old coordinate i*old_size/new_size, whereas
    torch.interpolate(align_corners=False) uses a half-pixel offset. The explicit
    grid matches Google's original reference and clamps border samples.
    """
    import torch
    import torch.nn.functional as F
    if videos.ndim!=5 or videos.shape[-1]!=3:raise ValueError('Expected BTHWC RGB')
    b,t,h,w,_=videos.shape;flat=videos.permute(0,1,4,2,3).reshape(b*t,3,h,w).float()
    yy=torch.arange(224,device=videos.device,dtype=torch.float32)*h/224
    xx=torch.arange(224,device=videos.device,dtype=torch.float32)*w/224
    y=2*yy/max(h-1,1)-1;x=2*xx/max(w-1,1)-1
    gy,gx=torch.meshgrid(y,x,indexing='ij');grid=torch.stack([gx,gy],-1)[None].expand(b*t,-1,-1,-1)
    frames=F.grid_sample(flat,grid,mode='bilinear',padding_mode='border',align_corners=True)
    return (frames.reshape(b,t,3,224,224).permute(0,2,1,3,4)/127.5-1).contiguous()


class VideoMetrics:
    def __init__(self,assets,device='cpu',lpips=True,fvd=True):
        import torch
        self.device=device;self.torch=torch;self.assets=Path(assets);self.lpips=None;self.i3d=None
        evidence=read_json(self.assets/'assets.json')
        if fvd:
            weight=self.assets/'i3d_torchscript.pt'
            if sha256(weight)!=evidence['i3d']['sha256']:raise ValueError('I3D weight checksum changed')
            self.i3d=torch.jit.load(str(weight),map_location=device).eval()
        if lpips:
            import lpips as package
            self.lpips=package.LPIPS(net='alex',version='0.1',verbose=False).eval().requires_grad_(False).to(device)
        self.provenance=evidence

    def image_scores(self,prediction,reference):
        torch=self.torch
        if prediction.shape!=reference.shape or prediction.ndim!=5 or prediction.shape[1]!=2 or prediction.shape[-1]!=3:
            raise ValueError('Expected [T,2,H,W,3] native RGB')
        p=torch.from_numpy(prediction.copy()).to(self.device).permute(0,1,4,2,3).flatten(0,1).float()/127.5-1
        r=torch.from_numpy(reference.copy()).to(self.device).permute(0,1,4,2,3).flatten(0,1).float()/127.5-1
        with torch.no_grad():
            values=[self.lpips(p[i:i+1],r[i:i+1]).item() for i in range(len(p))] if self.lpips is not None else []
        error=np.square((prediction.astype(np.float64)-reference.astype(np.float64))/255).mean(axis=(2,3,4))
        from skimage.metrics import structural_similarity
        ssim=[]
        for t in range(len(prediction)):
            for view in range(2):
                ssim.append(structural_similarity(reference[t,view],prediction[t,view],channel_axis=-1,data_range=255,
                    gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
        return dict(paired_LPIPS=None if not values else float(np.mean(values)),
            PSNR_dB=float(np.mean(-10*np.log10(np.maximum(error,1e-12)))),SSIM=float(np.mean(ssim)))

    def features(self,videos):
        torch=self.torch
        if self.i3d is None:raise ValueError('Verified I3D backend not loaded')
        x=torch.from_numpy(np.ascontiguousarray(videos)).to(self.device)
        with torch.no_grad():
            z=reference_resize224(x);out=self.i3d(z,rescale=False,resize=False,return_features=True)
        result=out.cpu().numpy()
        if result.shape!=(len(videos),400) or not np.isfinite(result).all():raise ValueError('Expected TFHub I3D 400 pre-softmax logits')
        return result


def frechet(x,y):
    """PSD eigendecomposition avoids spurious complex roots of singular matrices."""
    x,y=np.asarray(x,np.float64),np.asarray(y,np.float64)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y))<2:raise ValueError('FVD needs >=2 full videos per population')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite I3D features')
    a,b=np.cov(x,rowvar=False),np.cov(y,rowvar=False);s,u=np.linalg.eigh(a)
    root=(u*np.sqrt(np.maximum(s,0)))@u.T
    middle=(root@b@root);middle=(middle+middle.T)/2
    trace_root=np.sqrt(np.maximum(np.linalg.eigvalsh(middle),0)).sum()
    return float(max(0,np.square(x.mean(0)-y.mean(0)).sum()+np.trace(a)+np.trace(b)-2*trace_root))


def fvd_eight(reference_features,generated_features):
    """reference [N,2,400], generated [8,N,2,400]; one complete set per draw."""
    r,g=np.asarray(reference_features),np.asarray(generated_features)
    if g.shape!=(8,*r.shape) or r.ndim!=3 or r.shape[1:]!=(2,400):raise ValueError('FVD must retain eight complete paired test sets')
    scores=np.array([[frechet(r[:,v],g[s,:,v]) for v in range(2)] for s in range(8)])
    return dict(FVD=float(scores.mean()),FVD_air=float(scores[:,0].mean()),FVD_ground=float(scores[:,1].mean()),
        by_sample_index=scores.tolist(),videos_per_population=len(r),backend='TFHub-I3D-400-author-TorchScript',preprocessing='Google-reference-224-TF1-bilinear')
