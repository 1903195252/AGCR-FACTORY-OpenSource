"""DINOv2 RGB-D observation head for four classes and variable target counts."""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import backproject,normalize_depth
from .modules import CrossBlock


class ObservationHead(nn.Module):
    def __init__(self,width=256,heads=8,layers=6,queries=64,classes=4):
        super().__init__();self.width=width
        self.depth=nn.Conv2d(2,128,14,stride=14);self.fuse=nn.Linear(768+128,width)
        self.position=nn.Sequential(nn.Linear(9,width),nn.GELU(),nn.Linear(width,width))
        self.platform=nn.Embedding(2,width);self.queries=nn.Parameter(torch.randn(queries,width)*.02)
        self.blocks=nn.ModuleList([CrossBlock(width,heads) for _ in range(layers)])
        self.classifier=nn.Linear(width,classes+1);self.full=nn.Linear(width,width);self.visible=nn.Linear(width,width)
        self.center=nn.Linear(width,3);self.fallback_depth=nn.Linear(width,1);self.covariance=nn.Linear(width,3)
        self.dimensions=nn.Linear(width,3);self.heading=nn.Linear(width,2);self.reliability=nn.Linear(width,1)
        self.association=nn.Sequential(nn.Linear(width*2+3,width),nn.GELU(),nn.Linear(width,1))

    def forward(self,dino,depth_mm,valid,K,T,time_s,view,rgb_only=False):
        b,h,w=depth_mm.shape
        if (h,w)!=(480,720) or dino.shape[1:]!=(864,768):raise ValueError('Native image 480x720; DINO patch grid 24x36')
        log,ok=normalize_depth(depth_mm,valid)
        if rgb_only:log=torch.zeros_like(log);ok=torch.zeros_like(ok)
        depth=F.interpolate(torch.stack([log,ok.float()],1),size=(336,504),mode='nearest')
        fused=self.fuse(torch.cat([dino,self.depth(depth).flatten(2).transpose(1,2)],-1))
        points=backproject(torch.where(ok,depth_mm,0),K,T)
        weight=F.avg_pool2d(ok[:,None].float(),20,20)
        world=F.avg_pool2d((points*ok[...,None]).permute(0,3,1,2),20,20)/weight.clamp_min(1e-8)
        world=world.flatten(2).transpose(1,2)
        y,x=torch.meshgrid(torch.arange(24,device=dino.device),torch.arange(36,device=dino.device),indexing='ij')
        pixels=torch.stack([(x+.5)*20-.5,(y+.5)*20-.5,torch.ones_like(x)],-1).to(dino.dtype).flatten(0,1)
        camera_ray=torch.einsum('bij,nj->bni',torch.linalg.inv(K),pixels)
        rays=F.normalize(torch.einsum('bij,bnj->bni',T[:,:3,:3],camera_ray),dim=-1)
        uv=(pixels[:,:2]/pixels.new_tensor([720,480])*2-1)[None].expand(b,-1,-1)
        pos=torch.cat([world/20,uv,rays,time_s[:,None,None].expand(-1,864,1)],-1)
        memory=fused+self.position(pos)+self.platform(view)[:,None]
        query=self.queries[None].expand(b,-1,-1)
        for layer in self.blocks:query=layer(query,memory)
        P=torch.einsum('bqc,bnc->bqn',self.full(query),fused).div(self.width**.5).sigmoid()
        M=P*torch.einsum('bqc,bnc->bqn',self.visible(query),fused).div(self.width**.5).sigmoid()
        support=M*weight.flatten(2)
        centroid=torch.einsum('bqn,bnc->bqc',support,world)/support.sum(-1,keepdim=True).clamp_min(1e-8)
        ray=F.normalize(torch.einsum('bqn,bnc->bqc',M,rays),dim=-1)
        fallback=T[:,:3,3][:,None]+ray*(F.softplus(self.fallback_depth(query))+.1)
        center=torch.where(support.sum(-1,keepdim=True)>1e-5,centroid,fallback)+self.center(query)
        return dict(class_logits=self.classifier(query),P=P.reshape(b,-1,24,36),M=M.reshape(b,-1,24,36),
                    center=center,covariance=F.softplus(self.covariance(query))+1e-4,
                    dimensions=F.softplus(self.dimensions(query))+.01,
                    heading=F.normalize(self.heading(query),dim=-1),reliability=self.reliability(query).sigmoid().squeeze(-1),
                    tokens=query,scene_features=memory)

    def association_logits(self,left,right):
        a=left['tokens'][:,None].expand(-1,len(right['tokens']),-1)
        b=right['tokens'][None].expand(len(left['tokens']),-1,-1)
        delta=left['center'][:,None]-right['center'][None]
        variance=left['covariance'][:,None]+right['covariance'][None]
        mahal=(delta.square()/variance).sum(-1).sqrt()
        compatibility=left['class_logits'].softmax(-1)@right['class_logits'].softmax(-1).T
        cosine=F.cosine_similarity(a,b,dim=-1)
        return self.association(torch.cat([a,b,mahal[...,None],compatibility[...,None],cosine[...,None]],-1)).squeeze(-1)


def observation_loss(model,output,target):
    """Per-view Hungarian SUPERVISION, never passed into model inference.

    Targets: center/dimensions/heading/class_id, masks and detectable [B,N].
    Pair association is pretrained from independently matched A/G detections.
    """
    from scipy.optimize import linear_sum_assignment
    device=output['center'].device;b,q,_=output['center'].shape
    assigned=[];losses=[]
    for view in range(b):
        ids=torch.where(target['detectable'][view])[0]
        classes=torch.zeros(q,device=device,dtype=torch.long)
        p=output['class_logits'][view].softmax(-1)
        if len(ids):
            cost=torch.cdist(output['center'][view].detach(),target['center'][view,ids],p=1)/3
            cost=cost-p[:,target['class_id'][ids]].detach()
            rows,cols=linear_sum_assignment(cost.cpu().numpy());rows=torch.as_tensor(rows,device=device);cols=ids[torch.as_tensor(cols,device=device)]
            classes[rows]=target['class_id'][cols]
        else:rows=cols=torch.empty(0,device=device,dtype=torch.long)
        assigned.append((rows,cols))
        weight=torch.ones(p.shape[-1],device=device);weight[0]=.1
        loss=F.cross_entropy(output['class_logits'][view],classes,weight=weight)
        if len(rows):
            error=output['center'][view,rows]-target['center'][view,cols];cov=output['covariance'][view,rows]
            loss=loss+.5*(error.square()/cov+cov.log()).mean()
            loss=loss+F.l1_loss(output['dimensions'][view,rows],target['dimensions'][view,cols])
            loss=loss+(1-(output['heading'][view,rows]*target['heading'][view,cols]).sum(-1)).mean()
            for key,label in [('P','full_mask'),('M','visible_mask')]:
                pred=output[key][view,rows].clamp(1e-6,1-1e-6);ref=target[label][view,cols]
                loss=loss+F.binary_cross_entropy(pred,ref)+(1-(2*(pred*ref).sum((-2,-1))+1)/(pred.sum((-2,-1))+ref.sum((-2,-1))+1)).mean()
            reliability=torch.exp(-error.detach().norm(dim=-1)) # standardized 1 m center residual
            loss=loss+F.binary_cross_entropy(output['reliability'][view,rows],reliability)
        losses.append(loss)
    association=output['tokens'].sum()*0
    if b==2 and all(len(rows) for rows,cols in assigned):
        left={k:v[0,assigned[0][0]] for k,v in output.items() if k in ['tokens','center','covariance','class_logits']}
        right={k:v[1,assigned[1][0]] for k,v in output.items() if k in left}
        logits=model.association_logits(left,right)
        truth=(assigned[0][1][:,None]==assigned[1][1][None]).float()
        association=F.binary_cross_entropy_with_logits(logits,truth)
    return torch.stack(losses).mean()+association,dict(observation=float(torch.stack(losses).mean().detach()),association=float(association.detach()))
