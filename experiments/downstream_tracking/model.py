"""Official RT-DETR-R18 detector and independently trained tracking/readout heads.

Inputs contain images, metric depth, calibration only. No reference boxes,
instance masks, generator state, or reference track queries enter forward().
"""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def xyxy(box):
    return torch.cat((box[..., :2]-box[..., 2:]/2, box[..., :2]+box[..., 2:]/2),-1)


def box_iou(a,b):
    lo=torch.maximum(a[:,None,:2],b[None,:,:2]);hi=torch.minimum(a[:,None,2:],b[None,:,2:])
    inter=(hi-lo).clamp(min=0).prod(-1)
    aa=(a[:,2:]-a[:,:2]).clamp(min=0).prod(-1);bb=(b[:,2:]-b[:,:2]).clamp(min=0).prod(-1)
    return inter/(aa[:,None]+bb[None,:]-inter).clamp(min=1e-6)


def roi_grid(boxes,size=7):
    axis=(torch.arange(size,device=boxes.device,dtype=boxes.dtype)+.5)/size-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    offset=torch.stack((xx,yy),-1)
    return (boxes[:,:,None,None,:2]+offset*boxes[:,:,None,None,2:])*2-1


class CooperativeDetector(nn.Module):
    def __init__(self,assets,modality='rgbd',pretrained=True,detector=None):
        super().__init__();self.modality=modality
        if detector is None:
            from transformers import RTDetrConfig,RTDetrForObjectDetection
            config=RTDetrConfig.from_pretrained(assets,local_files_only=True)
            if config.backbone_config.depths!=[2,2,2,2] or config.backbone_config.layer_type!='basic':
                raise ValueError('The registered downstream backbone must be RT-DETR-R18')
            config.num_labels=4;config.id2label={i:n for i,n in enumerate(['worker','forklift','agv','uav'])}
            config.label2id={v:k for k,v in config.id2label.items()};config.disable_custom_kernels=True
            detector=(RTDetrForObjectDetection.from_pretrained(assets,config=config,ignore_mismatched_sizes=True,local_files_only=True)
                      if pretrained else RTDetrForObjectDetection(config))
        self.detector=detector;d=detector.config.d_model
        self.identity=nn.Linear(d,128);self.contact=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,2))
        self.pixel=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(8,32),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.ReLU())
        self.mask_query=nn.Linear(d,64)
        # 7x7 x (metric depth, validity, camera XYZ), plus detector query.
        self.depth_head=(nn.Sequential(nn.Linear(d+7*7*5,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,12))
                         if modality=='rgbd' else None)

    def forward(self,images,depth_mm,valid,K,labels=None):
        # Resize to the official 640x640 input; all outputs retain original normalized coordinates.
        rgb=images.float()/255 if images.dtype==torch.uint8 else images
        pixels=F.interpolate(rgb,size=(640,640),mode='bilinear',align_corners=False)
        out=self.detector(pixel_values=pixels,labels=labels,return_dict=True)
        count=self.detector.config.num_queries
        q=out.last_hidden_state[:,-count:];boxes=out.pred_boxes[:,-count:];logits=out.logits[:,-count:]
        features=self.pixel(rgb)
        result=dict(boxes=boxes,logits=logits,identity=F.normalize(self.identity(q),dim=-1),
                    contact=.5+self.contact(q),mask_logits=torch.einsum('bqd,bdhw->bqhw',self.mask_query(q),features)/8)
        # RT-DETR detaches decoder seeds after top-k selection. Its encoder
        # proposal heads therefore need their own post-forward supervision.
        encoder_logits=getattr(out,'enc_topk_logits',None)
        encoder_boxes=getattr(out,'enc_topk_bboxes',None)
        if (encoder_logits is None)!=(encoder_boxes is None):
            raise RuntimeError('Incomplete RT-DETR encoder candidates')
        if encoder_logits is not None:
            result.update(encoder_logits=encoder_logits,encoder_boxes=encoder_boxes)
        if labels is not None: result['detector_loss']=out.loss
        if self.depth_head is not None:
            b,_,h,w=rgb.shape
            yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
            z=depth_mm.float()/1000;v=valid.float();z=torch.where(v>.5,z,torch.zeros_like(z))
            x=(xx[None]-K[:,0,2,None,None])/K[:,0,0,None,None]*z
            y=(yy[None]-K[:,1,2,None,None])/K[:,1,1,None,None]*z
            field=torch.stack((z/60,v,x/60,y/60,z/60),1)
            grid=roi_grid(boxes.detach()).reshape(b,count*7,7,2)
            pooled=F.grid_sample(field,grid,align_corners=False).reshape(b,5,count,7,7).permute(0,2,1,3,4).flatten(2)
            head=self.depth_head(torch.cat((q,pooled),-1))
            result.update(center_camera_m=head[...,:3],dimensions_m=F.softplus(head[...,3:6])+.01,
                          heading_camera=F.normalize(head[...,6:9],dim=-1),center_var_m2=F.softplus(head[...,9:12])+.0001)
        return result


def matching(pred,targets):
    matches=[]
    for i,t in enumerate(targets):
        n=len(t['class_labels'])
        if not n: matches.append(([],[]));continue
        cost=(-pred['logits'][i].sigmoid()[:,t['class_labels']]+5*torch.cdist(pred['boxes'][i].float(),t['boxes'].float(),p=1)
              -2*box_iou(xyxy(pred['boxes'][i]).float(),xyxy(t['boxes']).float()))
        a,b=linear_sum_assignment(cost.detach().cpu().numpy());matches.append((a.tolist(),b.tolist()))
    return matches


def head_losses(pred,targets,temperature=.1):
    zero=sum(v.sum()*0 for v in pred.values() if isinstance(v,torch.Tensor))
    # Each in-place accumulator needs its own storage; keep the detector fallback zero.
    losses={name:zero.clone() for name in ('contact','mask','identity','box3d','covariance')}
    desc=[];ids=[];total=0
    for i,(pi,ti) in enumerate(matching(pred,targets)):
        if not pi: continue
        t=targets[i];n=len(pi);total+=n
        losses['contact']+=F.smooth_l1_loss(pred['contact'][i,pi],t['contact'][ti],reduction='sum')
        m=F.interpolate(t['masks'][ti,None].float(),pred['mask_logits'].shape[-2:],mode='nearest')[:,0]
        logits=pred['mask_logits'][i,pi];prob=logits.sigmoid()
        losses['mask']+=n*(F.binary_cross_entropy_with_logits(logits,m)+(1-(2*(prob*m).sum((-2,-1))+1)/(prob.sum((-2,-1))+m.sum((-2,-1))+1)).mean())
        desc.append(pred['identity'][i,pi]);ids.extend(t['identity_keys'][j] for j in ti)
        if 'center_camera_m' in pred:
            delta=pred['center_camera_m'][i,pi]-t['center_camera_m'][ti]
            var=pred['center_var_m2'][i,pi]
            losses['box3d']+=F.smooth_l1_loss(pred['center_camera_m'][i,pi],t['center_camera_m'][ti],reduction='sum')
            losses['box3d']+=F.smooth_l1_loss(pred['dimensions_m'][i,pi],t['dimensions_m'][ti],reduction='sum')
            losses['box3d']+=(1-(pred['heading_camera'][i,pi]*t['heading_camera'][ti]).sum(-1)).sum()
            losses['covariance']+=.5*(delta.square()/var+var.log()).sum()
    for name in losses: losses[name]=losses[name]/max(total,1)
    if desc:
        d=torch.cat(desc);similar=d@d.T/temperature
        eye=torch.eye(len(ids),device=d.device,dtype=torch.bool)
        positive=torch.tensor([[a==b for b in ids] for a in ids],device=d.device)&~eye
        eligible=positive.any(1)
        if eligible.any():
            logp=similar-similar.masked_fill(eye,-1e4).logsumexp(1,keepdim=True)
            losses['identity']=-(logp*positive).sum(1)[eligible].div(positive.sum(1)[eligible]).mean()
    losses['total']=pred.get('detector_loss',zero)+losses['contact']+losses['mask']+.1*losses['identity']+losses['box3d']+.1*losses['covariance']
    return losses
