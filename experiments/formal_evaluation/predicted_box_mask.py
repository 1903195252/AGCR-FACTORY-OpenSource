"""Prediction-only instance-mask localization; no threshold or box adjustment.

Masks are [..., H, W]; predicted XYXY boxes are [..., 4] and may broadcast
over the leading mask dimensions. Coordinates use the mask's own pixel grid.
Continuous predicted-box bounds include pixel centers in [x0,x1) x [y0,y1).
The caller must resize masks to the box grid first. Returned values preserve
the input dtype and all in-box probabilities; empty masks are not dropped.
"""
import numpy as np

POSTPROCESSING_VERSION='predicted-box-pixel-center-clip-v3'


def clip_numpy(masks,predicted_boxes_xyxy):
    masks=np.asarray(masks);boxes=np.asarray(predicted_boxes_xyxy,dtype=float)
    if masks.ndim<2 or boxes.ndim<1 or boxes.shape[-1]!=4:
        raise ValueError('Require masks [...,H,W] and predicted boxes [...,4]')
    try:boxes=np.broadcast_to(boxes,masks.shape[:-2]+(4,))
    except ValueError as exc:raise ValueError('Predicted boxes do not match mask instances') from exc
    if not np.isfinite(boxes).all() or (boxes[...,2:]<boxes[...,:2]).any():
        raise ValueError('Require finite, noninverted predicted XYXY boxes')
    y=np.arange(masks.shape[-2],dtype=float)+.5;x=np.arange(masks.shape[-1],dtype=float)+.5
    inside_x=(x>=boxes[...,0,None])&(x<boxes[...,2,None])
    inside_y=(y>=boxes[...,1,None])&(y<boxes[...,3,None])
    inside=inside_y[..., :,None]&inside_x[...,None,:]
    return np.where(inside,masks,np.zeros((),dtype=masks.dtype))


def clip_torch(masks,predicted_boxes_xyxy):
    """Torch equivalent; differentiable in masks, with fixed predicted bounds."""
    import torch
    if not isinstance(masks,torch.Tensor):raise TypeError('Expected a mask tensor')
    boxes=torch.as_tensor(predicted_boxes_xyxy,device=masks.device,dtype=torch.float64)
    if masks.ndim<2 or boxes.ndim<1 or boxes.shape[-1]!=4:
        raise ValueError('Require masks [...,H,W] and predicted boxes [...,4]')
    try:boxes=torch.broadcast_to(boxes,masks.shape[:-2]+(4,))
    except RuntimeError as exc:raise ValueError('Predicted boxes do not match mask instances') from exc
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[...,2:]<boxes[...,:2]).any()):
        raise ValueError('Require finite, noninverted predicted XYXY boxes')
    y=torch.arange(masks.shape[-2],device=masks.device,dtype=boxes.dtype)+.5
    x=torch.arange(masks.shape[-1],device=masks.device,dtype=boxes.dtype)+.5
    inside_x=(x>=boxes[...,0,None])&(x<boxes[...,2,None])
    inside_y=(y>=boxes[...,1,None])&(y<boxes[...,3,None])
    inside=inside_y[..., :,None]&inside_x[...,None,:]
    return torch.where(inside,masks,torch.zeros((),dtype=masks.dtype,device=masks.device))
