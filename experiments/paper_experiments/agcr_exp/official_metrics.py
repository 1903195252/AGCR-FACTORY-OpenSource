"""Adapters to independent, actual metric implementations; no proxy substitutions."""
import importlib
import numpy as np


def trackeval_sequence(reference_ids,prediction_ids,similarity):
    """Contiguous IDs over a whole episode; caller supplies class-aware box IoU.

    HOTA: call separately per view. Global IDF1: interleave both views' frames,
    preserving shared identities across views. Do not average per-frame IDF1.
    """
    if not(len(reference_ids)==len(prediction_ids)==len(similarity)):raise ValueError('Unequal sequence lengths')
    def remap(frames):
        unique=sorted({int(x) for row in frames for x in row});mapping={v:i for i,v in enumerate(unique)}
        out=[]
        for row in frames:
            if len(set(row))!=len(row):raise ValueError('Duplicate identity in one camera frame')
            out.append(np.array([mapping[int(x)] for x in row],dtype=int))
        return out,len(unique)
    gt,ng=remap(reference_ids);pred,npred=remap(prediction_ids)
    for a,b,m in zip(gt,pred,similarity):
        if np.shape(m)!=(len(a),len(b)):raise ValueError('Invalid similarity matrix')
        if not np.isfinite(m).all() or np.any((m<0)|(m>1)):raise ValueError('IoUs outside [0,1]')
    return dict(num_timesteps=len(gt),num_gt_ids=ng,num_tracker_ids=npred,
                num_gt_dets=sum(map(len,gt)),num_tracker_dets=sum(map(len,pred)),
                gt_ids=gt,tracker_ids=pred,similarity_scores=similarity)


def tracking_metrics(data):
    try:module=importlib.import_module('trackeval')
    except ImportError as exc:raise RuntimeError('Install official TrackEval; HOTA/IDF1 are not approximated') from exc
    hota=module.metrics.HOTA();identity=module.metrics.Identity({'THRESHOLD':.5,'PRINT_CONFIG':False})
    return dict(HOTA=hota.eval_sequence(data),Identity=identity.eval_sequence(data))


def combine_tracking(sequences):
    import trackeval
    h=trackeval.metrics.HOTA().combine_sequences({str(i):s['HOTA'] for i,s in enumerate(sequences)})
    d=trackeval.metrics.Identity({'THRESHOLD':.5,'PRINT_CONFIG':False}).combine_sequences({str(i):s['Identity'] for i,s in enumerate(sequences)})
    return dict(HOTA=float(np.mean(h['HOTA'])),IDF1=float(d['IDF1']),sufficient_statistics=dict(HOTA=h,Identity=d))


def lpips_model(device='cpu'):
    import lpips
    model=lpips.LPIPS(net='alex',version='0.1').eval().requires_grad_(False).to(device)
    return model


def paired_lpips(model,prediction,reference):
    """[T,2,3,H,W], RGB in [0,1], only original unpadded image pixels."""
    import torch
    if prediction.shape!=reference.shape or prediction.ndim!=5 or prediction.shape[1:3]!=(2,3):
        raise ValueError('LPIPS requires paired RGB videos')
    with torch.no_grad():
        value=model(prediction.flatten(0,1)*2-1,reference.flatten(0,1)*2-1)
    return float(value.mean())
