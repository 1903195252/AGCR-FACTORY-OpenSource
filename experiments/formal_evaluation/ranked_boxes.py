"""Lossless raw-box/logit sidecars and official RT-DETR ranked detections.

The fixed-threshold reader/tracker is unchanged. A forward hook may pass its
already computed output here; this module performs no image-model inference.
Ranked boxes follow installed Hugging Face RT-DETR focal-loss postprocessing:
top Q of flattened Q x C sigmoid scores, no NMS, and no coordinate clipping.
All Q ranked entries are retained, including scores below the tracking gate.
"""
from functools import lru_cache
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .common import write_json,read_json,sha256,VIEWS

SCHEMA='agcr-ranked-box-sidecar-1'


@lru_cache(maxsize=1)
def ranking_contract():
    import transformers
    from transformers import RTDetrImageProcessor
    source=Path(inspect.getfile(RTDetrImageProcessor))
    return dict(schema=SCHEMA,transformers_version=transformers.__version__,
        official_implementation='RTDetrImageProcessor.post_process_object_detection(use_focal_loss=True)',
        official_source_sha256=sha256(source),sidecar_source_sha256=sha256(__file__),
        class_probability='sigmoid',selection='top Q of flattened Q x C, including repeated query with different class',
        top_k='number of raw queries, 300 in native reader',nms=False,
        score_gate='none; official threshold=-1 retains every probability in [0,1]',
        boxes='normalized cxcywh to original-pixel xyxy, without clipping (official behavior)',
        class_ids='official zero-based class labels + 1: worker, forklift, agv, uav',
        coco_max_dets=[1,10,100],tracking_threshold_unchanged=True,
        distinction='v4 tracking still uses per-query max class, confidence >= .5 and clipped boxes; ranked AP is a separate score-ranked measurement')


def _array(tensor):
    if not isinstance(tensor,torch.Tensor):raise TypeError('Capture actual torch model output')
    t=tensor.detach().cpu()
    # Every bfloat16 value is exactly representable in float32; no rounding to
    # reduced precision. Other numpy-supported source floating dtypes stay as-is.
    if t.dtype==torch.bfloat16:t=t.float()
    return t.numpy().copy()


def export_arrays(output,canvas_size=(480,720)):
    boxes,logits=_array(output['boxes']),_array(output['logits'])
    if boxes.ndim!=3 or boxes.shape[-1]!=4 or logits.shape[:2]!=boxes.shape[:2] or logits.shape[-1]!=4:
        raise ValueError('Expected B x Q x 4 boxes and four-class logits')
    if not np.isfinite(boxes).all() or not np.isfinite(logits).all():raise ValueError('Nonfinite raw reader output')
    size=np.asarray(canvas_size,np.int64)
    if size.shape==(2,):size=np.broadcast_to(size,(len(boxes),2)).copy()
    if size.shape!=(len(boxes),2) or not (size>0).all():raise ValueError('Canvas sizes are (height,width) per image')
    return dict(boxes=boxes,logits=logits,canvas_size=size)


def official_ranked(boxes,logits,canvas_size):
    """CPU postprocessing only, calling the installed official implementation."""
    from transformers import RTDetrImageProcessor
    b=torch.as_tensor(np.asarray(boxes));l=torch.as_tensor(np.asarray(logits))
    if b.ndim!=3 or b.shape[-1]!=4 or l.shape[:2]!=b.shape[:2] or l.shape[-1]!=4:
        raise ValueError('Ranked batch axes/classes differ')
    sizes=torch.as_tensor(np.asarray(canvas_size),dtype=torch.long)
    if sizes.shape==(2,):sizes=sizes[None].expand(len(b),-1)
    if sizes.shape!=(len(b),2) or not torch.isfinite(b).all() or not torch.isfinite(l).all():
        raise ValueError('Invalid ranked boxes/logits/canvas')
    with torch.inference_mode():
        results=RTDetrImageProcessor().post_process_object_detection(
            SimpleNamespace(logits=l,pred_boxes=b),threshold=-1.,target_sizes=sizes,use_focal_loss=True)
        # Same operation supplies query IDs, absent from the official return.
        _,indices=torch.topk(l.sigmoid().flatten(1),l.shape[1],dim=-1)
    rows=[]
    for v,r in enumerate(results):
        if len(r['scores'])!=b.shape[1]:raise ValueError('Official postprocessor dropped a ranked entry')
        rows.append([dict(query_index=int(indices[v,j]//4),class_id=int(r['labels'][j])+1,
            score=float(r['scores'][j]),box_xyxy=r['boxes'][j].tolist()) for j in range(len(r['scores']))])
    return rows


def _validate_sequence(boxes,logits,canvas_size):
    b,l=np.asarray(boxes),np.asarray(logits)
    if b.ndim!=4 or b.shape[1:]!=(2,300,4) or l.shape!=b.shape or len(b)<1:
        raise ValueError('Native sidecar requires T x 2 x 300 x 4 full boxes/logits')
    if not np.isfinite(b).all() or not np.isfinite(l).all():raise ValueError('Invalid native raw queries')
    size=np.asarray(canvas_size,np.int64)
    if size.shape==(2,):size=np.broadcast_to(size,(len(b),2,2)).copy()
    elif size.shape==(2,2):size=np.broadcast_to(size,(len(b),2,2)).copy()
    if size.shape!=(len(b),2,2) or not (size>0).all():raise ValueError('Invalid frame/view canvas sizes')
    return b,l,size


def save_sidecar(path,boxes,logits,canvas_size,metadata):
    """Atomic NPZ then hash marker. A marker is never inferred from partial data."""
    path=Path(path);b,l,size=_validate_sequence(boxes,logits,canvas_size)
    if not isinstance(metadata,dict) or not metadata:raise ValueError('Explicit input/reader provenance metadata required')
    if path.exists() or path.with_suffix('.complete.json').exists():raise ValueError('Do not overwrite an existing sidecar')
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.%d'%os.getpid())
    contract=ranking_contract();payload=json.dumps(metadata,sort_keys=True,separators=(',',':'))
    try:
        with temp.open('wb') as stream:
            np.savez_compressed(stream,boxes=b,logits=l,canvas_size=size,metadata_json=np.asarray(payload))
            stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp.exists():temp.unlink()
    marker=dict(completed=True,sha256=sha256(path),frames=len(b),views=2,queries=300,classes=4,
        boxes_dtype=str(b.dtype),logits_dtype=str(l.dtype),metadata=metadata,ranking=contract)
    write_json(path.with_suffix('.complete.json'),marker)
    return marker


def load_sidecar(path,expected_metadata=None):
    path=Path(path);marker=read_json(path.with_suffix('.complete.json'))
    if marker.get('completed') is not True or marker['sha256']!=sha256(path):raise ValueError('Incomplete or changed ranked sidecar')
    if marker['ranking']!=ranking_contract():raise ValueError('Ranked implementation/protocol changed')
    if expected_metadata is not None and marker['metadata']!=expected_metadata:raise ValueError('Sidecar source input differs')
    with np.load(path,allow_pickle=False) as z:
        b,l,size=_validate_sequence(z['boxes'],z['logits'],z['canvas_size'])
        if json.loads(z['metadata_json'].item())!=marker['metadata']:raise ValueError('Sidecar metadata differs')
    if len(b)!=marker['frames']:raise ValueError('Sidecar frame count differs')
    return dict(boxes=b,logits=l,canvas_size=size),marker


def ranked_records(boxes,logits,meta,time_s,canvas_size=(480,720),score_from_frame=6):
    """COCO-compatible frame rows, retaining all raw queries outside history."""
    b,l,size=_validate_sequence(boxes,logits,canvas_size);times=np.asarray(time_s)
    if times.shape!=(len(b),) or not np.isfinite(times).all() or not (np.diff(times)>0).all():raise ValueError('Ranked timestamp axes differ')
    if not 0<=score_from_frame<len(b):raise ValueError('Explicit scored future range required')
    rows=[]
    for t in range(score_from_frame,len(b)):
        det=official_ranked(b[t],l[t],size[t])
        for v,view in enumerate(VIEWS):
            rows.append(dict(meta,frame=t,time_s=float(times[t]),view=view,canvas_size=size[t,v].tolist(),detections=det[v]))
    return rows


def ranked_coco_map(reference_rows,prediction_rows):
    """Reuse official COCO evaluator; never label the result pure image quality."""
    from .tracking import coco_map
    # The shared row validator requires a unique `track_id` even for box-only
    # AP. Supply a temporary, frame-local detection index solely to that schema
    # validator; COCO never uses it. Do not persist or treat it as an identity.
    rows=(reference_rows,prediction_rows)
    try:
        for group in rows:
            for r in group:
                for i,d in enumerate(r['detections']):
                    if 'track_id' not in d:
                        d['track_id']='ranked-frame-local-%d'%i;d['_temporary_ranked_schema_key']=True
        result=coco_map(reference_rows,prediction_rows)
    finally:
        for group in rows:
            for r in group:
                for d in r['detections']:
                    if d.pop('_temporary_ranked_schema_key',False):d.pop('track_id')
    return dict(**result,ranking=ranking_contract(),
        identity_policy='COCO box AP uses no tracking identity; schema-only frame-local indices are removed before return',
        interpretation='Ranked reference-future agreement mixes forecast/rendering/reader-domain error; not generated-image annotation accuracy')
