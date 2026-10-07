"""Capture all raw RT-DETR queries beside an unchanged fixed-threshold reader.

The scoped wrapper checks the native corpus loop order. It does not modify
model output, tracker inputs, association thresholds, or reader source code.
"""
from collections import deque
from pathlib import Path
import numpy as np

from experiments.formal_evaluation import ranked_boxes
from experiments.formal_evaluation.common import read_json,sha256,SAMPLE_SEEDS
from experiments.downstream_tracking.inference import reader_implementation


def ensure_sidecar(path,boxes,logits,size,metadata):
    path=Path(path);marker=path.with_suffix('.complete.json')
    if marker.is_file():
        previous,_=ranked_boxes.load_sidecar(path,metadata)
        if not np.array_equal(previous['boxes'],boxes) or not np.array_equal(previous['logits'],logits):
            raise ValueError('Repeated actual forward differs from committed raw-query sidecar')
        return
    if path.exists():
        # Only a marker-less derived sidecar in the new output is incomplete.
        # Retain its bytes for audit; never remove source pixels or committed data.
        orphan=path.with_name(path.name+'.uncommitted.'+sha256(path))
        if orphan.exists() and sha256(orphan)!=sha256(path):raise ValueError('Partial sidecar audit collision')
        path.replace(orphan)
    ranked_boxes.save_sidecar(path,boxes,logits,size,metadata)


class RankedReadoutCapture:
    def __init__(self,index_path,predictions,output,checkpoint,association):
        index=read_json(index_path);self.pending=deque();self.rows=[]
        common=dict(checkpoint_sha256=sha256(checkpoint),association_sha256=sha256(association),
                    index_sha256=sha256(index_path),reader=reader_implementation(),
                    input_scope='six observed history frames plus sixteen generated future frames; references never select queries',
                    tracking_output_unchanged=True)
        for family in index['families']:
            if family['split']!='test':continue
            folder=Path(output)/family['family_id']
            for record in family['records']:
                world,camera=record['world'],record['camera'];ep='%s_w%d_c%d'%(family['family_id'],world,camera)
                for draw,seed in enumerate(SAMPLE_SEEDS):
                    generated=Path(predictions)/family['family_id'];draw_marker=read_json(generated/('draw_%02d.json'%draw))
                    source='world%d_camera%d_sample%02d.npz'%(world,camera,draw)
                    dest=folder/('prediction_w%d_c%d_s%02d.jsonl'%(world,camera,draw));ranked=folder/('ranked_w%d_c%d_s%02d.npz'%(world,camera,draw))
                    metadata=dict(common,family_id=family['family_id'],episode=ep,world=world,camera=camera,draw=draw,seed=seed,
                        native_history_sha256=record['sha256'],generated_source_sha256=draw_marker['files'][source],
                        score_from_frame=6,frames=22,views=2)
                    row=dict(prediction=dest,ranked=ranked,metadata=metadata);self.rows.append(row)
                    if dest.with_suffix('.complete.json').exists():
                        if read_json(dest.with_suffix('.complete.json')).get('sha256')!=sha256(dest):raise ValueError('Changed completed v4 tracking output')
                        ranked_boxes.load_sidecar(ranked,metadata)
                    else:self.pending.append(row)

    def wrapper(self,original):
        def capture(model,association,rgb,depth,valid,K,poses,times,meta,floor_height_m=0.):
            if not self.pending:raise ValueError('Unexpected extra reader clip')
            row=self.pending[0]
            if meta.get('episode')!=row['metadata']['episode'] or len(rgb)!=22 or len(times)!=22 or rgb.shape[1]!=2:
                raise ValueError('Native reader iteration order/frame axes changed')
            outputs=[]
            def hook(_module,_args,output):outputs.append(ranked_boxes.export_arrays(output,rgb.shape[2:4]))
            handle=model.register_forward_hook(hook)
            try:result=original(model,association,rgb,depth,valid,K,poses,times,meta,floor_height_m=floor_height_m)
            finally:handle.remove()
            if len(outputs)!=22:raise ValueError('Capture must see one actual paired image forward per frame')
            boxes=np.stack([v['boxes'] for v in outputs]);logits=np.stack([v['logits'] for v in outputs])
            size=np.stack([v['canvas_size'] for v in outputs])
            ensure_sidecar(row['ranked'],boxes,logits,size,row['metadata'])
            self.pending.popleft();return result
        return capture

    def verify_complete(self):
        if self.pending:raise ValueError('Missing unfiltered query sidecars for complete corpus')
        for row in self.rows:
            ranked_boxes.load_sidecar(row['ranked'],row['metadata'])
        return dict(completed=True,clips=len(self.rows),frames_per_clip=22,tracking_threshold_unchanged=True,
                    ranking=ranked_boxes.ranking_contract(),source_sha256=sha256(__file__))
