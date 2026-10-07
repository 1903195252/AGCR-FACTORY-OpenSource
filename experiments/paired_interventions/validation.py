"""Fixed held-out validation panel, separate from final independent metrics."""
import hashlib
import torch

from .data import to_device


def panel_indices(dataset,count,seed=17029):
    rows=range(len(dataset))
    # Interleave cohort/direction cells, then hash-order within each stratum.
    buckets={}
    for i in rows:
        row=dataset.families[i];key=(row['cohort'],row['receiver'])
        buckets.setdefault(key,[]).append(i)
    for indices in buckets.values():
        indices.sort(key=lambda i:hashlib.sha256((str(seed)+dataset.families[i]['family_id']).encode()).hexdigest())
    result=[]
    while any(buckets.values()) and len(result)<count:
        for key in sorted(buckets):
            if buckets[key]: result.append(buckets[key].pop(0))
            if len(result)==count: break
    return result


@torch.no_grad()
def validate(model,dataset,indices,device,step,rank=0,world_size=1):
    rows=[];upstream_training=model.upstream.training;reader_training=model.reader.training
    model.upstream.eval();model.reader.eval()
    try:
        for position in range(rank,len(indices),world_size):
            family=to_device(dataset[indices[position]],device)
            seed=int(hashlib.sha256(family['metadata']['family_id'].encode()).hexdigest()[:8],16)
            generator=torch.Generator(device=device).manual_seed(seed)
            if model.phase=='video' and step>40000:
                # Fixed common random draws; decoded errors are history-only.
                measurements=[]
                for draw in range(8):
                    g=torch.Generator(device=device).manual_seed(seed+draw)
                    loss,terms=model.feedback(family,g,draw%2,0.)
                    # Checkpoint selection is identical across ablations.
                    # Their training losses may disable the camera term, but
                    # that must not disable it in the held-out ranking rule.
                    measurements.append(float(.1*terms['reappearance']+.1*terms['camera']))
                value=sum(measurements)/8;scope='fixed_reappearance_camera_mean_of_8'
            else:
                branch=dict(family['branches'][(position%2,(position//2)%2)],cohort=family['metadata']['cohort'])
                loss,_=model(branch,generator,step,family,False,0.)
                value=float(loss);scope='reference_conditioned_validation_objective'
            rows.append(dict(family=family['metadata']['family_id'],value=value,scope=scope))
    finally:
        model.upstream.train(upstream_training);model.reader.train(reader_training)
    gathered=[None]*world_size
    if world_size>1: torch.distributed.all_gather_object(gathered,rows)
    else: gathered[0]=rows
    all_rows=[row for part in gathered for row in part]
    if not all_rows: raise ValueError('Empty validation panel')
    if not all(torch.isfinite(torch.tensor(r['value'])) for r in all_rows): raise FloatingPointError('Invalid held-out validation')
    return dict(step=step,mean=sum(row['value'] for row in all_rows)/len(all_rows),rows=all_rows,
        scope='model_selection_auxiliary_objectives',independent_benchmark_metrics=False)
