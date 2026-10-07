"""Native branch arrays -> strict StageTwo inputs and separate supervision.

Only the fields returned under inputs may reach historical_posterior.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


CLASS_MAP = {3:1, 4:2, 13:3, 14:4}


def load_branch(path, device='cpu'):
    with np.load(Path(path),allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    if arrays['rgb'].shape[:2]!=(22,2):
        raise ValueError('Expected a 6+16 frame paired branch')
    K=arrays['K']
    if K.shape!=(22,2,3,3) or not np.array_equal(K,np.broadcast_to(K[0,0],K.shape)):
        raise ValueError('Current StageTwo requires fixed, shared intrinsics; per-camera intrinsics need an explicit frontend adaptation')
    def tensor(array):
        value=torch.from_numpy(np.ascontiguousarray(array)).to(device)
        return value.float() if value.dtype==torch.float64 else value
    inputs=dict(rgb=tensor(arrays['rgb'][:6]),depth_mm=tensor(arrays['depth_mm'][:6]),
                depth_valid=tensor(arrays['valid'][:6].astype(bool)),K=tensor(K[0,0]),
                camera_to_world=tensor(arrays['camera_to_world'][:6]),
                planned_camera_to_world=tensor(arrays['camera_to_world'][6:]),
                time_s=tensor(arrays['time_s'][:6]),future_time_s=tensor(arrays['time_s'][6:]))
    if any(int(value) not in CLASS_MAP for value in arrays['class_id']):
        raise ValueError('Unexpected native semantic class')
    classes=np.asarray([CLASS_MAP[int(value)] for value in arrays['class_id']],np.int64)
    supervision=dict(history_state=tensor(arrays['world_state'][:6]),state=tensor(arrays['world_state'][6:]),
                     class_id=tensor(classes),native_class_id=tensor(arrays['class_id']),instance_id=tensor(arrays['instance_id']),
                     present=tensor(arrays['present'][6:].astype(bool)),
                     history_instance=tensor(arrays['instance'][:6].astype(np.int64)),
                     future_instance=tensor(arrays['instance'][6:].astype(np.int64)),
                     rgb=tensor(arrays['rgb'][6:]),depth_mm=tensor(arrays['depth_mm'][6:]),
                     valid=tensor(arrays['valid'][6:].astype(bool)))
    for name in ('visible_fraction','projection_area','static_depth_mm','static_valid','isolated_depth_mm','isolated_support'):
        if name in arrays:
            supervision[name]=tensor(arrays[name][6:])
    supervision['history_visible_fraction']=tensor(arrays['visible_fraction'][:6])
    supervision['history_projection_area']=tensor(arrays['projection_area'][:6])
    if 'isolated_support' in arrays:
        supervision['history_isolated_support']=tensor(arrays['isolated_support'][:6])
    return dict(inputs=inputs,supervision=supervision)


def to_device(value,device):
    if isinstance(value,torch.Tensor): return value.to(device)
    if isinstance(value,dict): return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list): return [to_device(v,device) for v in value]
    return value


def replay_sample(branch,auxiliary_offset=0):
    """Native supervision -> the unchanged stage-two replay interface."""
    from experiments.paper_experiments.agcr_exp.stage2_data import area8,patch20,depth8
    x,y=branch['inputs'],branch['supervision']
    def cpu(v): return v.detach().cpu().numpy()
    def pool(v,fn): return torch.from_numpy(fn(cpu(v))).to(x['rgb'].device)
    ids=y['instance_id']
    hmask=y['history_instance'][:,:,None]==ids[None,None,:,None,None]
    fmask=y['future_instance'][:,:,None]==ids[None,None,:,None,None]
    detectable=(y['history_visible_fraction']>=.1)&(y['history_projection_area']>=144)
    z,p=depth8(cpu(y['isolated_depth_mm']));sz,sp=depth8(cpu(y['static_depth_mm']))
    labels=dict(state=y['state'],history_state=y['history_state'],class_id=y['class_id'],
        history_eligible=detectable.any(0).any(0),present=y['present'],
        history_present=torch.ones_like(y['history_state'][...,0],dtype=torch.bool),
        history_full_mask=pool(y['history_isolated_support'],patch20),history_visible_mask=pool(hmask,patch20),
        history_detectable=detectable,isolated_depth_mm=torch.from_numpy(z).to(x['rgb'].device),
        full_mask=torch.from_numpy(p).to(x['rgb'].device),visible_mask=pool(fmask,area8),
        static_depth_mm=torch.from_numpy(sz).to(x['rgb'].device),static_support=torch.from_numpy(sp).to(x['rgb'].device))
    a=auxiliary_offset
    aux=dict(rgb=y['rgb'][a],depth_mm=y['depth_mm'][a],valid=y['valid'][a],K=x['K'],
        T=x['planned_camera_to_world'][a],state=y['state'][a],class_id=y['class_id'],
        full_mask=pool(y['isolated_support'][a],patch20),visible_mask=pool(fmask[a],patch20),
        detectable=(y['visible_fraction'][a]>=.1)&(y['projection_area'][a]>=144))
    return dict(inputs=x,labels=labels,auxiliary=aux)


def historical_labels(branch):
    """Only the three historical fields consumed by Hungarian supervision.

    Constructing all future surface targets just to associate past queries
    causes unnecessary native-resolution masks and GPU-to-CPU transfers.
    These tensors never enter the history encoder or state prediction.
    """
    y=branch['supervision']
    return dict(history_state=y['history_state'],class_id=y['class_id'],
        history_detectable=(y['history_visible_fraction']>=.1)&(y['history_projection_area']>=144))


class FamilySampler:
    """Checkpointable global permutation; a rank owns every cell of a family.

    The cursor counts global draws, making resume independent of loader
    prefetch. The caller advances it only after a completed optimizer update.
    """
    def __init__(self,size,seed=17,cursor=0):
        if size<1: raise ValueError('Empty family stream')
        self.size,self.seed,self.cursor=size,seed,cursor

    def index(self,offset):
        epoch,position=divmod(self.cursor+offset,self.size)
        return int(np.random.default_rng(np.random.SeedSequence([self.seed,epoch])).permutation(self.size)[position])

    def rank_indices(self,batch,rank,world_size):
        if batch%world_size: raise ValueError('Global batch must divide evenly across ranks')
        return [self.index(i) for i in range(rank,batch,world_size)]

    def advance(self,batch): self.cursor+=batch

    def state_dict(self): return dict(size=self.size,seed=self.seed,cursor=self.cursor)


class OrdinaryDataset:
    """The accepted original collection in the same supervision interface."""
    def __init__(self,index_path,split='train'):
        from experiments.paper_experiments.agcr_exp.data import Dataset
        self.dataset=Dataset(index_path,split,primary=True,load_surfaces=True)
        self.index=self.dataset.index

    def __len__(self): return len(self.dataset)

    def __getitem__(self,index):
        from experiments.paper_experiments.agcr_exp.stage2_system import tensors
        sample=tensors(self.dataset[index],'cpu');x,y=sample['inputs'],sample['labels']
        x={k:v for k,v in x.items() if k not in ('ego_to_world','observation_available')}
        ids=y['track_id']
        def instance(m): return (m.long()*ids[None,None,:,None,None]).sum(2)
        supervision=dict(history_state=y['history_state'],state=y['state'],class_id=y['class_id'],instance_id=ids,
            present=y['present'],history_instance=instance(y['history_visible_mask']),future_instance=instance(y['visible_mask']),
            rgb=y['rgb'],depth_mm=y['depth_mm'],valid=y['depth_valid'],visible_fraction=y['visible_fraction'],
            projection_area=y['full_area'],history_visible_fraction=y['history_visible_fraction'],
            history_projection_area=y['history_full_mask'].sum((-2,-1)),history_isolated_support=y['history_full_mask'],
            static_depth_mm=y['static_depth_mm'],isolated_depth_mm=y['isolated_depth_mm'],isolated_support=y['full_mask'])
        return dict(inputs=x,supervision=supervision,metadata=sample['metadata'])


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(data)
    return h.hexdigest()


class FamilyDataset:
    """Whole audited factorial families, never split into unrelated clip rows."""
    def __init__(self,index_path,split,verify_files=True):
        self.path=Path(index_path).resolve()
        self.index=json.loads(self.path.read_text())
        if self.index.get('schema')!='agcr-intervention-index-1' or not self.index.get('audit_passed'):
            raise ValueError('An audited native intervention index is required')
        self.root=Path(self.index['root'])
        self.families=[row for row in self.index['families'] if row['split']==split]
        if not self.families:
            raise ValueError('No complete families in split '+split)
        for row in self.families:
            if {(r['world'],r['camera']) for r in row['records']}!={(w,c) for w in range(2) for c in range(2)}:
                raise ValueError('An indexed family lost one of its four cells')
            if verify_files:
                for record in row['records']:
                    if file_sha256(self.root/record['path'])!=record['sha256']:
                        raise ValueError('Changed audited branch: '+record['path'])

    def __len__(self):
        return len(self.families)

    def __getitem__(self,index):
        row=self.families[index]
        return dict(metadata={k:v for k,v in row.items() if k!='records'},
                    branches={(r['world'],r['camera']):load_branch(self.root/r['path']) for r in row['records']})

    def branch(self,index,cell):
        """Read only one cell for ordinary paired-data exposure."""
        row=self.families[index]
        record=next(r for r in row['records'] if (r['world'],r['camera'])==cell)
        branch=load_branch(self.root/record['path'])
        branch['cohort']=row['cohort']
        return branch,{k:v for k,v in row.items() if k!='records'}


def build_family_index(root,output,development=False):
    from .audit import audit_manifest
    root=Path(root).resolve()
    records=[]
    for marker in sorted((root/'families').glob('*/complete.json')):
        manifest=json.loads((marker.parent/'manifest.json').read_text())
        for record in manifest['records']:
            path=(marker.parent/record['path']).resolve()
            records.append(dict(record,path=str(path.relative_to(root)),sha256=file_sha256(path)))
    if not records:
        raise ValueError('No accepted native families')
    merged=root/'manifest.json'
    merged.write_text(json.dumps(dict(records=records),indent=2)+'\n')
    audit=audit_manifest(merged)
    (root/'global_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    groups={}
    for record in records:
        family=record['family_id']
        groups.setdefault(family,dict(family_id=family,split=record['split'],cohort=record['cohort'],
            receiver=record['receiver'],target_index=record['target_index'],scene_parent_id=record['scene_parent_id'],records=[]))['records'].append(record)
    result=dict(schema='agcr-intervention-index-1',root=str(root),development_only=bool(development),
        audit_passed=audit['passed'],manifest_sha256=file_sha256(merged),families=list(groups.values()))
    result['split_sha256']=hashlib.sha256(json.dumps([(r['family_id'],r['scene_parent_id'],r['split']) for r in result['families']],sort_keys=True).encode()).hexdigest()
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--development',action='store_true');a=p.parse_args()
    result=build_family_index(a.root,a.output,a.development)
    print(json.dumps(dict(families=len(result['families']),audit_passed=result['audit_passed'])))
