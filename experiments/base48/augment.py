"""Ordinary fixed-window Base48 RGB-D proposals for existing downstream readers.

Uses original ordinary episode/split identities, with all eight draws retained.
A CPU merge checks the full preregistered pool before enabling annotation.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from .train import BudgetLedger, SAMPLE_SEEDS
from .async_writer import BoundedWriter
from .inference_cache import inference_condition_cache
from .sample import (exclusive, SamplingStopped, validate_checkpoints, install_checkpoints,
                     sampler_source_hashes, RGB_SHAPE, DEPTH_SHAPE, finite_decoder)
from experiments.paired_interventions.data import OrdinaryDataset,to_device,file_sha256
from experiments.paired_interventions.train import atomic_json
from experiments.generation_exports.windows import fixed_rows,generate_pair,save_npz


def registered_rows(dataset,protocol,seed):
    if protocol.get('schema')!='agcr-base48-protocol-1' or protocol.get('paired_sample_seeds')!=SAMPLE_SEEDS:
        raise ValueError('Registered Base48 protocol and eight draws required')
    count=protocol.get('sampling',{}).get('augmentation_count')
    if type(count) is not int or count<1:
        raise ValueError('Register sampling.augmentation_count before generating proposals')
    if dataset.index.get('development_only'):
        raise ValueError('Ordinary accepted training split required')
    rows=fixed_rows(dataset,count,seed)
    if len(rows)!=count or len({r['uid'] for r in rows})!=count:
        raise ValueError('Qualified independent fixed-window count differs from protocol')
    if any(dataset.index['roles'].get(r['episode'])!='train' or r.get('role','train')!='train' for r in rows):
        raise ValueError('Augmentation sources must retain ordinary training episode roles')
    return rows


def source_hashes():
    result=sampler_source_hashes();root=Path(__file__).resolve().parents[2]
    for name in ('experiments/base48/augment.py','experiments/generation_exports/windows.py'):
        result[name]=file_sha256(root/name)
    return result


def make_contract(paths,protocol_path,renderer_path,state_path,renderer,state,rows,index,shards):
    return dict(schema='agcr-generated-fixed-windows-1',role='train',
        checkpoint_sha256=file_sha256(renderer_path),renderer_checkpoint_sha256=file_sha256(renderer_path),
        state_checkpoint_sha256=file_sha256(state_path),renderer_identity=renderer['identity'],state_identity=state['identity'],
        protocol_sha256=file_sha256(protocol_path),ordinary_index_sha256=file_sha256(paths['ordinary']),
        split_sha256=index['split_sha256'],selected_uids=[r['uid'] for r in rows],sample_seeds=SAMPLE_SEEDS,
        training_seed=state['identity']['seed'],num_shards=shards,development_only=False,
        coordinate_frame='ground_anchor_local_m',depth_unit='mm',output_modality='rgbd',
        state_source='history_inferred_unfiltered_candidates',conditions='history_inferred_state_only',
        geometry_padding=dict(height=480,padded_width=736,crop_left=8,crop_right=8),
        source_sha256=source_hashes(),sampling_steps=50,inference_composition_only=True,
        auxiliary_training_reader_used=False,proposal_selection='fixed_rows_before_generation_no_quality_filter')


def folder_for(root,uid):return Path(root)/hashlib.sha256(uid.encode()).hexdigest()[:20]


def ensure_history(path, values):
    """A resumed window keeps the existing history file and its committed hash."""
    path=Path(path)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old)!=set(values) or any(not np.array_equal(old[k],v) for k,v in values.items()):
                raise ValueError('Cached annotation history differs from the fixed ordinary source')
    else:
        save_npz(path,values)


def validate_proposal(item,row,draw,seed,contract,inspect_arrays=True):
    if (item.get('conditioning_uid')!=row['uid'] or item.get('episode')!=row['episode']
        or item.get('uid')!=row['uid']+':sample%02d'%draw or item.get('sample_seed')!=seed
        or item.get('draw')!=draw or item.get('role')!='train' or item.get('development_only')
        or item.get('split_sha256')!=contract['split_sha256']):
        raise ValueError('Proposal changed ordinary episode/split/seed identity')
    for key in ('rgbd','history'):
        if file_sha256(item[key+'_path'])!=item[key+'_sha256']:
            raise ValueError('Changed generated proposal '+key)
    if inspect_arrays:
        with np.load(item['rgbd_path'],allow_pickle=False) as arrays:
            required={'rgb','depth_mm','validity_probability','initial_state','class_id','internal_state',
                'state_present','predicted_visible_layers','predicted_full_layers','predicted_surface_depth_mm','K','camera_to_world','time_s'}
            if required-set(arrays):raise ValueError('Missing independently checkable generated fields')
            if arrays['rgb'].shape!=RGB_SHAPE or arrays['rgb'].dtype!=np.uint8:raise ValueError('Wrong native RGB shape')
            for key in ('depth_mm','validity_probability'):
                if arrays[key].shape!=DEPTH_SHAPE or not np.isfinite(arrays[key]).all():raise ValueError('Invalid native RGB-D proposal')
            if (arrays['depth_mm']<0).any() or ((arrays['validity_probability']<0)|(arrays['validity_probability']>1)).any():
                raise ValueError('Invalid depth/validity range')
            n=len(arrays['class_id'])
            if arrays['internal_state'].shape!=(16,n,11) or arrays['state_present'].shape!=(16,n):
                raise ValueError('Generated candidate state slots lost alignment')
        with np.load(item['history_path'],allow_pickle=False) as arrays:
            if arrays['rgb'].shape[0:2]!=(6,2) or arrays['rgb'].shape[2:]!=RGB_SHAPE[2:]:
                raise ValueError('Annotation history must contain only six native paired frames')
            if arrays['time_s'].shape!=(6,) or (arrays['time_s']>0).any():
                raise ValueError('Future observation in augmentation history')
    return True


def publish_proposal(folder,row,draw,seed,arrays,metadata,contract):
    """Publish only after lossless storage, checksum and full proposal validation."""
    folder=Path(folder);output=folder/('sample_%02d.npz'%draw)
    save_npz(output,arrays)
    item=dict(metadata,conditioning_uid=row['uid'],uid=row['uid']+':sample%02d'%draw,
        proposal_id=folder.name+'_%02d'%draw,draw=draw,sample_seed=seed,
        role='train',split_sha256=contract['split_sha256'],rgbd_path=str(output),rgbd_sha256=file_sha256(output),
        history_path=str(folder/'history.npz'),history_sha256=file_sha256(folder/'history.npz'),
        development_only=False,frames=16,views=['air','ground'],includes_history=False,
        output_modality='rgbd',generator_method='base48_'+contract['state_identity']['variant'])
    validate_proposal(item,row,draw,seed,contract)
    atomic_json(folder/('sample_%02d.json'%draw),item)
    return item


def merge(root,paths_path,protocol_path):
    root=Path(root);paths=json.loads(Path(paths_path).read_text());protocol=json.loads(Path(protocol_path).read_text())
    with exclusive(root/'.merge.lock'):
        contract=json.loads((root/'generation_contract.json').read_text())
        if (contract['protocol_sha256']!=file_sha256(protocol_path)
            or contract['ordinary_index_sha256']!=file_sha256(paths['ordinary'])
            or contract['source_sha256']!=source_hashes() or contract['sample_seeds']!=SAMPLE_SEEDS):
            raise ValueError('Augmentation contract changed')
        ds=OrdinaryDataset(paths['ordinary'],'train');rows=registered_rows(ds,protocol,contract['training_seed'])
        if contract['selected_uids']!=[r['uid'] for r in rows] or contract['split_sha256']!=ds.index['split_sha256']:
            raise ValueError('Augmentation selection or split changed')
        locks=[];digest=file_sha256(root/'generation_contract.json')
        try:
            for shard in range(contract['num_shards']):
                context=exclusive(root/'shards'/('shard_%04d.lock'%shard));context.__enter__();locks.append(context)
                marker=root/'shards'/('shard_%04d'%shard)/'completion.json'
                row=json.loads(marker.read_text())
                if (not row.get('completed') or row.get('whole_corpus_completed') is not False
                    or row.get('contract_sha256')!=digest or row.get('selected_uids')!=contract['selected_uids'][shard::contract['num_shards']]):
                    raise ValueError('An augmentation shard is incomplete')
            clips=[]
            for row in rows:
                for draw,seed in enumerate(SAMPLE_SEEDS):
                    item=json.loads((folder_for(root,row['uid'])/('sample_%02d.json'%draw)).read_text())
                    validate_proposal(item,row,draw,seed,contract);clips.append(item)
            document=dict(contract,clips=clips,completed=True,downstream_augmentation=True,
                          proposal_count=len(clips),acceptance_status='pending_independent_annotation')
            atomic_json(root/'manifest.json',document);atomic_json(root/'windows.json',document)
            complete=dict(completed=True,whole_corpus_completed=True,proposals=len(clips),
                expected=len(rows)*8,development_only=False,contract_sha256=digest,
                independently_accepted=False,updated_unix=time.time())
            atomic_json(root/'completion.json',complete);return complete
        except BaseException as error:
            atomic_json(root/'completion.json',dict(completed=False,reason='cpu_merge_validation_failed',error_type=type(error).__name__))
            raise
        finally:
            for context in reversed(locks):context.__exit__(None,None,None)


def generate(a):
    paths=json.loads(Path(a.paths).read_text());protocol=json.loads(Path(a.protocol).read_text())
    if a.num_shards<1 or not 0<=a.shard_index<a.num_shards or a.gpu_hours<=0:raise ValueError('Invalid shard/budget')
    if any('agcr-intervention-runs' in Path(x).resolve().parts for x in (a.output,a.budget_ledger)):
        raise ValueError('Pro runs and budgets are read-only')
    renderer=torch.load(a.renderer_checkpoint,map_location='cpu',weights_only=False)
    state=torch.load(a.state_checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoints(renderer,state,paths,a.protocol)
    ds=OrdinaryDataset(paths['ordinary'],'train');rows=registered_rows(ds,protocol,state['identity']['seed']);ds.dataset.rows=rows
    if a.num_shards>len(rows):raise ValueError('Every shard needs a whole fixed window')
    contract=make_contract(paths,a.protocol,a.renderer_checkpoint,a.state_checkpoint,renderer,state,rows,ds.index,a.num_shards)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'.contract.lock',blocking=True):
        path=root/'generation_contract.json'
        if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Cannot mix proposal contracts')
        if not path.exists():atomic_json(path,contract)
    digest=file_sha256(path);owned=list(range(a.shard_index,len(rows),a.num_shards))
    shardroot=root/'shards'/('shard_%04d'%a.shard_index);shardroot.mkdir(parents=True,exist_ok=True)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    with exclusive(root/'shards'/('shard_%04d.lock'%a.shard_index)):
        torch.set_num_threads(a.threads);device=torch.device(a.device);torch.cuda.set_device(device)
        ledger=BudgetLedger(a.budget_ledger,a.gpu_hours,1);complete=False;proposals=set();hook=None
        try:
            from .model import Base48System
            model=Base48System(paths,device,'video',renderer['identity']['variant_options'])
            install_checkpoints(model,renderer,state);del renderer,state
            def boundary(*_):
                if stop[0] or ledger.exhausted() or (a.stop_file and Path(a.stop_file).exists()):raise SamplingStopped('budget_or_stop')
                if shutil.disk_usage(root).free<a.min_free_gib*2**30:raise SamplingStopped('disk_floor')
            # generate_pair calls this native Wan module exactly once per Euler step.
            hook=model.wan.register_forward_pre_hook(boundary)
            def publish_complete(folder,row,draw,seed,arrays,metadata):
                publish_proposal(folder,row,draw,seed,arrays,metadata,contract)
                proposals.add((row['uid'],draw))
                status=dict(completed_proposals=len(proposals),expected_proposals=len(owned)*8,
                            whole_corpus_completed=False,uid=row['uid'],draw=draw)
                atomic_json(shardroot/'status.json',status);ledger.commit('augmentation_sampling',**status)
                print(json.dumps(status),flush=True)
            with BoundedWriter() as writer:
                for i in owned:
                    row=rows[i];folder=folder_for(root,row['uid']);inputs=None;metadata=None
                    for draw,seed in enumerate(SAMPLE_SEEDS):
                        marker=folder/('sample_%02d.json'%draw)
                        if marker.exists():
                            validate_proposal(json.loads(marker.read_text()),row,draw,seed,contract,False)
                            proposals.add((row['uid'],draw));continue
                        writer.wait_for_slot()
                        boundary()
                        if inputs is None:
                            record=ds[i];metadata=record['metadata'];inputs=to_device(record['inputs'],device)
                            history={k:v.cpu().numpy() for k,v in inputs.items() if k not in ('planned_camera_to_world','future_time_s')}
                            history['valid']=history.pop('depth_valid');ensure_history(folder/'history.npz',history)
                            del history
                        with torch.no_grad(), finite_decoder(model), inference_condition_cache(model.wan):
                            arrays=generate_pair(model,inputs,row['uid'],seed,contract['training_seed'],device)
                        writer.submit(publish_complete,folder,row,draw,seed,arrays,metadata)
                        del arrays
            complete=True
        except SamplingStopped:pass
        except BaseException as error:
            atomic_json(shardroot/'failure.json',dict(type=type(error).__name__,message=str(error),completed=False));raise
        finally:
            if hook:hook.remove()
            atomic_json(shardroot/'completion.json',dict(completed=complete,whole_corpus_completed=False,
                contract_sha256=digest,selected_uids=[rows[i]['uid'] for i in owned],completed_proposals=len(proposals),
                expected_proposals=len(owned)*8,reason='requires_cpu_merge' if complete else 'bounded_or_failed'))
            ledger.close('shard_completed' if complete else 'stopped',output=str(root),shard=a.shard_index)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    g=sub.add_parser('generate')
    for key in ('paths','protocol','renderer-checkpoint','state-checkpoint','output','budget-ledger'):g.add_argument('--'+key,required=True)
    g.add_argument('--gpu-hours',type=float,required=True);g.add_argument('--shard-index',type=int,default=0);g.add_argument('--num-shards',type=int,default=1)
    g.add_argument('--device',default='cuda:0');g.add_argument('--threads',type=int,default=8);g.add_argument('--stop-file')
    g.add_argument('--min-free-gib',type=float,default=25)
    m=sub.add_parser('merge')
    for key in ('output','paths','protocol'):m.add_argument('--'+key,required=True)
    return p.parse_args(argv)


def main(argv=None):
    a=arguments(argv)
    if a.action=='generate':generate(a)
    else:print(json.dumps(merge(a.output,a.paths,a.protocol),indent=2))


if __name__=='__main__':main()
