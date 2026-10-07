"""Complete Base48 RGB-D corpora from a common renderer and learned state model.

Generate each eight-seed/four-cell family on one shard. Only CPU merge may mark
an entire corpus complete. References never construct a generation condition.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .train import BudgetLedger, SAMPLE_SEEDS, SCHEMA, source_hashes, audit_index_ancestry
from .async_writer import BoundedWriter
from .inference_cache import inference_condition_cache
from experiments.paired_interventions.data import FamilyDataset, to_device, file_sha256, historical_labels
from experiments.paired_interventions.sample import stream_seed
from experiments.paired_interventions.sampling import pure_noise_euler
from experiments.paired_interventions.train import atomic_json, import_weights
from experiments.paired_interventions.world import predict_world, world_conditions, query_labels
from experiments.formal_evaluation.state_likelihood import unique_history_assignment, HISTORY_SELECTION_POLICY

RGB_SHAPE = (16, 2, 480, 720, 3)
DEPTH_SHAPE = RGB_SHAPE[:-1]
CELLS = tuple((w, c) for w in range(2) for c in range(2))
HISTORY_LABELS = ('history_state', 'class_id', 'history_visible_fraction', 'history_projection_area')


class SamplingStopped(RuntimeError):
    pass


class _ReleasedTrainingReader(torch.nn.Module):
    """Parameter-free lifecycle slot for Base48System.train/eval."""
    def forward(self, *args, **kwargs):
        raise RuntimeError('The auxiliary training reader is unavailable during generation')


def checked_decode(model, latent, condition):
    decoded = model.decode(latent, condition)
    for key in ('rgb', 'depth_mm', 'validity_logits'):
        if key not in decoded or not torch.isfinite(decoded[key]).all():
            raise FloatingPointError('Nonfinite native decoder output before quantization: ' + key)
    return decoded


@contextmanager
def finite_decoder(model):
    """Guard the reused native exporter before its float-to-byte conversion."""
    original = model.decode
    had_local = 'decode' in model.__dict__
    local = model.__dict__.get('decode')
    def guarded(*args, **kwargs):
        decoded = original(*args, **kwargs)
        for key in ('rgb', 'depth_mm', 'validity_logits'):
            if key not in decoded or not torch.isfinite(decoded[key]).all():
                raise FloatingPointError('Nonfinite native decoder output before quantization: ' + key)
        return decoded
    model.decode = guarded
    try:
        yield
    finally:
        if had_local:
            model.decode = local
        else:
            delattr(model, 'decode')


@contextmanager
def exclusive(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except OSError:
            raise RuntimeError('Another job owns '+str(path))
        yield


def selected_families(index, protocol, role):
    if protocol.get('schema') != 'agcr-base48-protocol-1':
        raise ValueError('Registered Base48 protocol required')
    if protocol.get('paired_sample_seeds') != SAMPLE_SEEDS:
        raise ValueError('All eight registered seeds are required')
    if protocol.get('native', {}).get('sampling_steps', 50) != 50:
        raise ValueError('Native generation requires 50 denoising steps')
    by_id = {row['family_id']: row for row in index['families']}
    if len(by_id) != len(index['families']): raise ValueError('Duplicate indexed family')
    if role == 'test':
        ids = [row['family_id'] for row in index['families'] if row['split'] == 'test']
    else:
        raise ValueError('Unknown corpus role')
    if not ids: raise ValueError('Empty registered sampling population')
    return list(ids)


def validate_checkpoints(renderer, state, paths, protocol_path):
    for payload, phase in ((renderer, 'video'), (state, 'state')):
        identity = payload.get('identity', {})
        if (identity.get('schema') != SCHEMA or identity.get('phase') != phase
                or identity.get('development_only') or not payload.get('completed')
                or not payload.get('validation_best')):
            raise ValueError('Sampling requires completed and validated Base48 '+phase+' weights')
        if identity.get('protocol_sha256') != file_sha256(protocol_path):
            raise ValueError('Sampling protocol differs from training')
        if identity.get('source_sha256') != source_hashes():
            raise ValueError('Base48 model/training source changed after checkpointing')
        for name, key in [('families','family_index_sha256'), ('ordinary','ordinary_index_sha256'),
                          ('upstream','parent_sha256'), ('statistics','statistics_sha256'),
                          ('empty_text','empty_text_sha256')]:
            if identity.get(key) != file_sha256(paths[name]):
                raise ValueError('Sampling provenance differs: '+name)
    if state['identity'].get('variant_options', {}).get('receiver_only'):
        raise ValueError('Receiver-only S0 is a state diagnostic, not a paired-video renderer condition')
    protocol = json.loads(Path(protocol_path).read_text())
    seed = protocol.get('evaluation', {}).get('video_training_seed')
    if seed is not None and state['identity']['seed'] != seed:
        raise ValueError('Video state-adaptation seed differs from preregistration')
    return True


def install_checkpoints(model, renderer, state):
    """An explicit inference composition; never rewrites either checkpoint."""
    required = {key for key in model.state_dict() if not key.startswith('wan.backbone.') or 'lora_' in key}
    missing = required - set(renderer['model'])
    if missing: raise ValueError('Incomplete renderer weights: '+str(sorted(missing)[:8]))
    import_weights(model, renderer['model'])
    upstream = {key[len('upstream.'):]: value for key, value in state['model'].items() if key.startswith('upstream.')}
    model.upstream.load_state_dict(upstream, strict=True)
    # Release the auxiliary training reader's parameters, but retain its module
    # slot because Base48System.train/eval visits it on every mode transition.
    # The upstream observation reader remains essential and is retained.
    if hasattr(model, 'reader'): model.reader = _ReleasedTrainingReader()
    model.eval().requires_grad_(False)


def inference_family(family):
    """Do not even transfer future reference tensors onto the sampling GPU."""
    return dict(metadata=family['metadata'], branches={cell:dict(inputs=branch['inputs'],
        supervision={name:branch['supervision'][name] for name in HISTORY_LABELS})
        for cell, branch in family['branches'].items()})


def history_agreement(family):
    for w in range(2):
        left, right = (family['branches'][w,c]['inputs'] for c in range(2))
        for key in ('rgb','depth_mm','depth_valid','K','camera_to_world','time_s','future_time_s'):
            if not torch.equal(left[key], right[key]):
                raise ValueError('Passive camera branches changed historical information: '+key)


@torch.no_grad()
def generate_family(model, family, sample_seed, training_seed, device, boundary=lambda:None,
                    publish_cell=None, before_cell=lambda:None):
    """One history-inferred state draw per world, reused by both camera paths."""
    history_agreement(family)
    name = family['metadata']['family_id']; outputs = {}; diagnostics = {}
    generator = lambda tag: torch.Generator(device=device).manual_seed(stream_seed(training_seed,name,sample_seed,tag))
    for world in range(2):
        boundary()
        branch = family['branches'][world,0]; inputs = branch['inputs']
        encoded = model.encoders.history(inputs['rgb'])
        history, draw = predict_world(model.upstream, inputs, encoded, generator('state'))
        # Historical references label the already predicted slots for scoring;
        # they do not select, remove or modify any slot used by the renderer.
        labels = historical_labels(branch)
        raw_ids = query_labels(history, labels)
        eligible = labels['history_detectable'].any(0).any(0)
        assignment = unique_history_assignment(history, raw_ids, eligible)
        diagnostics[str(world)] = assignment['stats']
        diagnostics[str(world)].update(
            duplicate_present_slot_steps=int(draw['present'][0,:,assignment['duplicate_mask']].sum()),
            unassociated_present_slot_steps=int(draw['present'][0,:,assignment['unassociated_mask']].sum()))
        anchor = encoded[1][-1,:,None]
        if tuple(anchor.shape) != (2,1,48,30,46):
            raise ValueError('Unexpected native 720x480 anchor latent')
        noises = torch.stack([torch.randn(4,48,30,46,device=device,generator=generator(tag)) for tag in ('air','ground')])
        for camera in range(2):
            before_cell()
            boundary()
            current = family['branches'][world,camera]
            condition = world_conditions(model.upstream, history, draw, current['inputs'])
            def velocity(z, s, conditions):
                boundary()  # before every one of the 50 native denoising steps
                return model.wan(z.permute(0,2,1,3,4),s,conditions).permute(0,2,1,3,4)
            with inference_condition_cache(model.wan):
                latent = pure_noise_euler(velocity,noises,anchor,condition,50,2).permute(0,2,1,3,4)
            boundary()
            decoded = checked_decode(model,latent,condition)
            tracks = history['result']['tracks']
            result = dict(
                rgb=((decoded['rgb'].permute(0,1,3,4,2).clamp(-1,1)+1)*127.5).round().byte(),
                depth_mm=decoded['depth_mm'], validity_probability=decoded['validity_logits'].sigmoid(),
                internal_state=draw['state'][0], state_present=draw['present'][0],
                historical_reference_ids=assignment['ids'], historical_reference_ids_raw=raw_ids,
                historical_selected_slot_mask=assignment['selected_mask'],
                historical_duplicate_slot_mask=assignment['duplicate_mask'],
                historical_unassociated_slot_mask=assignment['unassociated_mask'],
                historical_unmatched_slot_mask=assignment['unmatched_mask'],
                history_eligible_reference=eligible,
                predicted_visible_layers=condition['M'], predicted_full_layers=condition['P'],
                predicted_surface_depth_mm=condition['Z'], initial_state=history['result']['initial_state'],
                class_id=torch.stack([t['class_prob'][1:].argmax()+1 for t in tracks]) if tracks else raw_ids.new_empty(0),
                time_s=current['inputs']['future_time_s'], K=current['inputs']['K'],
                camera_to_world=current['inputs']['planned_camera_to_world'])
            arrays = {key:value.cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
                      for key,value in result.items()}
            outputs[world,camera] = publish_cell(world,camera,arrays) if publish_cell else arrays
            del arrays, result, decoded
    return outputs, diagnostics


def file_name(world, camera, draw):
    return 'world%d_camera%d_sample%02d.npz' % (world,camera,draw)


def validate_arrays(arrays):
    required = {'rgb','depth_mm','validity_probability','internal_state','state_present',
                'historical_reference_ids','historical_reference_ids_raw',
                'historical_duplicate_slot_mask','historical_unassociated_slot_mask',
                'predicted_visible_layers','predicted_full_layers','predicted_surface_depth_mm',
                'initial_state','class_id','time_s','K','camera_to_world'}
    if required - set(arrays): raise ValueError('Missing generated fields: '+str(sorted(required-set(arrays))))
    if arrays['rgb'].shape != RGB_SHAPE or arrays['rgb'].dtype != np.uint8:
        raise ValueError('Native paired RGB export must be uint8, 16x2x480x720x3')
    for key in ('depth_mm','validity_probability'):
        if arrays[key].shape != DEPTH_SHAPE or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid paired generated '+key)
    if (arrays['depth_mm']<0).any(): raise ValueError('Negative generated metric depth')
    if ((arrays['validity_probability']<0)|(arrays['validity_probability']>1)).any():
        raise ValueError('Invalid depth validity probability')
    ids = np.asarray(arrays['historical_reference_ids'])
    raw = np.asarray(arrays['historical_reference_ids_raw'])
    state, present = arrays['internal_state'], arrays['state_present']
    if ids.ndim!=1 or raw.shape!=ids.shape or state.shape!=(16,len(ids),11) or present.shape!=(16,len(ids)):
        raise ValueError('Generated slots lost their original shape')
    selected = ids[ids>=0]
    if len(selected)!=len(np.unique(selected)) or (ids < -1).any():
        raise ValueError('Historical scoring representatives are not one-to-one')
    if arrays['K'].shape!=(3,3) or arrays['camera_to_world'].shape!=(16,2,4,4) or arrays['time_s'].shape!=(16,):
        raise ValueError('Missing native camera/time calibration')
    return True


def publish_cell_file(folder, draw, world, camera, arrays):
    validate_arrays(arrays)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/file_name(world,camera,draw);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream: np.savez_compressed(stream,**arrays)
    os.replace(temp,path)
    return path.name,file_sha256(path)


def commit_draw_futures(folder,draw,seed,futures,metadata,diagnostics):
    if set(futures)!=set(CELLS):raise ValueError('All four cell writes must finish before committing a draw')
    files=dict(futures[cell].result() for cell in CELLS)
    atomic_json(Path(folder)/('draw_%02d.json'%draw),dict(seed=seed,files=files,metadata=metadata,
                                                       history_assignment=diagnostics))


def publish_draw(folder, draw, seed, outputs, metadata, diagnostics):
    if set(outputs) != set(CELLS): raise ValueError('A committed draw must contain all four cells')
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    files = {}
    # Validate all cells before publishing any committed marker.
    for arrays in outputs.values(): validate_arrays(arrays)
    for (w,c), arrays in outputs.items():
        name,digest=publish_cell_file(folder,draw,w,c,arrays);files[name]=digest
    atomic_json(folder/('draw_%02d.json'%draw),dict(seed=seed,files=files,metadata=metadata,
                                                history_assignment=diagnostics))


def validate_draw(folder, draw, seed, inspect_arrays=True):
    folder=Path(folder);marker=folder/('draw_%02d.json'%draw)
    if not marker.exists(): raise ValueError('Missing complete draw marker: '+str(marker))
    row=json.loads(marker.read_text());expected={file_name(w,c,draw) for w,c in CELLS}
    if row.get('seed')!=seed or set(row.get('files',{}))!=expected:
        raise ValueError('A draw must retain its seed and all four unique cells')
    for name,digest in row['files'].items():
        path=folder/name
        if not path.is_file() or file_sha256(path)!=digest:
            raise ValueError('Altered or missing generated cell: '+str(path))
        if inspect_arrays:
            with np.load(path,allow_pickle=False) as arrays: validate_arrays(arrays)
    return row


def sampler_source_hashes():
    root=Path(__file__).resolve().parents[2]
    sources=source_hashes()
    sources['experiments/base48/inference_cache.py']=file_sha256(Path(__file__).with_name('inference_cache.py'))
    for name in ('experiments/base48/sample.py','experiments/base48/async_writer.py','experiments/formal_evaluation/state_likelihood.py',
                 'experiments/formal_evaluation/common.py'):
        sources[name]=file_sha256(root/name)
    return sources


def make_manifest(paths, protocol_path, renderer_path, state_path, renderer, state,
                  role, family_ids, num_shards):
    return dict(schema='agcr-intervention-predictions-1', method='base48_'+state['identity']['variant'],
        split='test' if role=='test' else 'train', corpus_role=role, selected_family_ids=family_ids,
        sample_seeds=SAMPLE_SEEDS, checkpoint_sha256=file_sha256(renderer_path),
        renderer_checkpoint_sha256=file_sha256(renderer_path), state_checkpoint_sha256=file_sha256(state_path),
        renderer_identity=renderer['identity'], state_identity=state['identity'],
        training_seed=state['identity']['seed'], renderer_training_seed=renderer['identity']['seed'],
        index_sha256=file_sha256(paths['families']), protocol_sha256=file_sha256(protocol_path),
        source_sha256=sampler_source_hashes(), development_only=False, conditions='history_inferred_state_only',
        future_reference_conditioning=False, reader_scope='none_independent_reader_required',
        independent_visual_metrics=False, output_modality='rgbd', conditioning_modality='rgbd',
        depth_unit='mm_optical_axis_Z', primary_aggregation='mean_of_8', sampling_steps=50,
        same_world_state_draw_across_camera_paths=True, historical_selection_policy=HISTORY_SELECTION_POLICY,
        unmatched_slots_retained_in_generation=True, auxiliary_training_reader_used=False,
        num_shards=num_shards, perturbation={'kind':'clean'}, inference_composition_only=True)


def register_manifest(root, manifest):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'.manifest.lock',blocking=True):
        path=root/'manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:
            raise ValueError('Cannot mix sampler/corpus provenance')
        if not path.exists(): atomic_json(path,manifest)
    return file_sha256(path)


def merge(root, protocol_path, index_path):
    """CPU-only complete-corpus verification; no shard can claim this result."""
    root=Path(root)
    with exclusive(root/'.merge.lock'):
        manifest=json.loads((root/'manifest.json').read_text())
        protocol=json.loads(Path(protocol_path).read_text());index=json.loads(Path(index_path).read_text())
        if (manifest.get('index_sha256')!=file_sha256(index_path)
                or manifest.get('protocol_sha256')!=file_sha256(protocol_path)
                or manifest.get('sample_seeds')!=SAMPLE_SEEDS):
            raise ValueError('Merge index/protocol/seeds differ from sampling')
        family_ids=selected_families(index,protocol,manifest['corpus_role'])
        if manifest.get('selected_family_ids')!=family_ids: raise ValueError('Changed registered population')
        if manifest.get('source_sha256')!=sampler_source_hashes(): raise ValueError('Sampler implementation changed before merge')
        count=manifest['num_shards'];manifest_sha=file_sha256(root/'manifest.json')
        with_locks=[]
        try:
            for shard in range(count):
                context=exclusive(root/'shards'/('shard_%04d.lock'%shard));context.__enter__();with_locks.append(context)
                marker=root/'shards'/('shard_%04d'%shard)/'completion.json'
                if not marker.exists(): raise ValueError('A sampling shard has not completed')
                row=json.loads(marker.read_text());expected=family_ids[shard::count]
                if (row.get('completed') is not True or row.get('whole_corpus_completed') is not False
                        or row.get('manifest_sha256')!=manifest_sha or row.get('family_ids')!=expected):
                    raise ValueError('Incomplete or inconsistent whole-family shard')
            completed=0
            for family in family_ids:
                for draw,seed in enumerate(SAMPLE_SEEDS):
                    validate_draw(root/family,draw,seed,inspect_arrays=True);completed+=1
            result=dict(completed=True,development_only=False,whole_corpus_completed=True,
                manifest_sha256=manifest_sha,families=len(family_ids),draws_per_history=8,
                completed_family_draws=completed,generated_cells=completed*4,
                independent_visual_metrics=False,validated_cpu=True,updated_unix=time.time())
            atomic_json(root/'completion.json',result)
            return result
        except BaseException as error:
            atomic_json(root/'completion.json',dict(completed=False,whole_corpus_completed=False,
                reason='cpu_merge_validation_failed',error_type=type(error).__name__,updated_unix=time.time()))
            raise
        finally:
            for context in reversed(with_locks): context.__exit__(None,None,None)


def sample(a):
    paths=json.loads(Path(a.paths).read_text());protocol=json.loads(Path(a.protocol).read_text())
    index=json.loads(Path(paths['families']).read_text())
    if a.num_shards<1 or not 0<=a.shard_index<a.num_shards or a.gpu_hours<=0:
        raise ValueError('Invalid whole-family shard or GPU budget')
    if index.get('development_only') or not index.get('audit_passed'):
        raise ValueError('Sampling requires an audited formal index')
    audit_index_ancestry(paths['families'])
    family_ids=selected_families(index,protocol,a.role)
    if a.num_shards>len(family_ids): raise ValueError('Every shard must own at least one complete family')
    if any('agcr-intervention-runs' in Path(p).resolve().parts for p in (a.output,a.budget_ledger)):
        raise ValueError('Pro outputs and budget are preserved read-only')
    renderer=torch.load(a.renderer_checkpoint,map_location='cpu',weights_only=False)
    state=torch.load(a.state_checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoints(renderer,state,paths,a.protocol)
    manifest=make_manifest(paths,a.protocol,a.renderer_checkpoint,a.state_checkpoint,renderer,state,
                           a.role,family_ids,a.num_shards)
    root=Path(a.output).resolve();manifest_sha=register_manifest(root,manifest)
    shard_root=root/'shards'/('shard_%04d'%a.shard_index);shard_root.mkdir(parents=True,exist_ok=True)
    owned=family_ids[a.shard_index::a.num_shards]
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    with exclusive(root/'shards'/('shard_%04d.lock'%a.shard_index)):
        dataset=FamilyDataset(paths['families'],'test' if a.role=='test' else 'train')
        positions={row['family_id']:i for i,row in enumerate(dataset.families)}
        torch.set_num_threads(a.threads);device=torch.device(a.device);torch.cuda.set_device(device)
        ledger=BudgetLedger(a.budget_ledger,a.gpu_hours,1)
        complete=False;completed=set()
        try:
            from .model import Base48System
            model=Base48System(paths,device,'video',renderer['identity']['variant_options'])
            install_checkpoints(model,renderer,state)
            del renderer,state
            def boundary():
                if stop[0] or ledger.exhausted() or (a.stop_file and Path(a.stop_file).exists()):
                    raise SamplingStopped('budget_or_stop_request')
            def publish_complete(family_id,draw,seed,values,metadata,diagnostics):
                commit_draw_futures(root/family_id,draw,seed,values,metadata,diagnostics)
                completed.add((family_id,draw))
                status=dict(completed_family_draws=len(completed),total_family_draws=len(owned)*8,
                            family=family_id,draw=draw,whole_corpus_completed=False,updated_unix=time.time())
                atomic_json(shard_root/'status.json',status)
                ledger.commit('sampling',shard=a.shard_index,family=family_id,draw=draw)
                print(json.dumps(status),flush=True)
            with BoundedWriter() as writer:
                for family_id in owned:
                    family=None
                    for draw,seed in enumerate(SAMPLE_SEEDS):
                        marker=root/family_id/('draw_%02d.json'%draw)
                        if marker.exists():
                            validate_draw(root/family_id,draw,seed,inspect_arrays=False)
                            completed.add((family_id,draw));continue
                        boundary()
                        if family is None: family=to_device(inference_family(dataset[positions[family_id]]),device)
                        def cell_writer(w,c,arrays):
                            return writer.submit(publish_cell_file,root/family_id,draw,w,c,arrays)
                        values,diagnostics=generate_family(model,family,seed,manifest['training_seed'],device,boundary,
                                                          cell_writer,writer.wait_for_slot)
                        writer.submit(publish_complete,family_id,draw,seed,values,family['metadata'],diagnostics)
                    del family
            complete=True
        except SamplingStopped:
            pass
        except BaseException as error:
            atomic_json(shard_root/'failure.json',dict(type=type(error).__name__,message=str(error),completed=False))
            raise
        finally:
            atomic_json(shard_root/'completion.json',dict(completed=complete,whole_corpus_completed=False,
                manifest_sha256=manifest_sha,family_ids=owned,completed_family_draws=len(completed),
                expected_family_draws=len(owned)*8,shard_index=a.shard_index,num_shards=a.num_shards,
                reason='shard_finished_requires_cpu_merge' if complete else 'bounded_or_failed',updated_unix=time.time()))
            ledger.close('shard_completed' if complete else 'stopped',shard=a.shard_index,output=str(root))


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    generate=sub.add_parser('generate')
    for key in ('paths','protocol','renderer-checkpoint','state-checkpoint','output','budget-ledger'):
        generate.add_argument('--'+key,required=True)
    generate.add_argument('--gpu-hours',type=float,required=True)
    generate.add_argument('--role',choices=['test'],default='test')
    generate.add_argument('--shard-index',type=int,default=0);generate.add_argument('--num-shards',type=int,default=1)
    generate.add_argument('--device',default='cuda:0');generate.add_argument('--threads',type=int,default=8)
    generate.add_argument('--stop-file')
    combine=sub.add_parser('merge')
    for key in ('output','protocol','index'): combine.add_argument('--'+key,required=True)
    return p.parse_args(argv)


def main(argv=None):
    a=arguments(argv)
    if a.action=='merge': print(json.dumps(merge(a.output,a.protocol,a.index),indent=2))
    else: sample(a)


if __name__=='__main__':main()
