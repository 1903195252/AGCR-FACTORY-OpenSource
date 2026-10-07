"""Explicit v4 measurement amendment; original training validators remain intact.

CPU preflight verifies real frozen source bytes, independent candidate lineage,
and the complete registered generated corpus before allocating image inference.
Original identities/checkpoints/results are never rewritten or promoted in place.
"""
import argparse
import json
import math
from pathlib import Path
import types
from unittest.mock import patch

import torch

from . import evaluation
from .downstream import frozen_digest
from experiments.formal_evaluation.common import (read_json,write_json,sha256,SAMPLE_SEEDS,
                                                  prediction_manifest,require_completed_corpus)
from experiments.downstream_tracking.inference import reader_implementation

SCHEMA='agcr-explicit-reader-amendment-v4-1'
ALLOWED={'downstream_tracking/'+name for name in ('inference.py','association.py','annotations.py','rgb_acceptance.py')}
VERSION='retain-2d-box-mask-and-exclude-floor-proxy-v4'


def immutable(path,value):
    path=Path(path)
    if path.exists() and read_json(path)!=value:raise ValueError('Changed immutable amendment contract: '+str(path))
    if not path.exists():write_json(path,value)


def relative_source(name):
    path=Path(name)
    if 'experiments' in path.parts:path=Path(*path.parts[path.parts.index('experiments')+1:])
    if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid recorded source path')
    return path


def frozen_source_evidence(expected,frozen_roots,current_root=None,current_sources=None):
    """Return hashes actually read from disk, not unverified checkpoint claims."""
    current_root=Path(current_root or Path(evaluation.__file__).resolve().parents[2])
    if not expected:raise ValueError('Original training source map is absent')
    if current_sources is not None and set(current_sources)!=set(expected):raise ValueError('Training source set changed')
    verified={};witness={};changes={}
    for name,digest in expected.items():
        rel=relative_source(name);current=current_root/'experiments'/rel
        if not current.is_file():raise ValueError('Missing current source: '+str(rel))
        actual=sha256(current)
        if actual!=digest and str(rel) not in ALLOWED:raise ValueError('Non-whitelisted scientific source changed: '+str(rel))
        candidates=[current]+[p for root in frozen_roots for p in (Path(root)/'experiments'/rel,Path(root)/rel,Path(root)/rel.name)]
        matching=next((p for p in candidates if p.is_file() and sha256(p)==digest),None)
        if matching is None:raise ValueError('No byte-verified frozen source witness: '+str(rel))
        verified[name]=sha256(matching);witness[name]=dict(path=str(matching.resolve()),sha256=verified[name])
        if actual!=digest:changes[str(rel)]=dict(original=digest,current=actual)
    return verified,dict(frozen_source_files=witness,allowed_inference_changes=changes)


def validate_parent(saved,index,protocol,protocol_sha,role,index_sha,frozen_roots):
    expected=saved.get('identity',{}).get('source_sha256',{})
    verified,evidence=frozen_source_evidence(expected,frozen_roots,current_sources=evaluation.source_hashes())
    # Same original validator bytecode and all original structural/quality tests.
    # Its isolated globals resolve source hashes from the byte-verified snapshot.
    # The public function/global source_hashes/default behavior are never patched.
    original=evaluation.validate_reader
    namespace=dict(original.__globals__,source_hashes=lambda:dict(verified))
    validator=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    identity=validator(saved,index,protocol,protocol_sha,role,index_sha)
    return identity,evidence


def inference_contract():
    from experiments.formal_evaluation.ranked_boxes import ranking_contract
    value=reader_implementation()
    if value['reader_postprocessing']!=VERSION:raise ValueError('Only the audited v4 inference policy is admitted')
    return dict(**value,amendment_source_sha256=sha256(__file__),ranking=ranking_contract(),
                ranked_capture_source_sha256=sha256(Path(__file__).with_name('reader_ranked_capture.py')),
                evaluation_sources=evaluation.evaluation_source_hashes())


def protected_output(output,run_root):
    output=Path(output).resolve();base=Path(run_root).resolve()/'amended_evaluation'/'reader_v4'
    if base not in output.parents:raise ValueError('Use a new run_root/amended_evaluation/reader_v4/... output')
    return output


def validate_candidate(a,parent,index):
    from experiments.formal_evaluation.reader_collection import verify_exclusion,digest
    from experiments.formal_evaluation.clean_reference import select_families
    candidate=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    folder=Path(a.checkpoint).parent;identity=candidate['identity'];manifest=read_json(folder/'manifest.json')
    completed=read_json(folder/'completion.json');parent_sha=sha256(a.parent)
    if (identity!=manifest or identity.get('schema')!='agcr-independent-reader-domain-repair-1' or
            identity.get('scope')!='domain_adaptation_candidate' or identity.get('role')!='video_evaluation' or
            identity.get('modality')!='rgb' or identity.get('parent_checkpoint_sha256')!=parent_sha or
            identity.get('parent_training_identity')!=parent['identity'] or
            not candidate.get('completed') or not candidate.get('run_completed') or
            candidate.get('training_completed_steps')!=128 or completed.get('completed') is not True or
            completed.get('step')!=128 or completed.get('selected_checkpoint_sha256')!=sha256(a.checkpoint) or
            completed.get('parent_checkpoint_sha256')!=parent_sha):
        raise ValueError('Candidate parent/manifest/completed training lineage failed')
    _,source_evidence=frozen_source_evidence(identity['source_sha256'],a.frozen_source_root)
    dedicated=read_json(a.dedicated_index);excluded=read_json(a.excluded_index);ordinary=read_json(a.ordinary_index)
    capture=Path(a.dedicated_index).parent;collection=read_json(capture/'reader_collection_manifest.json')
    audit=read_json(capture/'global_audit.json');old_audit=read_json(a.excluded_audit)
    exclusion=verify_exclusion(dedicated,excluded,audit,old_audit)
    capture_done=read_json(capture/'completion.json')
    if (collection['ordinary_index_sha256']!=sha256(a.ordinary_index) or collection['excluded_index_sha256']!=sha256(a.excluded_index) or
            collection['excluded_audit_sha256']!=sha256(a.excluded_audit) or capture_done.get('completed') is not True or
            capture_done.get('index_sha256')!=sha256(a.dedicated_index)):
        raise ValueError('Independent capture audit lineage differs')
    if (identity.get('paired_index_sha256')!=sha256(a.dedicated_index) or identity.get('ordinary_index_sha256')!=sha256(a.ordinary_index) or
            identity.get('validation_index_sha256')!=sha256(a.index) or identity.get('generator_weights_used') is not False or
            identity.get('input_labels_used') is not False or identity.get('test_used_for_selection') is not False or
            identity.get('threshold_selection_performed') is not False):
        raise ValueError('Candidate adaptation data/selection policy mismatch')
    # The audit files are evidence; independently verify all actual captured cells.
    for family in dedicated['families']:
        if family['split']!='train' or len(family['records'])!=4:raise ValueError('Four-cell dedicated TRAIN families required')
        for record in family['records']:
            if sha256(Path(dedicated['root'])/record['path'])!=record['sha256']:raise ValueError('Dedicated training pixels changed')
    expected=sorted(parent['identity']['training_episodes']+['dedicated_reader:'+f['family_id'] for f in dedicated['families']])
    if identity.get('training_episodes')!=expected or identity.get('training_partition_sha256')!=digest(expected):
        raise ValueError('Candidate independent training partition differs')
    panel=[f['family_id'] for f in select_families(index,'val',1)]
    reports=[]
    for step in (0,32,64,96,128):
        report=read_json(folder/('validation_%04d.json'%step))
        if (not math.isfinite(report.get('objective',float('nan'))) or report.get('complete') is not True or report.get('step')!=step or report.get('test_used') is not False or
                report.get('threshold_selection_performed') is not False or report.get('paired_families')!=panel or
                report.get('ordinary_uids')!=parent['identity']['validation_panel']['selected_uids']):
            raise ValueError('Complete fixed VAL selection reports required')
        reports.append(report)
    chosen=min(reports,key=lambda r:r['objective'])
    if (candidate.get('step')!=chosen['step'] or candidate.get('holdout_validation')!=chosen or
            candidate.get('frozen_frontend_sha256')!=frozen_digest(candidate['model'])):
        raise ValueError('Selected candidate weights do not match the minimum fixed validation objective')
    original=read_json(a.parent_association);association=read_json(a.association)
    for key,value in original.items():
        if key!='checkpoint_sha256' and association.get(key)!=value:raise ValueError('Frozen association value changed: '+key)
    if (original.get('checkpoint_sha256')!=parent_sha or association.get('checkpoint_sha256')!=sha256(a.checkpoint) or
            association.get('source_association_sha256')!=sha256(a.parent_association)):
        raise ValueError('Candidate association lineage differs')
    quality=read_json(a.quality_report);current=inference_contract()
    if (quality.get('complete') is not True or quality.get('split')!='val' or quality.get('checkpoint_sha256')!=sha256(a.checkpoint) or
            quality.get('association_sha256')!=sha256(a.association) or quality.get('index_sha256')!=sha256(a.index) or
            quality.get('threshold_selection_performed') is not False or quality.get('reference_used_for_scoring_only') is not True or
            any(quality.get(k)!=current[k] for k in ('reader_postprocessing','reader_source_sha256'))):
        raise ValueError('Matching v4 clean-reference VAL evidence is required, never TEST model selection')
    val_ids={f['family_id'] for f in index['families'] if f['split']=='val'}
    if not quality.get('family_ids') or not set(quality['family_ids'])<=val_ids:raise ValueError('Reader quality evidence must be VAL only')
    files=[folder/'manifest.json',folder/'completion.json',capture/'reader_collection_manifest.json',capture/'global_audit.json',
           Path(a.excluded_audit),Path(a.quality_report),*(folder/('validation_%04d.json'%s) for s in (0,32,64,96,128))]
    return candidate,dict(candidate_identity=identity,candidate_checkpoint_sha256=sha256(a.checkpoint),selected_step=chosen['step'],
        original_candidate_flags=dict(diagnostic_only=identity.get('diagnostic_only'),quality_accepted=candidate.get('quality_accepted')),
        independent_collection=exclusion,evidence_files={str(p.resolve()):sha256(p) for p in files},candidate_sources=source_evidence,
        selection='completed 128-update independent TRAIN adaptation, minimum fixed VAL objective; no TEST selection',
        quality_report_scope='documented matching VAL measurement check; no claim of generator improvement')


def complete_pixels(index_path,predictions):
    index=read_json(index_path);manifest=prediction_manifest(predictions);require_completed_corpus(predictions)
    families=[f for f in index['families'] if f['split']=='test']
    if manifest.get('split')!='test' or manifest.get('index_sha256')!=sha256(index_path):raise ValueError('Use the unchanged complete registered TEST corpus')
    if manifest.get('sample_seeds')!=list(SAMPLE_SEEDS):raise ValueError('All eight original draws required')
    checked=0
    for family in families:
        for draw,seed in enumerate(SAMPLE_SEEDS):
            marker=Path(predictions)/family['family_id']/('draw_%02d.json'%draw)
            row=read_json(marker)
            if row.get('seed')!=seed:raise ValueError('Draw seed mismatch')
            for record in family['records']:
                name='world%d_camera%d_sample%02d.npz'%(record['world'],record['camera'],draw)
                path=marker.parent/name
                if not path.is_file():raise ValueError('Full eight-draw source pixels missing; old readouts cannot replace v4 inference: '+str(path))
                if row['files'].get(name)!=sha256(path):raise ValueError('Changed lossless generated pixels')
                checked+=1
    return dict(families=len(families),draws=8,cells_per_family=4,verified_source_files=checked,
                generator_manifest_sha256=sha256(Path(predictions)/'manifest.json'))


def restoration_evidence(run_root,index,predictions):
    root=Path(run_root).resolve();predictions=Path(predictions).resolve()
    if predictions!=root/'regenerated_evaluation/S1/test':return None
    source=root/'predictions/S1/test';path=predictions/'regeneration_completion.json';record=read_json(path)
    if (record.get('restored') is not True or record.get('lossless_restored') is not True or
            record.get('expected_files')!=640 or record.get('matched_files')!=640 or
            record.get('manifest_sha256')!=sha256(predictions/'manifest.json') or
            record.get('source_manifest_sha256')!=sha256(source/'manifest.json') or
            record.get('retention_sha256')!=sha256(source/'retention.json') or
            sha256(predictions/'manifest.json')!=sha256(source/'manifest.json')):
        raise ValueError('Lossless retained S1 restoration evidence is incomplete or changed')
    count=0
    for family in index['families']:
        if family['split']!='test':continue
        for draw in range(8):
            name=family['family_id']+'/draw_%02d.json'%draw;old=read_json(source/name);new=read_json(predictions/name)
            if old.get('seed')!=new.get('seed') or old.get('files')!=new.get('files') or len(old.get('files',{}))!=4:
                raise ValueError('Restored S1 pixel checksums differ from immutable original draw markers')
            count+=4
    if count!=640:raise ValueError('Restored S1 needs the whole original 640-file corpus')
    return dict(scope='original_S1_pixels_losslessly_restored',files=count,evidence_sha256={str(p):sha256(p) for p in
        (path,source/'manifest.json',source/'retention.json',predictions/'manifest.json')})


def readout_contract(a):
    output=protected_output(a.output,a.run_root);protocol=read_json(a.protocol);ordinary=read_json(a.ordinary_index);index=read_json(a.index)
    parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    _,frozen=validate_parent(parent,ordinary,protocol,sha256(a.protocol),'video_evaluation',sha256(a.ordinary_index),a.frozen_source_root)
    _,candidate=validate_candidate(a,parent,index)
    population=complete_pixels(a.index,a.predictions)
    return dict(schema=SCHEMA,scope='full_eight_draw_video_readout',reason='measurement_reader_repaired_after_controlled_diagnostics',
        original_training_identity_preserved=True,original_formal_completion_modified=False,
        protocol_sha256=sha256(a.protocol),index_sha256=sha256(a.index),ordinary_index_sha256=sha256(a.ordinary_index),
        parent_checkpoint_sha256=sha256(a.parent),checkpoint_sha256=sha256(a.checkpoint),association_sha256=sha256(a.association),
        frozen_training_evidence=frozen,candidate_lineage=candidate,population=population,
        restoration=restoration_evidence(a.run_root,index,a.predictions),inference=inference_contract(),output=str(output))


def run_readout(a):
    root=protected_output(a.output,a.run_root)
    if a.action=='check':
        contract=readout_contract(a);immutable(root/'amendment.json',contract);return contract
    from experiments.formal_evaluation import readout
    from experiments.formal_evaluation import runtime
    from .reader_ranked_capture import RankedReadoutCapture
    from .train import BudgetLedger
    from .sample import exclusive
    ledger=BudgetLedger(a.budget_ledger,a.gpu_hours,1)
    done=False
    try:
        contract=readout_contract(a);immutable(root/'amendment.json',contract)
        with exclusive(root/'.readout.lock'):
            stop_file=getattr(a,'stop_file',None)
            if not ledger.exhausted() and not (stop_file and Path(stop_file).exists()):
                class CumulativeBudget(runtime.Budget):
                    def expired(self):
                        ledger.commit('amended_video_readout')
                        return super().expired() or ledger.exhausted() or bool(stop_file and Path(stop_file).exists())
                # Only the scheduling boundary is extended. All original reader
                # inference, thresholds, full population and metrics stay intact.
                capture=RankedReadoutCapture(a.index,a.predictions,a.output,a.checkpoint,a.association)
                with patch.object(runtime,'Budget',CumulativeBudget),patch.object(readout,'infer_arrays',capture.wrapper(readout.infer_arrays)):
                    done=readout.generator_readout(a.index,a.predictions,a.checkpoint,a.assets,a.association,a.output,a.device,
                                           gpu_hours=max(1e-9,a.gpu_hours-ledger.used))
                if done is None:done=(root/'completion.json').is_file() and read_json(root/'completion.json').get('completed') is True
                if done:write_json(root/'ranked_completion.json',capture.verify_complete())
            write_json(root/'amendment_completion.json',dict(schema=SCHEMA,completed=bool(done),amendment_sha256=sha256(root/'amendment.json'),
                full_eight_draws=bool(done),old_result_reused=False,formal_training_flags_modified=False))
    finally:ledger.close('completed' if done else 'stopped')
    return contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','readout'])
    for key in ('run-root','protocol','ordinary-index','index','predictions','parent','checkpoint','parent-association','association',
                'dedicated-index','excluded-index','excluded-audit','quality-report','assets','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--frozen-source-root',action='append',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--budget-ledger');p.add_argument('--gpu-hours',type=float,default=.25);p.add_argument('--stop-file')
    a=p.parse_args()
    if a.action=='readout' and not a.budget_ledger:raise ValueError('An independently counted runtime ledger is required')
    contract=run_readout(a);print(json.dumps(dict(schema=SCHEMA,scope=contract['scope'],population=contract['population'])))

if __name__=='__main__':main()
