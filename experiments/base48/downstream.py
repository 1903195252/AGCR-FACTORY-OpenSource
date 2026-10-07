"""Base48 identity-recovery experiments with a common frozen visual frontend.

Commands: cohort, frontend, cache, train. This is an independent short protocol,
not a completion shortcut for the Pro 40k detector-training schedule. Exported
best.pt files retain CooperativeDetector's state_dict and can be consumed by
downstream_tracking.calibrate and formal_evaluation.downstream unchanged.

The annotation and video_evaluation frontends train on their respective native
partitions once. The downstream frontend is shared across all data regimes.
Only the identity head changes between R/G/RG/RG1/Rnew. "G" refers to identity
training data, not a detector that has never seen rendered training data.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.downstream_tracking.data import (
    ClipDataset, ROLES, make_targets, move, native_clip, pack_images, sha256,
)
from experiments.downstream_tracking.model import (
    CooperativeDetector, box_iou, head_losses, matching, xyxy,
)
from experiments.downstream_tracking.train import (
    atomic, batch_indices, rng, restore_rng,
)
from experiments.paper_experiments.agcr_exp.downstream import make_cohort
from experiments.paper_experiments.agcr_exp.protocol import digest
from experiments.paper_experiments.agcr_exp.selection import (
    rendered_candidates, validation_panel, validation_mean,
)
from .train import BudgetLedger, assert_base_paths, atomic_checkpoint


SCHEMA = 'agcr-base48-independent-tracker-1'
CACHE_SCHEMA = 'agcr-base48-query-cache-1'
REGIMES = ('R', 'G', 'RG', 'RG1', 'Rnew')
IMAGE_KEYS = {'images', 'depth_mm', 'valid', 'K'}
FRONTEND_MODALITIES = {'annotation': 'rgbd', 'downstream': 'rgbd', 'video_evaluation': 'rgb'}


def frontend_modality(role):
    """A common RGB reader compares RGB-only and RGB-D generators fairly."""
    if role not in FRONTEND_MODALITIES:
        raise ValueError('Unknown independent frontend role')
    return FRONTEND_MODALITIES[role]


def settings(protocol, stage=None):
    if protocol.get('schema') != 'agcr-base48-protocol-1':
        raise ValueError('Separate registered Base48 protocol required')
    cfg = dict(protocol.get('downstream', {}))
    if cfg.get('frontend_modalities', FRONTEND_MODALITIES) != FRONTEND_MODALITIES:
        raise ValueError('Register common RGB video evaluation and RGB-D annotation/tracking frontends')
    n = cfg.get('total_clips')
    if type(n) is not int or n < 2 or n % 2:
        raise ValueError('downstream.total_clips must be a positive even number')
    if cfg.get('modality') != 'rgbd':
        raise ValueError('The Base identity experiment has one fixed RGB-D modality')
    gate = cfg.get('frontend_gate', {})
    for key in ('min_recall_50', 'min_reference_targets', 'min_positive_anchors', 'confidence'):
        if key not in gate or not math.isfinite(float(gate[key])):
            raise ValueError('Register frontend validation gate: ' + key)
    if not 0 < gate['min_recall_50'] <= 1 or not 0 < gate['confidence'] < 1:
        raise ValueError('Nontrivial recall and confidence gates required')
    if gate['min_reference_targets'] < 1 or gate['min_positive_anchors'] < 1:
        raise ValueError('Holdout must contain targets and identity supervision')
    if stage is not None:
        s = dict(cfg.get(stage, {}))
        for key in ('steps', 'batch', 'validate_every', 'validation_panel'):
            if type(s.get(key)) is not int or s[key] < 1:
                raise ValueError('Register downstream.%s.%s' % (stage, key))
        if not math.isfinite(float(s.get('lr', 0))) or s.get('lr', 0) <= 0:
            raise ValueError('Register positive downstream learning rate')
        if stage == 'identity' and s['batch'] % 2:
            raise ValueError('Mixed-source identity batches must be even')
        s.setdefault('weight_decay', .01)
        s.setdefault('temperature', .1)
        s.setdefault('clip_norm', 1.)
        s.setdefault('warmup_steps', min(100, s['steps']))
        s.setdefault('save_every', s['validate_every'])
        if min(s['temperature'], s['clip_norm'], s['save_every']) <= 0 or s['warmup_steps'] < 0:
            raise ValueError('Invalid registered optimizer settings')
        return cfg, s
    return cfg


def source_hashes():
    files = [Path(__file__), Path(__file__).with_name('train.py')]
    root = Path(__file__).resolve().parents[1]
    for directory in ('downstream_tracking', 'paper_experiments/agcr_exp'):
        files += [p for p in (root / directory).glob('*.py')
                  if not p.name.startswith(('test_', 'check_'))]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


def frozen_digest(state):
    """Identity is the only parameter allowed to change during head adaptation."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith('identity.'):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def positive_anchor_count(keys):
    count = Counter(keys)
    return sum(count[k] > 1 for k in keys) if len(count) > 1 else 0


def identity_objective(head, features, keys, temperature=.1):
    """Supervised contrastive identity training on predicted, not GT, queries."""
    if len(features) != len(keys):
        raise ValueError('Query/identity-label mismatch')
    zero = sum(p.sum() * 0 for p in head.parameters())
    if len(keys) < 2:
        return zero, dict(anchors=0, correct=0)
    embedding = F.normalize(head(features.float()), dim=-1)
    scores = embedding @ embedding.T / temperature
    eye = torch.eye(len(keys), device=scores.device, dtype=torch.bool)
    same = torch.tensor([[a == b for b in keys] for a in keys], device=scores.device)
    positive = same & ~eye
    eligible = positive.any(1) & (~same).any(1)
    n = int(eligible.sum())
    if not n:
        return zero, dict(anchors=0, correct=0)
    logp = scores - scores.masked_fill(eye, -torch.inf).logsumexp(1, keepdim=True)
    loss = -(logp * positive).sum(1)[eligible].div(positive.sum(1)[eligible]).mean()
    top = scores.masked_fill(eye, -torch.inf).argmax(1)
    correct = int(same[torch.arange(len(keys), device=scores.device), top][eligible].sum())
    return loss, dict(anchors=n, correct=correct)


def predicted_queries(model, image_inputs):
    """Read detector queries from an image-only forward; no labels/DN queries."""
    if set(image_inputs) != IMAGE_KEYS:
        raise ValueError('Only RGB, observed depth/validity and calibration may enter the frontend')
    captured = []
    def hook(_module, _args, output):
        count = int(model.detector.config.num_queries)
        captured.append(output.last_hidden_state[:, -count:])
    handle = model.detector.register_forward_hook(hook)
    try:
        prediction = model(**image_inputs)  # deliberately no labels argument
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape[:2] != prediction['boxes'].shape[:2]:
        raise RuntimeError('Unexpected RT-DETR query layout')
    return prediction, captured[0]


def matched_features(prediction, queries, targets):
    features, keys = [], []
    for frame, (pi, ti) in enumerate(matching(prediction, targets)):
        if pi:
            features.append(queries[frame, pi])
            keys.extend(targets[frame]['identity_keys'][j] for j in ti)
    if not features:
        return queries.reshape(-1, queries.shape[-1])[:0], []
    return torch.cat(features), keys


def candidate_detection_loss(prediction, targets):
    """Focal classification and visible-box regression after image-only forward."""
    class_target = torch.zeros_like(prediction['logits'])
    box_loss = prediction['boxes'].sum() * 0
    count = 0
    for i, (pi, ti) in enumerate(matching(prediction, targets)):
        if not pi:
            continue
        target = targets[i]
        class_target[i, pi, target['class_labels'][ti]] = 1
        boxes = prediction['boxes'][i, pi]
        reference = target['boxes'][ti]
        box_loss += F.l1_loss(boxes, reference, reduction='sum')
        box_loss += (1 - box_iou(xyxy(boxes), xyxy(reference)).diag()).sum()
        count += len(pi)
    probability = prediction['logits'].sigmoid()
    weight = (class_target - probability).abs().pow(2)
    weight *= class_target * .25 + (1 - class_target) * .75
    classification = (F.binary_cross_entropy_with_logits(
        prediction['logits'], class_target, reduction='none') * weight).sum() / max(count, 1)
    return classification + 5 * box_loss / max(count, 1)


def visual_loss(prediction, targets):
    """Supervise final and encoder queries without feeding labels into the network."""
    result = head_losses(prediction, targets)['total'] + candidate_detection_loss(prediction, targets)
    if ('encoder_logits' in prediction) != ('encoder_boxes' in prediction):
        raise ValueError('Incomplete RT-DETR encoder candidates')
    if 'encoder_logits' in prediction:
        # Top-k selection and detached decoder seeds otherwise leave these
        # newly initialized class heads without a learning signal.
        encoder = dict(logits=prediction['encoder_logits'], boxes=prediction['encoder_boxes'])
        result = result + candidate_detection_loss(encoder, targets)
    return result


def detection_counts(prediction, targets, confidence):
    hits = total = 0
    for i, (pi, ti) in enumerate(matching(prediction, targets)):
        total += len(targets[i]['class_labels'])
        if not pi:
            continue
        score, category = prediction['logits'][i, pi].sigmoid().max(-1)
        iou = box_iou(xyxy(prediction['boxes'][i, pi]), xyxy(targets[i]['boxes'][ti])).diag()
        hits += int(((score >= confidence) & (iou >= .5) &
                     (category == targets[i]['class_labels'][ti])).sum())
    return hits, total


def holdout_gate(report, gate):
    return bool(report.get('complete') and math.isfinite(report.get('objective', float('nan')))
                and report.get('reference_targets', 0) >= gate['min_reference_targets']
                and report.get('recall_50', 0) >= gate['min_recall_50']
                and report.get('positive_anchors', 0) >= gate['min_positive_anchors'])


def validate_frontend_payload(saved, index, role, expected_protocol=None):
    identity = saved.get('identity', {})
    report = saved.get('holdout_validation', {})
    if identity.get('schema') != SCHEMA or identity.get('scope') != 'frontend':
        raise ValueError('Use the independently validated Base frontend artifact')
    if identity.get('role') != role or identity.get('development_only', True):
        raise ValueError('Wrong frontend role or development checkpoint')
    if identity.get('modality') != frontend_modality(role):
        raise ValueError('Frontend modality differs from the registered role')
    if not saved.get('completed') or not saved.get('run_completed') or saved.get('step', 0) < 1:
        raise ValueError('An actually trained and completed frontend is required')
    if expected_protocol and identity.get('protocol_sha256') != expected_protocol:
        raise ValueError('Frontend protocol mismatch')
    if identity.get('split_sha256') != index['split_sha256']:
        raise ValueError('Frontend split mismatch')
    training = set(identity.get('training_episodes', []))
    validation = set(report.get('episodes', []))
    if not training or not validation or training & validation:
        raise ValueError('Independent native train/validation partitions required')
    if any(index['roles'].get(e) != ROLES[role] for e in training):
        raise ValueError('Frontend training-role leakage')
    if any(index['roles'].get(e) != 'val' for e in validation):
        raise ValueError('Frontend must be selected on validation, not test')
    if report.get('step') != saved['step'] or not holdout_gate(report, identity['frontend_gate']):
        raise ValueError('This checkpoint has no passing holdout validation')
    if saved.get('frozen_frontend_sha256') != frozen_digest(saved['model']):
        raise ValueError('Frontend parameters differ from the validated artifact')
    return saved


def selected_val(index, size):
    rows = [r for r in index['windows'] if r['role'] == 'val' and r['history_covered']]
    selected, panel = validation_panel(rows, size)
    if len(selected) < size:
        raise ValueError('Validation data smaller than registered panel')
    return selected, panel


class Job:
    def __init__(self, args, paths):
        self.root = Path(args.output).resolve()
        ledger = Path(args.budget_ledger or self.root / 'budget.json')
        assert_base_paths(self.root, ledger, paths)
        if not args.device.startswith('cuda'):
            raise ValueError('Execution uses an allocated CUDA device; CPU helpers are testable separately')
        self.budget = BudgetLedger(ledger, args.gpu_hours, 1)
        self.stopped = False
        self.stop_file = args.stop_file
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stopped', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stopped', True))
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.device = torch.device(args.device)
            torch.cuda.set_device(self.device)
            torch.set_num_threads(4)
        except BaseException:
            self.budget.close('initialization_failed')
            raise

    def stop(self):
        return self.stopped or self.budget.exhausted() or bool(self.stop_file and Path(self.stop_file).exists())

    def close(self, state):
        self.budget.close(state)


def immutable_manifest(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Existing artifact identity differs: ' + str(path))
    atomic(path, value)


def record_update(root, record):
    atomic(root / 'status.json', record)
    with (root / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(record, allow_nan=False), flush=True)


def resolve(args):
    paths = json.loads(Path(args.paths).read_text())
    protocol = json.loads(Path(args.protocol).read_text())
    index_path = args.index or paths.get('ordinary')
    assets = args.assets or paths.get('reader_assets') or paths.get('rtdetr_assets')
    if not index_path:
        raise ValueError('Specify --index or paths.ordinary')
    index = json.loads(Path(index_path).read_text())
    if index.get('development_only', True):
        raise ValueError('Base formal identity studies require a native accepted index')
    return paths, protocol, str(index_path), index, assets


def make_base_cohort(index, generated, regime, n, seed):
    if regime not in REGIMES:
        raise ValueError('Unknown Base regime')
    mapping = {'R': 'R', 'G': 'G', 'RG': 'R_G', 'RG1': 'R_G', 'Rnew': 'R_plus_new_render'}
    result = make_cohort(rendered_candidates(index), generated, mapping[regime],
                         n // 2 if regime == 'Rnew' else n, .5, seed,
                         rendered_candidates(index, 'augmentation_train'), 'rgbd')
    if result['clip_count'] != n:
        raise ValueError('All Base regimes must contain exactly N clips')
    result.update(schema='agcr-base48-cohort-1', regime=regime,
                  semantics='identity_head_training_only_common_rendered_frontend',
                  development_only=False)
    return result


def cohort_command(args):
    paths, protocol, index_path, index, _ = resolve(args)
    cfg, schedule = settings(protocol, 'identity')
    assert_base_paths(args.output, str(args.output) + '.ledger', paths)
    generated = []
    annotation_hash = None
    if args.regime in ('G', 'RG', 'RG1'):
        if not args.accepted_generated or not args.annotation_checkpoint:
            raise ValueError('Generated groups need accepted proposals and an independent annotation checkpoint')
        annotator = torch.load(args.annotation_checkpoint, map_location='cpu', weights_only=False)
        validate_frontend_payload(annotator, index, 'annotation', sha256(args.protocol))
        annotation_hash = sha256(args.annotation_checkpoint)
        document = json.loads(Path(args.accepted_generated).read_text())
        generated = document['accepted']
        for row in generated:
            if row.get('development_only', True) or row.get('annotation_reader_sha256') != annotation_hash:
                raise ValueError('Generated labels do not come from the validated independent annotator')
            if row.get('acceptance_schema') != 'agcr-acceptance-2':
                raise ValueError('RGB-D cohorts require accepted independent RGB-D labels')
            if sha256(row['annotation_path']) != row['annotation_sha256']:
                raise ValueError('Changed independent annotations')
    cohort = make_base_cohort(index, generated, args.regime, cfg['total_clips'], args.seed)
    cohort.update(protocol_sha256=sha256(args.protocol), index_sha256=sha256(index_path),
                  updates=schedule['steps'], batch_size=schedule['batch'],
                  annotation_reader_sha256=annotation_hash,
                  annotation_checkpoint=str(Path(args.annotation_checkpoint).resolve()) if args.annotation_checkpoint else None,
                  accepted_manifest_sha256=sha256(args.accepted_generated) if args.accepted_generated else None)
    immutable_manifest(args.output, cohort)


def lr_at(schedule, step):
    warmup = schedule['warmup_steps']
    ramp = min((step + 1) / max(warmup, 1), 1.)
    progress = max(0, step - warmup) / max(schedule['steps'] - warmup, 1)
    return schedule['lr'] * ramp * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def frontend_validation(model, index, rows, panel, device, job, step, gate):
    objectives = []; hits = total = anchors = 0
    model.eval()
    for row in rows:
        if job.stop():
            return None
        x, labels = native_clip(index['dataset_root'], row['episode'], row['start'] + 6)
        targets = move(make_targets(x, labels, row['episode']), device)
        prediction, queries = predicted_queries(model, move(pack_images(x), device))
        objectives.append(float(visual_loss(prediction, targets)))
        good, count = detection_counts(prediction, targets, gate['confidence'])
        hits += good; total += count
        _, keys = matched_features(prediction, queries, targets)
        anchors += positive_anchor_count(keys)
    objective, layouts = validation_mean(rows, objectives)
    return dict(complete=True, step=step, objective=objective, objective_by_layout=layouts,
                panel=panel, episodes=sorted({r['episode'] for r in rows}),
                reference_targets=total, correct_detections_50=hits,
                recall_50=hits / max(total, 1), positive_anchors=anchors,
                scope='independent_native_validation_no_ground_truth_network_inputs')


def frontend_command(args):
    paths, protocol, index_path, index, assets = resolve(args)
    cfg, schedule = settings(protocol, 'frontend')
    if not assets:
        raise ValueError('Specify RT-DETR --assets')
    modality = frontend_modality(args.role)
    data = ClipDataset(index_path, args.role, modality=modality)
    rows, panel = selected_val(index, schedule['validation_panel'])
    training = sorted({r['episode'] for r in data.rows})
    identity = dict(schema=SCHEMA, scope='frontend', role=args.role, modality=modality, seed=args.seed,
                    development_only=False, protocol_sha256=sha256(args.protocol),
                    index_sha256=sha256(index_path), split_sha256=index['split_sha256'],
                    training_episodes=training, training_partition_sha256=digest(training),
                    source_sha256=source_hashes(), schedule=schedule,
                    frontend_gate=cfg['frontend_gate'], validation_panel=panel,
                    assets={p.name: sha256(p) for p in sorted(Path(assets).glob('*')) if p.is_file()},
                    generator_weights_used=False, input_labels_used=False,
                    visual_objective='image_only_final_plus_encoder_detection_independent_auxiliary_v2',
                    declared_protocol='Base48 independently registered short adaptation; not Pro40k')
    job = Job(args, paths); state = 'failed'
    try:
        immutable_manifest(job.root / 'manifest.json', identity)
        last = job.root / 'last.pt'
        if last.exists() and not args.resume:
            raise ValueError('Existing frontend checkpoint; use --resume')
        saved = torch.load(last, map_location='cpu', weights_only=False) if args.resume else None
        if saved and saved['identity'] != identity:
            raise ValueError('Frontend resume contract changed')
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = CooperativeDetector(assets, modality, pretrained=not bool(saved)).to(job.device)
        prefixes = schedule.get('trainable_prefixes', ['detector.', 'identity.', 'contact.', 'pixel.', 'mask_query.', 'depth_head.'])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(any(name.startswith(p) for p in prefixes))
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError('No visual frontend parameters selected')
        optimizer = torch.optim.AdamW(parameters, lr=schedule['lr'], weight_decay=schedule['weight_decay'])
        step = 0; best = float('inf'); validation = None; payload = saved
        if saved:
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            step = saved['step']; best = saved['best_validation']; restore_rng(saved['rng'])
            validation = saved.get('holdout_validation')
        model.eval()  # fixed image-only path and BN behavior; gradients remain enabled
        while step < schedule['steps'] and not job.stop():
            optimizer.zero_grad(set_to_none=True); value = 0.
            for i in batch_indices(data, args.seed, step, schedule['batch']):
                x, y = data[i]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(**move(pack_images(x), job.device))
                    loss = visual_loss(prediction, move(y, job.device)) / schedule['batch']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite frontend objective')
                loss.backward(); value += float(loss.detach())
            nn.utils.clip_grad_norm_(parameters, schedule['clip_norm'])
            for group in optimizer.param_groups:
                group['lr'] = lr_at(schedule, step)
            optimizer.step(); step += 1
            improved = False
            if (step % schedule['validate_every'] == 0 or step == schedule['steps']) and not job.stop():
                validation = frontend_validation(model, index, rows, panel, job.device, job, step, cfg['frontend_gate'])
                if validation:
                    validation['passed'] = holdout_gate(validation, cfg['frontend_gate'])
                    atomic(job.root / 'validation.json', validation)
                    improved = validation['passed'] and validation['objective'] < best
                    if improved:
                        best = validation['objective']
            payload = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                           step=step, completed=False, run_completed=False, rng=rng(),
                           best_validation=best, holdout_validation=validation,
                           gpu_hours=job.budget.used, frozen_frontend_sha256=frozen_digest(model.state_dict()))
            if improved:
                atomic_checkpoint(job.root / 'best.pt', payload)
            if improved or step % schedule['save_every'] == 0 or step == schedule['steps'] or job.stop():
                atomic_checkpoint(last, payload)
            record_update(job.root, dict(state='running', step=step, steps=schedule['steps'],
                   scope='frontend', loss=value, gpu_hours=job.budget.used, development_only=False,
                   validation=validation if validation and validation.get('step') == step else None))
        if payload is not None:
            atomic_checkpoint(last, payload)
        complete = step == schedule['steps'] and (job.root / 'best.pt').exists()
        if complete:
            selected = torch.load(job.root / 'best.pt', map_location='cpu', weights_only=False)
            selected.update(completed=True, run_completed=True, training_completed_steps=step,
                            best_step=selected['step'], selection='passing_holdout_then_minimum_objective')
            atomic_checkpoint(job.root / 'best.pt', selected)
            validate_frontend_payload(selected, index, args.role, sha256(args.protocol))
        state = 'completed' if complete else 'validation_failed' if step == schedule['steps'] else 'paused'
        atomic(job.root / 'completion.json', dict(completed=complete, trained_steps=step,
               validation_passed=complete, development_only=False, role=args.role, schema=SCHEMA,
               checkpoint_sha256=sha256(job.root / 'best.pt') if complete else None))
        atomic(job.root / 'status.json', dict(state=state, step=step, steps=schedule['steps'], gpu_hours=job.budget.used))
    finally:
        job.close(state)


def cache_command(args):
    paths, protocol, index_path, index, assets = resolve(args)
    cfg, schedule = settings(protocol, 'identity')
    saved = torch.load(args.frontend, map_location='cpu', weights_only=False)
    validate_frontend_payload(saved, index, 'downstream', sha256(args.protocol))
    frontend_hash = sha256(args.frontend)
    if args.split == 'train':
        if not args.cohort:
            raise ValueError('Training query cache requires an immutable cohort')
        data = ClipDataset(index_path, 'downstream', args.cohort, 'rgbd')
        cohort = data.cohort
        if cohort.get('schema') != 'agcr-base48-cohort-1' or cohort.get('protocol_sha256') != sha256(args.protocol):
            raise ValueError('Use the registered Base cohort')
        if cohort.get('annotation_reader_sha256') == frontend_hash:
            raise ValueError('Generated annotator must be independent of downstream frontend')
        rows = data.rows; panel = None
    else:
        data = None; cohort = None
        rows, panel = selected_val(index, schedule['validation_panel'])
    identity = dict(schema=CACHE_SCHEMA, split=args.split, protocol_sha256=sha256(args.protocol),
                    frontend_sha256=frontend_hash, frozen_frontend_sha256=saved['frozen_frontend_sha256'],
                    index_sha256=sha256(index_path), split_sha256=index['split_sha256'],
                    cohort_sha256=sha256(args.cohort) if args.cohort else None,
                    cohort_regime=cohort['regime'] if cohort else None,
                    source_sha256=source_hashes(), rows=rows, validation_panel=panel,
                    feature_policy='predicted_detector_query; labels_only_offline_matching',
                    development_only=False)
    job = Job(args, paths); state = 'failed'
    try:
        immutable_manifest(job.root / 'manifest.json', identity)
        model = CooperativeDetector(assets, 'rgbd', pretrained=False).to(job.device)
        model.load_state_dict(saved['model']); model.eval().requires_grad_(False)
        entries = []
        for i, row in enumerate(rows):
            key = hashlib.sha256(row['uid'].encode()).hexdigest()
            file = job.root / (key + '.pt'); marker = job.root / (key + '.json')
            if marker.exists():
                entry = json.loads(marker.read_text())
                if not file.exists() or entry['sha256'] != sha256(file):
                    raise ValueError('Changed cached predicted features')
                entries.append(entry); continue
            if job.stop():
                break
            if data is not None:
                inputs, targets = data[i]
            else:
                inputs, labels = native_clip(index['dataset_root'], row['episode'], row['start'] + 6)
                targets = make_targets(inputs, labels, row['episode'])
            with torch.no_grad():
                prediction, queries = predicted_queries(model, move(pack_images(inputs), job.device))
                features, keys = matched_features(prediction, queries, move(targets, job.device))
            atomic_checkpoint(file, dict(features=features.float().cpu(), identity_keys=keys,
                                         uid=row['uid'], frontend_sha256=frontend_hash))
            entry = dict(uid=row['uid'], path=str(file), sha256=sha256(file),
                         source=row.get('source', 'rendered'), episode=row['episode'],
                         positive_anchors=positive_anchor_count(keys))
            atomic(marker, entry); entries.append(entry)
            atomic(job.root / 'status.json', dict(completed_clips=len(entries), total_clips=len(rows), gpu_hours=job.budget.used))
        complete = len(entries) == len(rows)
        state = 'completed' if complete else 'paused'
        atomic(job.root / 'completion.json', dict(completed=complete, development_only=False,
               manifest_sha256=sha256(job.root / 'manifest.json'), clips=entries,
               total_clips=len(rows), positive_anchors=sum(x['positive_anchors'] for x in entries)))
    finally:
        job.close(state)


class QueryCache:
    def __init__(self, root, split, protocol_hash, frontend_hash):
        root = Path(root)
        self.manifest = json.loads((root / 'manifest.json').read_text())
        completion = json.loads((root / 'completion.json').read_text())
        if (self.manifest.get('schema') != CACHE_SCHEMA or self.manifest.get('split') != split
                or self.manifest.get('protocol_sha256') != protocol_hash
                or self.manifest.get('frontend_sha256') != frontend_hash):
            raise ValueError('Query cache role/provenance mismatch')
        if not completion.get('completed') or completion.get('development_only', True):
            raise ValueError('Incomplete/development cache cannot train a formal identity head')
        if completion['manifest_sha256'] != sha256(root / 'manifest.json'):
            raise ValueError('Cache manifest changed')
        self.entries = completion['clips']; self.rows = self.manifest['rows']
        if len(self.rows) != len(self.entries) or [r['uid'] for r in self.rows] != [r['uid'] for r in self.entries]:
            raise ValueError('Partial or reordered query cache')
        self.signature = sha256(root / 'completion.json')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        entry = self.entries[i]
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('Query cache data changed')
        data = torch.load(entry['path'], map_location='cpu', weights_only=False)
        if data['uid'] != entry['uid'] or data['frontend_sha256'] != self.manifest['frontend_sha256']:
            raise ValueError('Mixed frontend cache')
        return data['features'], data['identity_keys']


@torch.no_grad()
def head_validation(head, cache, device, job, temperature):
    losses = []; count = correct = 0
    for i in range(len(cache.rows)):
        if job.stop():
            return None
        features, keys = cache[i]
        loss, metrics = identity_objective(head, features.to(device), keys, temperature)
        losses.append(float(loss)); count += metrics['anchors']; correct += metrics['correct']
    value, layouts = validation_mean(cache.rows, losses)
    return dict(complete=True, objective=value, objective_by_layout=layouts,
                positive_anchors=count, retrieval_correct=correct,
                retrieval_accuracy=correct / max(count, 1),
                episodes=sorted({r['episode'] for r in cache.rows}))


def train_command(args):
    paths, protocol, index_path, index, assets = resolve(args)
    cfg, schedule = settings(protocol, 'identity')
    frontend = torch.load(args.frontend, map_location='cpu', weights_only=False)
    validate_frontend_payload(frontend, index, 'downstream', sha256(args.protocol))
    frontend_hash = sha256(args.frontend); protocol_hash = sha256(args.protocol)
    train = QueryCache(args.train_cache, 'train', protocol_hash, frontend_hash)
    val = QueryCache(args.val_cache, 'val', protocol_hash, frontend_hash)
    if len(train.rows) != cfg['total_clips']:
        raise ValueError('All regimes must use the registered matched N')
    training = sorted({r['episode'] for r in train.rows})
    if set(training) & {r['episode'] for r in val.rows}:
        raise ValueError('Identity training/validation episode leakage')
    identity = dict(schema=SCHEMA, scope='identity_head', role='downstream', modality='rgbd',
                    seed=args.seed, regime=train.manifest['cohort_regime'], development_only=False,
                    protocol_sha256=protocol_hash, index_sha256=sha256(index_path),
                    split_sha256=index['split_sha256'], schedule=schedule, source_sha256=source_hashes(),
                    training_episodes=training, training_partition_sha256=digest(training),
                    frontend_training_episodes=frontend['identity']['training_episodes'],
                    frontend_checkpoint_sha256=frontend_hash,
                    train_cache_sha256=train.signature, val_cache_sha256=val.signature,
                    generator_weights_used=False, trainable_parameters=['identity.weight', 'identity.bias'],
                    declared_protocol='Base48 identity-only adaptation; fixed detection and geometry')
    job = Job(args, paths); state = 'failed'
    try:
        immutable_manifest(job.root / 'manifest.json', identity)
        last = job.root / 'last.pt'
        if last.exists() and not args.resume:
            raise ValueError('Existing identity checkpoint; use --resume')
        saved = torch.load(last, map_location='cpu', weights_only=False) if args.resume else None
        if saved and saved['identity'] != identity:
            raise ValueError('Identity resume contract changed')
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        weight = frontend['model']['identity.weight']
        head = nn.Linear(weight.shape[1], weight.shape[0]).to(job.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=schedule['lr'], weight_decay=schedule['weight_decay'])
        step = 0; best = float('inf'); validation = None; payload = saved
        if saved:
            head.load_state_dict({k[len('identity.'):]: v for k, v in saved['model'].items() if k.startswith('identity.')})
            optimizer.load_state_dict(saved['optimizer']); step = saved['step']
            best = saved['best_validation']; validation = saved.get('holdout_validation'); restore_rng(saved['rng'])
        while step < schedule['steps'] and not job.stop():
            optimizer.zero_grad(set_to_none=True); value = 0.; anchors = 0
            for i in batch_indices(train, args.seed, step, schedule['batch']):
                features, keys = train[i]
                loss, metrics = identity_objective(head, features.to(job.device), keys, schedule['temperature'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite identity objective')
                (loss / schedule['batch']).backward(); value += float(loss) / schedule['batch']
                anchors += metrics['anchors']
            if anchors == 0:
                raise ValueError('No positive/negative identity supervision in this batch')
            nn.utils.clip_grad_norm_(head.parameters(), schedule['clip_norm'])
            for group in optimizer.param_groups:
                group['lr'] = lr_at(schedule, step)
            optimizer.step(); step += 1; improved = False
            if (step % schedule['validate_every'] == 0 or step == schedule['steps']) and not job.stop():
                validation = head_validation(head, val, job.device, job, schedule['temperature'])
                if validation:
                    validation.update(step=step, scope='identity_head_native_validation')
                    validation['passed'] = validation['positive_anchors'] >= cfg['frontend_gate']['min_positive_anchors']
                    improved = validation['passed'] and validation['objective'] < best
                    if improved:
                        best = validation['objective']
                    atomic(job.root / 'validation.json', validation)
            model_state = dict(frontend['model'])
            model_state.update({'identity.' + k: v.detach().cpu() for k, v in head.state_dict().items()})
            payload = dict(identity=identity, model=model_state, optimizer=optimizer.state_dict(),
                           step=step, completed=False, run_completed=False, rng=rng(),
                           best_validation=best, holdout_validation=validation,
                           frontend_validation=frontend['holdout_validation'], gpu_hours=job.budget.used,
                           frozen_frontend_sha256=frontend['frozen_frontend_sha256'])
            if improved or step % schedule['save_every'] == 0 or step == schedule['steps'] or job.stop():
                if frozen_digest(model_state) != frontend['frozen_frontend_sha256']:
                    raise RuntimeError('Frozen visual frontend changed during identity training')
                atomic_checkpoint(last, payload)
            if improved:
                atomic_checkpoint(job.root / 'best.pt', payload)
            record_update(job.root, dict(state='running', step=step, steps=schedule['steps'],
                   scope='identity_head', loss=value, positive_anchors=anchors, gpu_hours=job.budget.used,
                   validation=validation if validation and validation.get('step') == step else None))
        if payload is not None:
            atomic_checkpoint(last, payload)
        complete = step == schedule['steps'] and (job.root / 'best.pt').exists()
        if complete:
            selected = torch.load(job.root / 'best.pt', map_location='cpu', weights_only=False)
            selected.update(completed=True, run_completed=True, training_completed_steps=step,
                            best_step=selected['step'], selection='native_validation_identity_objective')
            atomic_checkpoint(job.root / 'best.pt', selected)
        state = 'completed' if complete else 'validation_failed' if step == schedule['steps'] else 'paused'
        atomic(job.root / 'completion.json', dict(completed=complete, development_only=False,
               role='downstream', schema=SCHEMA, scope='identity_head', trained_steps=step,
               checkpoint_sha256=sha256(job.root / 'best.pt') if complete else None))
        atomic(job.root / 'status.json', dict(state=state, step=step, steps=schedule['steps'], gpu_hours=job.budget.used))
    finally:
        job.close(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('cohort', 'frontend', 'cache', 'train'):
        p = sub.add_parser(name)
        for key in ('protocol', 'paths', 'output'):
            p.add_argument('--' + key, required=True)
        p.add_argument('--index'); p.add_argument('--assets')
        p.add_argument('--seed', type=int, default=17)
        if name != 'cohort':
            p.add_argument('--gpu-hours', type=float, required=True,
                           help='Cumulative cap for this job ledger, including setup/validation/restarts')
            p.add_argument('--budget-ledger'); p.add_argument('--stop-file')
            p.add_argument('--resume', action='store_true'); p.add_argument('--device', default='cuda:0')
        if name == 'cohort':
            p.add_argument('--regime', choices=REGIMES, required=True)
            p.add_argument('--accepted-generated'); p.add_argument('--annotation-checkpoint')
        if name == 'frontend':
            p.add_argument('--role', choices=tuple(ROLES), required=True)
        if name in ('cache', 'train'):
            p.add_argument('--frontend', required=True)
        if name == 'cache':
            p.add_argument('--split', choices=('train', 'val'), required=True); p.add_argument('--cohort')
        if name == 'train':
            p.add_argument('--train-cache', required=True); p.add_argument('--val-cache', required=True)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('These independent cached-head jobs use one GPU each; parallelize jobs, not duplicate writers')
    {'cohort': cohort_command, 'frontend': frontend_command, 'cache': cache_command,
     'train': train_command}[args.command](args)


if __name__ == '__main__':
    main()
