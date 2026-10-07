"""Reference-only clip quotas and reproducible coverage-stratified selection."""
import math
from collections import Counter, defaultdict
from .protocol import DEFAULT, digest


def _features(row):
    features = {('layout', str(row['layout'])), ('event', row['script_event'])}
    features.update(('class', str(c)) for c in row['target_classes'])
    features.add(('focal_class', str(row.get('focal_class_id', row['target_classes'][0]))))
    for pair in row.get('anchor_pair', []):
        features.add(('visibility_pair', str(pair)))
    features.add(('phase', 'event' if row.get('event_window') else 'context'))
    return features


def _stratified_iter(rows, seed=17029):
    """Greedy marginal coverage, then balanced layout/event/class exposure.

    Only reference metadata is used. Input order, method and model scores cannot
    affect selection. The ordering is nested for smaller sample budgets.
    """
    if len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate window identifiers')
    remaining = {r['uid']: r for r in rows}
    features = {uid: tuple(sorted(_features(r))) for uid, r in remaining.items()}
    counts = Counter()
    ties = {uid:digest([seed,uid]) for uid in remaining}
    while remaining:
        def score(uid):
            f = features[uid]
            unseen = sum(counts[x] == 0 for x in f)
            balance = math.fsum(1. / (1 + counts[x]) for x in f)
            return (-unseen, -balance, ties[uid])
        uid = min(remaining, key=score)
        row=remaining.pop(uid); counts.update(features[uid])
        yield row


def stratified_order(rows, seed=17029, limit=None):
    from itertools import islice
    return list(islice(_stratified_iter(rows,seed),limit)) if limit is not None else list(_stratified_iter(rows,seed))


def validation_panel(rows, minimum=64, seed=17029):
    if minimum < 1 or not rows:
        raise ValueError('Empty/invalid validation panel')
    if any(r['role'] != 'val' for r in rows):
        raise ValueError('Validation selection must not use training or test data')
    # Cover every available marginal even when the requested size is too small.
    ordered = _stratified_iter(rows, seed)
    required = set().union(*(_features(r) for r in rows))
    covered = set(); selected = []
    for row in ordered:
        selected.append(row); covered.update(_features(row))
        if len(selected) >= minimum and covered == required:
            break
    selected_features = Counter(f for r in selected for f in _features(r))
    result = dict(schema='agcr-validation-panel-1', algorithm='marginal-coverage-fsum-v1', seed=seed,
                  requested_minimum=minimum, selected_uids=[r['uid'] for r in selected],
                  selected_windows=len(selected), available_windows=len(rows),
                  available_marginals=[list(f) for f in sorted(required)],
                  coverage=[dict(dimension=k[0], value=k[1], windows=v)
                            for k,v in sorted(selected_features.items())],
                  weighting='equal_layout_then_equal_camera_group_then_equal_window')
    result['panel_sha256'] = digest(result)
    return selected, result


def rendered_candidates(index, role='train'):
    """The four manuscript windows only; no dense/duplicate windows to fill quotas."""
    rows = []
    for r in index['windows']:
        if r['role'] != role or r['start'] not in DEFAULT['evaluation_starts']:
            continue
        if not r['history_covered']:
            continue
        rows.append(dict(r, split_sha256=index['split_sha256'], frames=16,
                         views=['air', 'ground'], includes_history=False))
    if len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate rendered candidates')
    return rows


def quota_report(index, target=None):
    """A fixed target, never reduced after seeing generated-data acceptance/results."""
    target = DEFAULT['downstream']['total_clips'] if target is None else target
    result = dict(schema='agcr-downstream-quota-1', target_clips=target,
                  split_sha256=index['split_sha256'], development_only=index['development_only'],
                  fixed_starts=DEFAULT['evaluation_starts'], pools={})
    for role in ['train', 'augmentation_train']:
        all_rows = [r for r in index['windows'] if r['role'] == role
                    and r['start'] in DEFAULT['evaluation_starts']]
        eligible = rendered_candidates(index, role)
        deficit = max(0, target - len(eligible))
        by_layout = Counter(r['layout'] for r in eligible)
        planned_max=index.get('planned_fixed_window_capacity_upper_bound',{}).get(role)
        structural_deficit=None if planned_max is None else max(0,target-planned_max)
        result['pools'][role] = dict(
            fixed_windows_before_filter=len(all_rows), eligible_clips=len(eligible),
            unique_episodes=len({r['episode'] for r in eligible}),
            unique_camera_groups=len({r['camera_group'] for r in eligible}),
            deficit_clips=deficit, missing_eligible_clips_to_target=deficit,
            deficit_is_final=not index['development_only'],
            planned_fixed_window_capacity_upper_bound=planned_max,
            structural_deficit_lower_bound=structural_deficit,
            # Two episodes / camera group, four fixed windows / episode.
            minimum_additional_camera_groups=((deficit+7)//8 if not index['development_only'] else
                None if structural_deficit is None else (structural_deficit+7)//8),
            eligible_by_layout={str(k):v for k,v in sorted(by_layout.items())},
            eligible_by_event=dict(Counter(r['script_event'] for r in eligible)),
            eligible_event_windows=sum(bool(r['event_window']) for r in eligible))
    result['reference_quota_ready'] = (not index['development_only'] and
        all(p['deficit_clips'] == 0 for p in result['pools'].values()))
    result['supplement_policy'] = (
        'Capture new camera-path groups in training layouts; keep both repeats in the same role. '
        'Do not reuse reader/ID-test paths. Each supplement must pass event and history-coverage '
        'checks before counting. Minimum groups assume all eight fixed windows qualify and are '
        'a lower bound, not a guaranteed acquisition count. Freeze a new index before training.')
    return result


def validation_mean(rows, losses):
    """Fixed panel: equal layouts, then camera groups, then selected windows."""
    import math
    if len(rows)!=len(losses) or not rows or not all(math.isfinite(x) for x in losses):
        raise ValueError('Incomplete/nonfinite validation results')
    groups=defaultdict(lambda:defaultdict(list))
    for row,value in zip(rows,losses):
        groups[row['layout']][row['camera_group']].append(value)
    layouts={str(layout):sum(sum(v)/len(v) for v in by.values())/len(by)
             for layout,by in groups.items()}
    return sum(layouts.values())/len(layouts),layouts
