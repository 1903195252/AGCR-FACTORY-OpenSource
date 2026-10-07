"""Reference-matched observation diagnostics; not independent tracking evaluation."""
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class ObservationDiagnostics:
    def __init__(self):
        self.rows = []
        self.association = dict(tp=0, fp=0, tn=0, fn=0)

    @torch.no_grad()
    def add(self, model, output, target):
        assigned = []
        visibility = target['detectable'].long()
        groups = visibility[0] * 2 + visibility[1]
        names = {0:'neither_detectable', 1:'ground_only', 2:'aerial_only', 3:'both_visible'}
        for view in range(2):
            ids = torch.where(target['detectable'][view])[0]
            if not len(ids):
                assigned.append((torch.empty(0, device=ids.device, dtype=torch.long), ids))
                continue
            probability = output['class_logits'][view].softmax(-1)
            cost = torch.cdist(output['center'][view], target['center'][view, ids], p=1) / 3
            cost = cost - probability[:, target['class_id'][ids]]
            query, reference = linear_sum_assignment(cost.cpu().numpy())
            query = torch.as_tensor(query, device=ids.device)
            reference = ids[torch.as_tensor(reference, device=ids.device)]
            assigned.append((query, reference))
            center = (output['center'][view, query]-target['center'][view, reference]).norm(dim=-1)
            dimensions = (output['dimensions'][view, query]-target['dimensions'][view, reference]).abs().mean(-1)
            cosine = (output['heading'][view, query]*target['heading'][view, reference]).sum(-1).clamp(-1,1)
            heading = torch.acos(cosine)*180/np.pi
            correct = (probability[query].argmax(-1)==target['class_id'][reference]).float()
            ious = []
            for name, key in [('P','full_mask'), ('M','visible_mask')]:
                pred, ref = output[name][view, query], target[key][view, reference]
                ious.append(torch.minimum(pred, ref).sum((-2,-1))/torch.maximum(pred, ref).sum((-2,-1)).clamp_min(1e-8))
            values = torch.stack([center, dimensions, heading, correct, *ious], -1).cpu().tolist()
            classes = target['class_id'][reference].cpu().tolist()
            states = groups[reference].cpu().tolist()
            for values_row, category, state in zip(values, classes, states):
                self.rows.append(dict(zip(['center_error_m','dimension_mae_m','heading_error_deg','class_accuracy',
                                           'full_mask_soft_iou','visible_mask_soft_iou'], values_row),
                                      class_id=category, visibility=names[state], view='air' if view==0 else 'ground'))
        if all(len(query) for query, _ in assigned):
            left={key:output[key][0,assigned[0][0]] for key in ('tokens','center','covariance','class_logits')}
            right={key:output[key][1,assigned[1][0]] for key in left}
            pred=model.association_logits(left,right).sigmoid() >= .5
            ref=assigned[0][1][:,None]==assigned[1][1][None]
            for name, value in [('tp',pred&ref),('fp',pred&~ref),('tn',~pred&~ref),('fn',~pred&ref)]:
                self.association[name] += int(value.sum())

    def result(self):
        metrics=['center_error_m','dimension_mae_m','heading_error_deg','class_accuracy',
                 'full_mask_soft_iou','visible_mask_soft_iou']
        def summarize(rows):
            return dict(target_view_instances=len(rows),
                        **{key:float(np.mean([r[key] for r in rows])) if rows else None for key in metrics})
        groups={}
        for field in ('visibility','class_id','view'):
            buckets=defaultdict(list)
            for row in self.rows:
                buckets[str(row[field])].append(row)
            groups[field]={key:summarize(rows) for key,rows in sorted(buckets.items())}
        a=self.association
        precision=a['tp']/(a['tp']+a['fp']) if a['tp']+a['fp'] else None
        recall=a['tp']/(a['tp']+a['fn']) if a['tp']+a['fn'] else None
        return dict(schema='agcr-observation-diagnostics-1',
                    matching='Reference Hungarian assignment; diagnostic only, not detector/tracker test metrics',
                    mask_resolution=[24,36], mask_metric='soft intersection-over-union',
                    aggregation='mean over reference-detectable target/view instances',
                    overall=summarize(self.rows), by=groups,
                    association=dict(**a,precision=precision,recall=recall))
