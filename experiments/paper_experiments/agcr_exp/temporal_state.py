"""History-only identities, confidence-weighted filtering and recurrent state.

The inference method takes estimated observations, never simulator identities.
Reference assignment is confined to the separate supervised_loss function.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .modules import hungarian_association


def chol_inverse(matrix):
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    return torch.cholesky_inverse(torch.linalg.cholesky((matrix+matrix.transpose(-1, -2))*.5+1e-6*eye))


def gaussian_loss(error, covariance):
    eye = torch.eye(error.shape[-1], device=error.device, dtype=error.dtype)
    L = torch.linalg.cholesky((covariance+covariance.transpose(-1, -2))*.5+1e-6*eye)
    distance = (error[..., None]*torch.cholesky_solve(error[..., None], L)).sum((-2, -1))
    return .5*(distance+2*L.diagonal(dim1=-2, dim2=-1).log().sum(-1))/error.shape[-1]


def contrastive(features, identities, temperature=.1):
    if len(features) < 2:
        return features.sum()*0
    features = F.normalize(features, dim=-1)
    score = features@features.T/temperature
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    same = (identities[:, None] == identities[None])&~eye&(identities[:, None] >= 0)
    good = same.any(-1)
    logp = score-torch.logsumexp(score.masked_fill(eye, -1e4), -1, keepdim=True)
    return -(logp*same).sum(-1)[good].div(same.sum(-1)[good]).mean() if good.any() else score.sum()*0


class StateScale(nn.Module):
    def __init__(self, statistics):
        super().__init__()
        for name in ('mean', 'std'):
            self.register_buffer(name, torch.tensor(statistics['state'][name], dtype=torch.float32))
        self.register_buffer('acc_mean', torch.tensor(statistics['acceleration']['mean'], dtype=torch.float32))
        self.register_buffer('acc_std', torch.tensor(statistics['acceleration']['std'], dtype=torch.float32))
        self.register_buffer('velocity_mean', torch.tensor([statistics['class_velocity'][str(k)]['mean'] for k in range(1, 5)], dtype=torch.float32))
        self.register_buffer('velocity_cov', torch.tensor([statistics['class_velocity'][str(k)]['covariance'] for k in range(1, 5)], dtype=torch.float32))

    def encode(self, states):
        normalized = (states-self.mean)/self.std
        return torch.cat([normalized[..., :6], states[..., 6:7].sin(), states[..., 6:7].cos(), normalized[..., 7:]], -1)


class TemporalState(nn.Module):
    def __init__(self, statistics, pair_head, width=256):
        super().__init__()
        import copy
        self.scale = StateScale(statistics)
        self.association = copy.deepcopy(pair_head)
        self.reliability = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.memory = nn.GRUCell(width, width)
        self.log_process = nn.Parameter(torch.full((6,), -5.))
        self.token = nn.Sequential(nn.Linear(width+6+36+1, width), nn.GELU(), nn.Linear(width, width))
        self.presence = nn.Linear(width, 1)
        self.readout = nn.Linear(width, 5+3+6)
        # Four quadrant-pooled DINO / single-frame VAE descriptors, direction,
        # physical time and evidence weight. Source encoders remain frozen.
        self.appearance = nn.Sequential(nn.Linear(768+48+5, width), nn.GELU(), nn.Linear(width, width))

    def probabilities(self, tracks, observations):
        if not tracks:
            return observations['tokens'].new_empty((0, len(observations['tokens'])))
        descriptors = torch.stack([track['descriptor'] for track in tracks])
        center = torch.stack([track['mean'][:3] for track in tracks])
        cov = torch.stack([track['covariance'][:3, :3] for track in tracks])
        probs = torch.stack([track['class_prob'] for track in tracks])
        left = descriptors[:, None].expand(-1, len(observations['tokens']), -1)
        right = observations['tokens'][None].expand(len(tracks), -1, -1)
        delta = center[:, None]-observations['center'][None]
        R = torch.diag_embed(observations['covariance'])
        inverse = chol_inverse(cov[:, None]+R[None])
        mahal = ((delta[..., None]* (inverse@delta[..., None])).sum((-2, -1))+1e-8).sqrt()
        compatible = probs@observations['class_logits'].softmax(-1).T
        cosine = F.cosine_similarity(left, right, dim=-1)
        return self.association(torch.cat([left, right, mahal[..., None], compatible[..., None], cosine[..., None]], -1)).squeeze(-1).sigmoid()

    def predict(self, track, dt):
        scale = self.scale.std[:6]
        transition = torch.eye(6, device=scale.device)
        transition[:3, 3:] = torch.eye(3, device=scale.device)*dt
        prior = dict(track)
        prior['mean'] = transition@track['mean']
        prior['covariance'] = transition@track['covariance']@transition.T+torch.diag((F.softplus(self.log_process)+1e-4)*scale.square())*dt
        prior['age'] = track['age']+dt
        return prior

    def update(self, track, evidence, initial=False):
        """Physical units externally, Cholesky operations in standardized units."""
        scale = self.scale.std[:6]
        mean0 = self.scale.mean[:6]
        weights = torch.stack([item['weight'] for item in evidence]) if evidence else scale.new_empty(0)
        total = weights.sum()
        if evidence:
            centers = torch.stack([item['center'] for item in evidence])
            measurement = torch.diag_embed(torch.stack([item['covariance'] for item in evidence])/scale[:3].square())
            inverse = chol_inverse(measurement)
            precision = (weights[:, None, None]*inverse).sum(0)
            information = (weights[:, None]*(inverse@((centers-mean0[:3])/scale[:3])[..., None]).squeeze(-1)).sum(0)
            if initial:
                position_cov = chol_inverse(precision)
                position = position_cov@information
                class_prob = (weights[:, None]*torch.stack([item['class_prob'] for item in evidence])).sum(0)/total.clamp_min(1e-8)
                cls = int(class_prob[1:].argmax())
                track['mean'] = torch.cat([position*scale[:3]+mean0[:3], self.scale.velocity_mean[cls]])
                track['covariance'] = torch.block_diag(position_cov*scale[:3, None]*scale[None, :3], self.scale.velocity_cov[cls])
                track['class_prob'] = class_prob
                track['memory'] = scale.new_zeros(256)
                track['weight_sum'] = scale.new_zeros(())
            else:
                C = track['covariance']/scale[:, None]/scale[None, :]
                J = chol_inverse(C)
                eta = J@((track['mean']-mean0)/scale)
                postC = chol_inverse(J+F.pad(precision, (0, 3, 0, 3)))
                post = postC@(eta+F.pad(information, (0, 3)))
                track['mean'] = post*scale+mean0
                track['covariance'] = postC*scale[:, None]*scale[None, :]
            descriptor = (weights[:, None]*torch.stack([item['token'] for item in evidence])).sum(0)/total.clamp_min(1e-8)
            gate = 1-torch.exp(-total)
            track['memory'] = track['memory']+gate*(self.memory(descriptor, track['memory'])-track['memory'])
            denominator = track['weight_sum']+total
            for name, key in [('class_prob', 'class_prob'), ('dimensions', 'dimensions'), ('heading', 'heading'), ('descriptor', 'token')]:
                old = track.get(name, torch.zeros_like(evidence[0][key]))*track['weight_sum']
                value = (weights[:, None]*torch.stack([item[key] for item in evidence])).sum(0)
                track[name] = (old+value)/denominator.clamp_min(1e-8)
            track['heading'] = F.normalize(track['heading'], dim=-1)
            track['weight_sum'] = denominator
            track['age'] = scale.new_zeros(1)
            track['score'] = track.get('score', 0.)+sum(item['confidence'] for item in evidence)
            track['appearance'].extend([item['appearance'] for item in evidence])
        normalized = (track['mean']-mean0)/scale
        covariance = track['covariance']/scale[:, None]/scale[None, :]
        track['token'] = self.token(torch.cat([track['memory'], normalized, covariance.flatten(), track['age']]))
        track['presence_logit'] = self.presence(track['token']).squeeze(-1)
        return track

    def appearance_tokens(self, output, q, dino, vae, camera, time, weight):
        mask = output['M'][q]
        yy, xx = torch.meshgrid(torch.arange(24, device=mask.device), torch.arange(36, device=mask.device), indexing='ij')
        grid = torch.stack([((xx+.5)*20+8)/736*2-1, (yy+.5)*20/480*2-1], -1).float()
        latent = F.grid_sample(vae[None], grid[None], mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        features = torch.cat([dino.reshape(24, 36, 768), latent], -1)
        cx = (mask*xx).sum()/mask.sum().clamp_min(1e-8)
        cy = (mask*yy).sum()/mask.sum().clamp_min(1e-8)
        pooled = []
        for lower_y, lower_x in [(True, True), (True, False), (False, True), (False, False)]:
            part = mask*(yy <= cy if lower_y else yy > cy)*(xx <= cx if lower_x else xx > cx)
            pooled.append((features*part[..., None]).sum((0, 1))/part.sum().clamp_min(1e-8))
        direction = F.normalize(output['center'][q]-camera[:3, 3], dim=-1)
        metadata = torch.cat([direction, time.reshape(1), weight.reshape(1)])[None].expand(4, -1)
        return self.appearance(torch.cat([torch.stack(pooled), metadata], -1))

    def forward(self, observations, pair_head, dino, vae, cameras, times):
        tracks, trace = [], []
        for t, pair in enumerate(observations):
            if t:
                tracks = [self.predict(track, float(times[t]-times[t-1])) for track in tracks]
            prior = tracks
            selected, outputs = [], []
            for v in range(2):
                output = {key: value[v] for key, value in pair.items()}
                confidence = output['class_logits'].softmax(-1)[:, 1:].amax(-1)
                ids = torch.where(confidence.detach() > .5)[0]
                selected.append(ids)
                outputs.append({key: output[key][ids] for key in ('tokens', 'center', 'covariance', 'class_logits')})
            assignment, scores, used = [], [], [set(), set()]
            gathered = [[] for _ in tracks]
            def evidence(view, query, alpha):
                out = {key: value[view] for key, value in pair.items()}
                visibility = out['M'][query].sum()/out['P'][query].sum().clamp_min(1e-8)
                reliability = self.reliability(out['tokens'][query]).sigmoid().squeeze(-1)
                weight = alpha*visibility*reliability
                return dict(center=out['center'][query], covariance=out['covariance'][query], token=out['tokens'][query],
                            class_prob=out['class_logits'][query].softmax(-1), dimensions=out['dimensions'][query],
                            heading=out['heading'][query], weight=weight, reliability=reliability, view=view, query=query,
                            confidence=float(out['class_logits'][query].softmax(-1)[1:].max().detach()),
                            appearance=self.appearance_tokens(out, query, dino[t, view], vae[t, view], cameras[t, view], times[t], weight))
            for v in range(2):
                score = self.probabilities(prior, outputs[v])
                scores.append(score)
                matches = hungarian_association(score)
                assignment.append(matches)
                for k, j in matches:
                    used[v].add(j)
                    gathered[k].append(evidence(v, int(selected[v][j]), score[k, j]))
            tracks = [self.update(dict(track), gathered[k]) for k, track in enumerate(tracks)]
            unmatched = [[j for j in range(len(selected[v])) if j not in used[v]] for v in range(2)]
            new_pairs = []
            if unmatched[0] and unmatched[1]:
                left = {k: v[unmatched[0]] for k, v in outputs[0].items()}
                right = {k: v[unmatched[1]] for k, v in outputs[1].items()}
                probability = pair_head.association_logits(left, right).sigmoid()
                for a, g in hungarian_association(probability):
                    ja, jg = unmatched[0][a], unmatched[1][g]
                    used[0].add(ja); used[1].add(jg)
                    new_pairs.append([evidence(0, int(selected[0][ja]), probability[a, g]), evidence(1, int(selected[1][jg]), probability[a, g])])
            for v in range(2):
                for j in unmatched[v]:
                    if j not in used[v]:
                        new_pairs.append([evidence(v, int(selected[v][j]), times.new_tensor(1.))])
            for items in new_pairs:
                if float(torch.stack([e['weight'] for e in items]).sum().detach()) <= 0:
                    continue
                strongest = max(items, key=lambda e:e['confidence'])
                track = dict(birth=(t, strongest['view'], strongest['query']), appearance=[])
                tracks.append(self.update(track, items, initial=True))
            tracks = sorted(tracks, key=lambda track:track['score'], reverse=True)[:64]
            trace.append(dict(prior=prior, tracks=tracks, selected=selected, scores=scores, evidence=gathered))
        if tracks:
            tokens = torch.stack([track['token'] for track in tracks])
            initial = torch.stack([torch.cat([track['mean'], torch.atan2(track['heading'][0], track['heading'][1]).reshape(1),
                                             times.new_zeros(1), track['dimensions']]) for track in tracks])
        else:
            tokens, initial = times.new_empty((0, 256)), times.new_empty((0, 11))
        return dict(tracks=tracks, trace=trace, tokens=tokens, initial_state=initial)


def historical_assignment(observations, labels):
    """Supervision map built AFTER inference. IDs are not inputs to tracking."""
    assignments = []
    for t, pair in enumerate(observations):
        views = []
        for v in range(2):
            mapping = torch.full((64,), -1, device=pair['center'].device, dtype=torch.long)
            ids = torch.where(labels['history_detectable'][t, v])[0]
            if len(ids):
                p = pair['class_logits'][v].softmax(-1)
                cost = torch.cdist(pair['center'][v].detach(), labels['history_state'][t, ids, :3], p=1)/3-p[:, labels['class_id'][ids]].detach()
                a, b = linear_sum_assignment(cost.cpu().numpy())
                mapping[a] = ids[b]
            views.append(mapping)
        assignments.append(views)
    return assignments


def supervised_loss(model, result, observations, labels):
    mapping = historical_assignment(observations, labels)
    zero = result['tokens'].sum()*0
    losses = {name: [] for name in ('association', 'posterior', 'state_readout', 'presence', 'reliability')}
    identity_features, appearance_features, identity_ids = [], [], []
    for t, event in enumerate(result['trace']):
        def label(track):
            time, view, query = track['birth']
            return int(mapping[time][view][query])
        for v in range(2):
            if event['prior'] and len(event['selected'][v]):
                left = torch.tensor([label(track) for track in event['prior']], device=zero.device)
                right = mapping[t][v][event['selected'][v]]
                truth = (left[:, None] == right[None])&(left[:, None] >= 0)
                losses['association'].append(F.binary_cross_entropy(event['scores'][v].clamp(1e-6, 1-1e-6), truth.float()))
            for query in event['selected'][v]:
                k = int(mapping[t][v][query])
                token = observations[t]['tokens'][v, query]
                if k >= 0:
                    error = (observations[t]['center'][v, query]-labels['history_state'][t, k, :3])/model.scale.std[:3]
                    target = torch.exp(-error.square().mean()).detach()
                    identity_features.append(token); identity_ids.append(k)
                else:
                    target = zero.detach()
                losses['reliability'].append((model.reliability(token).sigmoid().squeeze()-target).square())
        for track in event['tracks']:
            k = label(track)
            losses['presence'].append(F.binary_cross_entropy_with_logits(track['presence_logit'], zero.new_tensor(float(k >= 0))))
            if k < 0:
                continue
            ref = labels['history_state'][t, k]
            scale = model.scale.std[:6]
            losses['posterior'].append(gaussian_loss((track['mean']-ref[:6])/scale, track['covariance']/scale[:, None]/scale[None]))
            out = model.readout(track['token'])
            state_target = (ref[:6]-model.scale.mean[:6])/scale
            losses['state_readout'].append(F.cross_entropy(out[:5][None], labels['class_id'][k:k+1])+
                F.l1_loss(out[5:8], ref[8:11].clamp_min(.01).log())+F.l1_loss(out[8:], state_target))
    identity = contrastive(torch.stack(identity_features), torch.tensor(identity_ids, device=zero.device)) if identity_features else zero
    for track in result['tracks']:
        t, v, q = track['birth']; k = int(mapping[t][v][q])
        if k >= 0:
            appearance_features.extend([(tokens.mean(0), k) for tokens in track['appearance']])
    appearance = contrastive(torch.stack([x for x, _ in appearance_features]), torch.tensor([k for _, k in appearance_features], device=zero.device)) if appearance_features else zero
    terms = {name: torch.stack(items).mean() if items else zero for name, items in losses.items()}
    terms.update(identity=identity, appearance_identity=appearance)
    reference_ids = torch.tensor([int(mapping[t][v][q]) for t, v, q in [track['birth'] for track in result['tracks']]], device=zero.device, dtype=torch.long)
    return sum(terms.values()), terms, reference_ids
