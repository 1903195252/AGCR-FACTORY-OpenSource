"""Execution-only extension of the unchanged original StageTwo objectives.

The eight-view geometry forward retains two-view loss normalization. Cached
history is an autograd graph from the same inputs, never a detached feature.
"""
import torch
from experiments.paper_experiments.agcr_exp.stage2_system import (StageTwo,
    observation_target,matched_state_supervision,geometry_intrinsics,
    geometry_loss,observation_loss,supervised_loss)


class RuntimeStageTwo(StageTwo):
    def objective(self, sample, encoded, auxiliary_features, geometry_chunk=2, timing=None, history_result=None):
        import time
        ticks = [time.monotonic()]
        def mark(name):
            if timing is not None:
                if encoded[0].is_cuda:
                    torch.cuda.synchronize()
                now = time.monotonic()
                timing[name] = now-ticks[0]
                ticks[0] = now
        inputs, labels = sample['inputs'], sample['labels']
        result, observations, shared, separate = self.history(inputs, encoded) if history_result is None else history_result
        mark('history')
        zero = shared.sum()*0
        terms = {}
        # Phase one supplied framewise detection/paired association only. Finish
        # the historical frontend here while retaining its original supervision.
        obs_loss = []
        for t in range(6):
            target = observation_target(labels['history_state'][t], labels['class_id'], labels['history_full_mask'][t],
                                        labels['history_visible_mask'][t], labels['history_detectable'][t])
            obs_loss.append(observation_loss(self.observation, observations[t], target)[0])
        observation = torch.stack(obs_loss).mean()
        tracking, tracking_parts, reference_ids = supervised_loss(self.tracker, result, observations, labels)
        terms.update(observation=observation, **tracking_parts)
        mark('history_supervision')
        known, ids, eligible, previous_reference, reference, presence = matched_state_supervision(result['initial_state'], reference_ids, labels)
        dynamics, dynamics_parts, rollout = self.dynamics.objective(result['tokens'], shared, result['initial_state'],
            previous_reference, reference, eligible, presence)
        terms.update(dynamics_parts)
        mark('dynamics')
        cameras = inputs['planned_camera_to_world'].flatten(0, 1)
        K = geometry_intrinsics(inputs['K'])
        with torch.no_grad():
            voxels, projection = self.project_history(inputs, observations, K, cameras)
        mark('static_projection')
        # Training uses matched reference FUTURE STATES, explicitly separated
        # from history construction; deployment instead passes the sampled state.
        states = torch.where(known[None, :, None], reference, result['initial_state'][None].expand(16, -1, -1))
        states = states[:, None].expand(-1, 2, -1, -1).flatten(0, 1)
        exists = presence[:, None].expand(-1, 2, -1).flatten(0, 1)
        geometry_labels = {}
        for key in ('isolated_depth_mm', 'full_mask', 'visible_mask'):
            value = labels[key][:, :, ids]
            value = value*known[None, None, :, None, None]
            geometry_labels[key] = value.flatten(0, 1)
        geometry_labels.update(static_depth_mm=labels['static_depth_mm'].flatten(0, 1), static_support=labels['static_support'].flatten(0, 1))
        geometry_total = zero
        for start in range(0, 32, geometry_chunk):
            end = min(32, start+geometry_chunk)
            prediction = self.geometry(shared, separate, result['tokens'], states[start:end], exists[start:end], projection[start:end], K,
                cameras[start:end], inputs['future_time_s'].repeat_interleave(2)[start:end], torch.arange(start, end, device=K.device)%2)
            # Batch the network, retain the original per-pair denominators.
            # Averaging each eight-view mask globally changes the objective
            # when foreground/valid-depth coverage differs between frames.
            for pair_start in range(start, end, 2):
                offset = pair_start-start
                pair_prediction = {key:value[offset:offset+2] for key,value in prediction.items()}
                loss, parts = geometry_loss(pair_prediction, {key:value[pair_start:pair_start+2] for key,value in geometry_labels.items()})
                geometry_total = geometry_total+loss*2/32
                for key, value in parts.items():
                    terms[key] = terms.get(key, zero)+value*2/32
        mark('future_surfaces')
        reader, reader_parts = self.auxiliary_loss(sample['auxiliary'], auxiliary_features)
        terms.update(reader_parts)
        mark('auxiliary_reader')
        loss = observation+tracking+dynamics+geometry_total+reader
        diagnostics = dict(tracks=len(reference_ids), matched_tracks=int(eligible.sum()),
                           eligible_reference_tracks=int(labels['history_eligible'].sum()), static_voxels=len(voxels))
        if eligible.any():
            diagnostics['anchor_position_error_m'] = float((result['initial_state'][eligible, :3]-labels['history_state'][-1, ids[eligible], :3]).norm(dim=-1).mean().detach())
            diagnostics['mode_mean_ADE_m'] = float((rollout[:, :, eligible, :3]-reference[None, :, eligible, :3]).norm(dim=-1).mean().detach())
        return loss, terms, diagnostics

    def history(self, inputs, encoded):
        available=inputs.get('observation_available')
        if available is None:return super().history(inputs,encoded)
        if available.shape!=(6,2) or available.dtype!=torch.bool:
            raise ValueError('Observation availability must be Boolean [6,2]')
        clean={k:v for k,v in inputs.items() if k!='observation_available'}
        if bool(available.all()):return super().history(clean,encoded)
        expected={'rgb','depth_mm','depth_valid','K','camera_to_world','planned_camera_to_world','time_s','future_time_s'}
        if set(clean)!=expected or bool((clean['time_s']>0).any()):raise ValueError('Historical inputs only')
        dino,vae=encoded
        pair=self.observation(dino.flatten(0,1),clean['depth_mm'].flatten(0,1),clean['depth_valid'].flatten(0,1),
            clean['K'][None].expand(12,-1,-1),clean['camera_to_world'].flatten(0,1),clean['time_s'].repeat_interleave(2),
            torch.arange(2,device=dino.device).repeat(6))
        flat=available.flatten()
        # Absence is not an image. Remove its queries and background memory,
        # including biases in the feature encoder of the black placeholder.
        for key,value in pair.items():
            if key=='class_logits':
                missing=torch.full_like(value,-40);missing[...,0]=40
            else:missing=torch.zeros_like(value)
            pair[key]=torch.where(flat.reshape(12,*([1]*(value.ndim-1))),value,missing)
        observations=[{key:value[2*t:2*t+2] for key,value in pair.items()} for t in range(6)]
        result=self.tracker(observations,self.observation,dino,vae,clean['camera_to_world'],clean['time_s'])
        features=torch.stack([out['scene_features'] for out in observations])
        union=torch.stack([1-(1-out['M']*(out['class_logits'].softmax(-1)[...,1:].amax(-1)>.5)[...,None,None]).prod(1) for out in observations]).flatten(-2)
        background=features*(1-union[...,None])
        memory=background[available].reshape(1,-1,256)
        if memory.shape[1]==0:memory=background.new_zeros(1,1,256)
        shared=self.scene.shared[None];separate=[]
        for block in self.scene.blocks:shared=block(shared,memory)
        for view in range(2):
            per=background[:,view][available[:,view]].reshape(1,-1,256)
            if per.shape[1]==0:per=background.new_zeros(1,1,256)
            query=self.scene.per_view[view:view+1]
            for block in self.scene.blocks:query=block(query,per)
            separate.append(query[0])
        return result,observations,shared,torch.stack(separate)

    def project_history(self,inputs,observations,K,cameras):
        if 'observation_available' in inputs:
            inputs=dict(inputs,depth_valid=inputs['depth_valid']&inputs['observation_available'][...,None,None])
        return super().project_history(inputs,observations,K,cameras)
