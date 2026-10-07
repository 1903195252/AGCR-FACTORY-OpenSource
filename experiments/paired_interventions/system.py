"""Executable objectives for upstream, reader, depth warmup and paired video.

All intervention generation is conditioned on an inferred historical state.
Future references enter only the explicit replay/supervision functions.
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import replay_sample,historical_labels
from .world import load_upstream,HistoryQueryReader,predict_world,world_conditions,query_labels,read_generated
from .video_modules import ConditionEncoder,PairedWan,NativeVAE,RGBDDepthDecoder
from .video_losses import (depth_loss,reference_for_queries,readout_loss,reappearance_window,
    paired_readout_loss,surface_loss,correspondence_loss,flow_weights)
from .objectives import camera_readout_consistency,stochastic_feedback_surrogate
from .bridge import evidence_pair_objective
from .sampling import pure_noise_euler
from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders


class JointSystem(nn.Module):
    def __init__(self,paths,device,phase,variant):
        super().__init__()
        self.upstream,self.parent_step=load_upstream(paths['upstream'],paths['statistics'],device)
        self.reader=HistoryQueryReader(self.upstream.reader).to(device)
        self.encoders=FrozenImageEncoders(paths['dino_code'],paths['dino_weights'],paths['wan']+'/vae',device)
        self.vae=NativeVAE(self.encoders)
        self.condition=ConditionEncoder().to(device)
        self.depth=RGBDDepthDecoder().to(device)
        self.wan=None
        if phase=='video':
            self.wan=PairedWan(paths['wan'],paths['empty_text'],device)
            self.wan.encoder=self.condition
            self.wan.ordered=variant['ordered']
        self.phase,self.variant=phase,variant
        self.feedback_gradient_norms={}
        self.set_phase(phase,False)

    def set_phase(self,phase,feedback_phase):
        # DDP registers all parameters that can later receive gradients at
        # construction. Prefix freezing is implemented by no_grad in prepared;
        # no optimizer step/weight decay occurs while their grad is None.
        self.upstream.requires_grad_(phase in ('dynamics','video'))
        self.reader.requires_grad_(phase in ('dynamics','reader'))
        self.condition.requires_grad_(phase in ('depth','video'))
        self.depth.requires_grad_(phase in ('depth','video'))
        self.upstream.train(phase=='dynamics' or feedback_phase)
        self.reader.train(phase in ('reader','dynamics'))
        if self.wan is not None: self.wan.train()

    def trainable_groups(self):
        # Include upstream parameters now, even while phase-one video freezes
        # them, so unfreezing preserves optimizer state and DDP registration.
        groups={'upstream':[],'lora':[],'new':[]}
        for name,p in self.named_parameters():
            if name.startswith('upstream.'):
                if self.phase in ('video','dynamics'): groups['upstream'].append(p)
            elif 'lora_' in name: groups['lora'].append(p)
            elif p.requires_grad: groups['new'].append(p)
        return [dict(params=p,lr=1e-4 if k=='new' else 1e-5,name=k) for k,p in groups.items() if p]

    def prepared(self,branch,generator,grad,build_condition=True):
        x=branch['inputs'];encoded=self.encoders.history(x['rgb'])
        with nullcontext() if grad else torch.no_grad():
            history,draw=predict_world(self.upstream,x,encoded,generator,draw_grad=build_condition)
            condition=world_conditions(self.upstream,history,draw,x,self.variant.get('detach_decoded_feedback_state',False)) if build_condition else None
        return encoded,history,draw,condition

    def decode(self,z,c):
        rgb,depth,log,valid=[],[],[],[]
        geometry=self.condition.depth_features(c)
        for view in range(2):
            def one(latent,geom,view=view):
                image,*features=self.vae.decode(latent.float())
                with torch.autocast('cuda',dtype=torch.bfloat16): d=self.depth(latent.float(),features,geom,view)
                return image[0,:,1:,:,8:-8].permute(1,0,2,3),d['depth_mm'][0,1:,:,8:-8].float(),d['log_depth'][0,1:,:,8:-8].float(),d['validity_logits'][0,1:,:,8:-8].float()
            if torch.is_grad_enabled(): values=checkpoint(one,z[view:view+1],geometry[view:view+1],use_reentrant=False)
            else: values=one(z[view:view+1],geometry[view:view+1])
            for destination,value in zip((rgb,depth,log,valid),values): destination.append(value)
        return dict(rgb=torch.stack(rgb,1),depth_mm=torch.stack(depth,1),log_depth=torch.stack(log,1),validity_logits=torch.stack(valid,1))

    def reference_condition(self,branch,history,ids):
        """Base rendering supervision only. Feedback never calls this path."""
        y=branch['supervision'];known=ids>=0;safe=ids.clamp_min(0)
        state=history['result']['initial_state'][None].expand(16,-1,-1)
        state=torch.where(known[None,:,None],y['state'][:,safe],state)
        present=y['present'][:,safe]&known[None]
        draw=dict(state=state[None],present=present[None])
        with torch.no_grad(): return world_conditions(self.upstream,history,draw,branch['inputs'])

    def reader_objective(self,branch,history,ids,offset=0):
        y=branch['supervision'];x=branch['inputs']
        reference=reference_for_queries(branch,ids)
        descriptors=torch.stack([t['descriptor'] for t in history['result']['tracks']]) if len(ids) else history['shared'].new_empty(0,256)
        feat=self.encoders.features(y['rgb'][offset])
        out=self.reader(feat,y['depth_mm'][offset],y['valid'][offset],x['K'][None].expand(2,-1,-1),
            x['planned_camera_to_world'][offset],x['future_time_s'][offset].expand(2),
            torch.arange(2,device=feat.device),descriptors.detach())
        out={k:v[None] for k,v in out.items()}
        ref={k:(v[offset:offset+1] if k!='known' else v) for k,v in reference.items()}
        a,b=readout_loss(out,ref,descriptors,self.upstream.tracker.scale.std[:3])
        return a+b

    def video_readout(self,decoded,c,x):
        return read_generated(self.reader,self.encoders.dino,decoded['rgb'],decoded['depth_mm'],
            decoded['validity_logits'].sigmoid(),x,c['identity'])

    def base_video(self,branch,c,z0,generator,history,ids):
        s=torch.randn((),device=z0.device,generator=generator).sigmoid();s=5*s/(1+4*s)
        noise=torch.randn(z0[:, :, 1:].shape,device=z0.device,generator=generator)
        def corrupt(time): return torch.cat([z0[:,:,:1],(1-time)*z0[:,:,1:]+time*noise],2)
        prediction=self.wan(corrupt(s),s,c)
        weight,domain=flow_weights(branch)
        flow=((prediction[:,:,1:]-(noise-z0[:,:,1:])).square()*weight).sum()/(domain.sum()*48).clamp_min(1)
        low=.05+.25*torch.rand((),device=z0.device,generator=generator)
        corrupted=corrupt(low);velocity=self.wan(corrupted,low,c)
        clean=torch.cat([z0[:,:,:1],corrupted[:,:,1:]-low*velocity[:,:,1:]],2)
        out=self.decode(clean,c);y=branch['supervision']
        rgb=F.l1_loss(out['rgb'],y['rgb'].permute(0,1,4,2,3).float()/127.5-1)
        d=depth_loss(out['log_depth'],out['validity_logits'],y['depth_mm'],y['valid'])
        reader=self.video_readout(out,c,branch['inputs'])
        reference=reference_for_queries(branch,ids);scales=self.upstream.tracker.scale.std[:3]
        ground,identity=readout_loss(reader,reference,c['identity'],scales)
        paired=paired_readout_loss(reader,reference,scales)
        surface=surface_loss(out['depth_mm'],c)
        corr=correspondence_loss(out['depth_mm'],branch,scales)
        loss=flow+.1*rgb+d+.1*ground+.05*identity+.1*paired+.1*surface+.1*corr
        return loss,dict(flow=flow,rgb_l1=rgb,depth=d,ground=ground,identity=identity,paired=paired,surface=surface,correspondence=corr)

    def evidence(self,family,generator):
        zero=self.upstream.tracker.scale.std.sum()*0
        if family['metadata']['cohort']!='evidence':
            return zero,dict(evidence_covered=0,evidence_total=0)
        inputs=[];enc=[];matched=[];paths=[]
        focal=family['metadata']['target_index']
        for world in range(2):
            branch=family['branches'][(world,0)];x=branch['inputs']
            encoded=self.encoders.history(x['rgb'])
            # Matching uses historical labels only, after inference.
            with torch.no_grad():
                h,_=predict_world(self.upstream,x,encoded,generator)
                ids=query_labels(h,historical_labels(branch))
            where=torch.where(ids==focal)[0]
            matched.append(int(where[0]) if len(where) else -1)
            inputs.append(x);enc.append(encoded);paths.append(branch['supervision']['state'][:,focal,:6])
        result=evidence_pair_objective(self.upstream,inputs,enc,matched,torch.stack(paths),
            self.upstream.tracker.scale.std[:6],generator)
        return result['loss']*float(self.variant['evidence']),dict(evidence_covered=result['covered_worlds'],evidence_total=2,
            raw_evidence_loss=result['loss'].detach())

    def feedback(self,family,generator,world,baseline):
        a=family['branches'][(world,0)];b=family['branches'][(world,1)]
        _,history,draw,ca=self.prepared(a,generator,True)
        cb=world_conditions(self.upstream,history,draw,b['inputs'],self.variant.get('detach_decoded_feedback_state',False))
        # These tensors belong only to this inferred feedback draw, not the
        # independent evidence objective or supervised replay. Their hooks
        # establish whether decoded errors actually reach the historical path.
        self.feedback_gradient_norms={}
        watched=dict(predicted_future_state=draw['state'],history_scene=history['shared'],
            history_track_tokens=history['result']['tokens'],history_initial_state=history['result']['initial_state'])
        for name,tensor in watched.items():
            if tensor.requires_grad:
                def record(gradient,name=name):
                    self.feedback_gradient_norms[name]=self.feedback_gradient_norms.get(name,0.)+float(gradient.detach().float().norm())
                tensor.register_hook(record)
        if self.variant.get('detach_decoded_feedback_state',False):
            ca={k:v.detach() for k,v in ca.items()};cb={k:v.detach() for k,v in cb.items()}
        ids=query_labels(history,historical_labels(a))
        # Single-image anchor encoding never sees a future reference image.
        anchor=self.encoders.history(a['inputs']['rgb'])[1][-1,:, :,None]
        # [2,48,1,30,46] -> sampler time-major convention.
        anchor=anchor.permute(0,2,1,3,4)
        noise=torch.randn(2,4,48,30,46,device=anchor.device,generator=generator)
        readouts=[];references=[];feedback=[]
        for branch,c in [(a,ca),(b,cb)]:
            def generated_readout(condition,inputs):
                def velocity(z,s,condition): return self.wan(z.permute(0,2,1,3,4),s,condition).permute(0,2,1,3,4)
                latent=pure_noise_euler(velocity,noise,anchor,condition,steps=50,grad_tail=2).permute(0,2,1,3,4)
                decoded=self.decode(latent,condition)
                return self.video_readout(decoded,condition,inputs)
            # Retain dense readouts rather than two full denoising/decoder
            # graphs simultaneously. Recompute uses the identical pure noise.
            out=checkpoint(generated_readout,c,branch['inputs'],use_reentrant=False)
            reference=reference_for_queries(branch,ids)
            ground,identity=readout_loss(out,reference,c['identity'],self.upstream.tracker.scale.std[:3],
                event=reappearance_window(reference['visible']),use_dice=False)
            # Ambiguous futures are not regressed to an arbitrarily chosen branch.
            feedback.append((ground+identity)*float(family['metadata']['cohort']=='evidence'))
            readouts.append(out);references.append(reference)
        centers=torch.stack([o['center'].permute(1,0,2,3) for o in readouts])[None]
        desc=torch.stack([o['descriptor'].permute(1,0,2,3) for o in readouts])[None]
        visible=torch.stack([r['visible'].permute(1,0,2) for r in references])[None]
        camera=camera_readout_consistency(centers,desc,visible,self.upstream.tracker.scale.std[:3])
        reappear=torch.stack(feedback).mean()
        physical=.1*reappear+.1*camera*float(self.variant['camera'])
        log_probability=draw['discrete_log_prob']
        if self.variant.get('detach_decoded_feedback_state',False): log_probability=log_probability.detach()
        value=stochastic_feedback_surrogate(physical[None],log_probability,baseline)
        return value,dict(camera=camera,reappearance=reappear,baseline_observation=physical.detach(),
            feedback_tracks=len(ids),matched_feedback_tracks=int((ids>=0).sum()))

    def forward(self,branch,generator,update=0,family=None,feedback=False,baseline=0.):
        self.feedback_gradient_norms={}
        phase=self.phase
        # This base pass needs historical queries only. Its sampled geometry
        # was discarded and rebuilt from reference states below. Keep the
        # sampled draw to preserve the exact RNG stream, but skip unused
        # geometry and its graph. Upstream training uses objective/feedback.
        encoded,history,draw,c=self.prepared(branch,generator,phase=='dynamics',build_condition=False)
        replay=replay_sample(branch,update%16) if phase=='dynamics' or feedback else None
        ids=query_labels(history,historical_labels(branch))
        stats=dict(predicted_tracks=len(ids),matched_tracks=int((ids>=0).sum()))
        if phase=='reader':
            loss=self.reader_objective(branch,history,ids,update%16)
            return loss,dict(stats,reader=loss)
        if phase=='dynamics':
            auxiliary=self.encoders.features(replay['auxiliary']['rgb'])
            cached=(history['result'],history['observations'],history['shared'],history['separate'])
            loss,parts,_=self.upstream.objective(replay,encoded,auxiliary,geometry_chunk=8,history_result=cached)
            if branch.get('cohort')=='ambiguous' and self.variant['ambiguous_nll_only']:
                loss=loss-.1*parts['rollout_l1']
            reader=self.reader_objective(branch,history,ids,update%16)
            # Appendix D applies evidence preference in final video feedback,
            # after the base dynamics fit; it is not an early replacement for
            # the observation/dynamics likelihood.
            return loss+reader,dict(stats,**parts,dense_reader=reader)
        rgb=torch.cat([branch['inputs']['rgb'][-1:],branch['supervision']['rgb']],0)
        z0=self.vae.encode(rgb)
        c=self.reference_condition(branch,history,ids)
        if phase=='depth':
            out=self.decode(z0,c)
            loss=depth_loss(out['log_depth'],out['validity_logits'],branch['supervision']['depth_mm'],branch['supervision']['valid'])
            return loss,dict(stats,depth=loss)
        seed=int(torch.randint(2**31-1,(),device=z0.device,generator=generator))
        def base_objective(branch,condition,latent,history,ids):
            local=torch.Generator(device=z0.device).manual_seed(seed)
            return self.base_video(branch,condition,latent,local,history,ids)
        # Save the objective boundary, not both full RGB-D graphs. The explicit
        # local generator makes recomputation independent of later RNG draws.
        loss,parts=checkpoint(base_objective,branch,c,z0,history,ids,use_reentrant=False)
        if feedback:
            if family is None: raise ValueError('Feedback requires an audited whole family')
            extra,details=self.feedback(family,generator,(update//8)%2,baseline)
            e,coverage=self.evidence(family,generator)
            # Replay is intentionally supervised separately; it cannot supply
            # future-state conditions to the preceding pure-noise branch.
            auxiliary=self.encoders.features(replay['auxiliary']['rgb'])
            replay_loss,replay_parts,_=self.upstream.objective(replay,encoded,auxiliary,geometry_chunk=2)
            if branch.get('cohort')=='ambiguous' and self.variant['ambiguous_nll_only']:
                replay_loss=replay_loss-.1*replay_parts['rollout_l1']
            loss=loss+extra*float(self.variant['feedback'])+.1*e+replay_loss
            parts.update(details,**coverage,evidence=e,base_replay=replay_loss)
        return loss,dict(stats,**parts)
