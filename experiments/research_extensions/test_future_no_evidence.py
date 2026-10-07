"""CPU proof of the single loss-factor change and the separate typed gates."""
import ast
import copy
import inspect
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from . import train_history_future as old
from . import train_future_no_evidence as new
from . import typed_state_test as old_test
from . import typed_no_evidence_test as test


class NoEvidenceTests(unittest.TestCase):
    def test_all_numerical_helpers_are_unchanged(self):
        for name in ('free_mode_means','project_centers','free_motion_projection_loss',
                     'evidence_pair_loss','geometry_surrogate','prepare_row','future_tensors'):
            a=ast.dump(ast.parse(inspect.getsource(getattr(old,name))),include_attributes=False)
            b=ast.dump(ast.parse(inspect.getsource(getattr(new,name))),include_attributes=False)
            self.assertEqual(a,b,name)

    def test_training_loop_only_changes_evidence_coefficient_and_metric(self):
        def loop(module):
            s=inspect.getsource(module.run)
            return s[s.index('        while step < args.steps'):s.index("        if status.get('checkpoint_step')")]
        changed=loop(new).replace('                    # Evaluate the identical evidence branch/RNG stream, but contribute zero gradient.\n','')
        changed=changed.replace('(0.*evidence).backward(); total += 0.*float(evidence.detach())',
                                '(.1*evidence).backward(); total += .1*float(evidence.detach())')
        changed=changed.replace('metrics.update(more, evidence_loss=float(evidence.detach()), evidence_weight=0.)',
                                'metrics.update(more, evidence_loss=float(evidence.detach()))')
        self.assertEqual(changed,loop(old))

    def test_evidence_draws_and_rng_retained_but_gradient_zero(self):
        import experiments.base48.model as base
        import experiments.paired_interventions.objectives as objectives
        model=nn.Linear(1,1,bias=False);model.weight.data.fill_(.5)
        from types import SimpleNamespace
        model.scale=SimpleNamespace(std=torch.ones(6))
        calls=[]
        def rollout(_model,_tokens,_shared,initial,_presence,generator,_slot,**kwargs):
            value=torch.rand((),generator=generator);calls.append(float(value))
            return dict(state=model.weight.reshape(1,1,1,1,1).expand(1,1,1,1,6)*value,
                        mode_logits=torch.zeros(1,1),log_survival=torch.zeros(1,1))
        def density(state,_logits,_ref,*_args):
            v=state.sum();return torch.stack((v,-v))[None]
        history=dict(empty=False,tokens=None,shared=None,initial=torch.ones(1,1,11),
                     presence_logits=None,covariance=None)
        rows=[dict(history=history,focal_slot=0),dict(history=history,focal_slot=0)]
        with patch.object(base,'persistent_target_rollouts',rollout),patch.object(objectives,'candidate_log_density',density):
            global_before=torch.get_rng_state().clone()
            first,_=old.evidence_pair_loss(model,rows,torch.zeros(2,16,6),7,17029)
            old_calls=calls[:];calls.clear();(.1*first).backward();old_grad=model.weight.grad.clone()
            model.zero_grad();second,_=new.evidence_pair_loss(model,rows,torch.zeros(2,16,6),7,17029)
            (0.*second).backward()
            self.assertEqual(old_calls,calls);self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
            self.assertTrue(torch.equal(first,second));self.assertGreater(float(old_grad.abs().sum()),0)
            self.assertEqual(float(model.weight.grad.abs().sum()),0)

    def payload(self):
        return dict(schema=new.SCHEMA,parent=dict(sha256='p'),step=2048,target_steps=2048,completed=True,
            history_checkpoint=dict(path='history',sha256='h'),future_overlay={'dynamics.w':torch.ones(1)},
            identity=dict(configuration=dict(arm='paired_geometry',component_ablation='no_evidence',
                evidence_forward_retained=True,stochastic_order_unchanged=True,seed=17029,steps=2048,lr=1e-5,
                weights=dict(evidence=0.,motion_projection=.5,common_geometry=.1,reappearance_geometry=.1),
                history_frozen=True,future_truth_conditions=False,deployment_sampler_changed=False,
                ambiguous_nll_and_presence_only=True),data=dict(split='train',test_used=False,
                selection_uses_validation_scores=False,index_sha256='i',branches=[dict(split='train',camera=0)])))

    def test_new_checkpoint_schema_seed_weight_and_partial_gates(self):
        with patch.object(new,'sha256',side_effect=lambda p:'h'if p=='history'else'p'):
            p=self.payload();new.validate_checkpoint(p,'parent','i',True)
            for mode in ('oldschema','seed','evidence','partial','future'):
                q=copy.deepcopy(p)
                if mode=='oldschema':q['schema']=old.SCHEMA
                elif mode=='seed':q['identity']['configuration']['seed']=17043
                elif mode=='evidence':q['identity']['configuration']['weights']['evidence']=.1
                elif mode=='partial':q.update(step=2000,completed=False)
                else:q['identity']['configuration']['future_truth_conditions']=True
                with self.assertRaises(ValueError):new.validate_checkpoint(q,'parent','i',True)

    def test_exact_same_parent_initial_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'parent.pt';torch.save(dict(model={
                'upstream.dynamics.w':torch.arange(3.),'upstream.geometry.w':torch.ones(2)}),parent)
            p=dict(future_overlay={'dynamics.w':torch.arange(3.),'geometry.w':torch.ones(2)})
            new.assert_parent_initialization(p,parent,root/'initialization.json')
            p['future_overlay']['dynamics.w'][0]=3
            with self.assertRaises(ValueError):new.assert_parent_initialization(p,parent,root/'wrong.json')

    def test_typed_formal_entry_keeps160_and_original_eight_draws(self):
        lock=dict(schema=test.SCHEMA+'-lock',split='test',sample_seeds=test.SEEDS,
            sampling_stream_training_seed=17,camera=0,worlds=[0,1],selection_uses_test=False,
            density_draws_per_seed=4,density_bandwidth=.2,family_ids=['f%03d'%i for i in range(160)],
            checkpoints={test.ARMS[0]:dict(steps=2048,arm=test.ARMS[0],sha256='a'*64,history_sha256='b'*64)})
        test.validate_lock(lock)
        self.assertEqual(test.SEEDS,old_test.SEEDS)
        with self.assertRaises(ValueError):old_test.validate_lock(lock)
        wrong=copy.deepcopy(lock);wrong['family_ids']=wrong['family_ids'][:20]
        with self.assertRaises(ValueError):test.validate_lock(wrong)
        wrong=copy.deepcopy(lock);wrong['sample_seeds'][0]=999
        with self.assertRaises(ValueError):test.validate_lock(wrong)


if __name__=='__main__':unittest.main()
