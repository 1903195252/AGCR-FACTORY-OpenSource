"""CPU contract/gradient tests. Synthetic tensors are NOT benchmark evidence."""
from itertools import permutations
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .audit import audit_manifest
from .metrics import branch_diagnostics, evidence_effect, occlusion_violations, reappearance_timing, transition_errors
from .objectives import (OrderedAppearanceGate, candidate_log_density, evidence_preference,
                         camera_readout_consistency, reappearance_feedback, stochastic_feedback_surrogate,
                         dynamics_replay_loss)
from .rollout import persistent_mode_rollouts, sample_shared_future
from .sampling import pure_noise_euler


torch.set_num_threads(2)


class ObjectiveTests(unittest.TestCase):
    def test_ambiguous_replay_does_not_pull_every_mode_to_one_realization(self):
        nll=torch.tensor(2.,requires_grad=True); collapsed_mean=torch.tensor(3.,requires_grad=True)
        loss=dynamics_replay_loss(dict(dynamics_nll=nll,rollout_l1=collapsed_mean),ambiguous=True)
        loss.backward()
        self.assertEqual(nll.grad.item(),1)
        self.assertIsNone(collapsed_mean.grad)

    def test_evidence_prefers_supported_future_and_trains_mixture(self):
        paths=torch.tensor([[[[[0.]]],[[[2.]]]]],requires_grad=True)
        logits=torch.tensor([[2.,-2.]],requires_grad=True)
        targets=torch.tensor([[[[0.]],[[2.]]]],requires_grad=True)
        scores=candidate_log_density(paths,logits,targets,[1.],.2)
        correct=evidence_preference(scores,torch.tensor([0]))
        wrong=evidence_preference(scores,torch.tensor([1]))
        self.assertLess(correct.item(),wrong.item())
        wrong.backward()
        self.assertGreater(logits.grad.abs().sum().item(),0)
        self.assertIsNone(targets.grad)

    def test_survival_density_is_not_silently_renormalized(self):
        args=(torch.zeros(1,2,3,4,1),torch.zeros(1,2),torch.zeros(1,2,4,1),[1.],.2)
        a=candidate_log_density(*args)
        b=candidate_log_density(*args,log_survival=torch.full((1,2,3),-3.))
        torch.testing.assert_close(b,a-3)

    def test_camera_readout_uses_world_coordinates_and_reference_visibility(self):
        center=torch.zeros(1,2,2,3,1,3,requires_grad=True)
        identity=torch.ones(1,2,2,3,1,4)
        vis=torch.ones(1,2,2,3,1,dtype=torch.bool)
        self.assertEqual(camera_readout_consistency(center,identity,vis,torch.ones(3)).item(),0)
        shifted=center+torch.tensor([0.,1.]).reshape(1,2,1,1,1,1)
        loss=camera_readout_consistency(shifted,identity,vis,torch.ones(3)); loss.backward()
        self.assertGreater(loss.item(),0)
        self.assertGreater(center.grad.abs().sum().item(),0)

    def test_hidden_mask_error_is_supervised_but_hidden_center_is_not(self):
        center=torch.zeros(1,2,3,1,3,requires_grad=True)
        mask=torch.full((1,2,3,1,2,2),3.,requires_grad=True)
        identity=torch.ones(1,2,3,1,4)
        vis=torch.zeros(1,2,3,1,dtype=torch.bool)
        loss=reappearance_feedback(center,identity,mask,torch.ones_like(center),identity,
                                   torch.zeros_like(mask),vis,~vis,torch.ones(3))
        loss.backward()
        self.assertEqual(center.grad.abs().sum().item(),0)
        self.assertGreater(mask.grad.abs().sum().item(),0)

    def test_score_function_preserves_loss_value_and_updates_mode_probability(self):
        parameter=torch.tensor(1.,requires_grad=True)
        logits=torch.tensor([0.,0.],requires_grad=True)
        loss=(parameter-3).square()[None]
        surrogate=stochastic_feedback_surrogate(loss,logits.log_softmax(-1)[0:1],0.)
        torch.testing.assert_close(surrogate,loss.mean())
        surrogate.backward()
        self.assertNotEqual(parameter.grad.item(),0)
        self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_pure_noise_tail_updates_upstream_and_reclamps_anchor(self):
        parameter=torch.tensor(1.,requires_grad=True)
        calls=[]
        def velocity(z,s,c):
            calls.append(torch.is_grad_enabled())
            return torch.ones_like(z)*c['state']
        anchor=torch.ones(1,1,2)*7
        out=pure_noise_euler(velocity,torch.zeros(1,4,2),anchor,{'state':parameter},steps=5,grad_tail=2)
        out[:,1:].sum().backward()
        self.assertEqual(calls,[False,False,False,True,True])
        torch.testing.assert_close(out[:,:1],anchor)
        self.assertNotEqual(parameter.grad.item(),0)


class AppearanceGateTests(unittest.TestCase):
    def test_reversed_visibility_is_identical_before_mean_mixer(self):
        torch.manual_seed(17)
        gate = OrderedAppearanceGate(8)
        features = torch.ones(1, 4, 8)
        disappearing = torch.tensor([[1., 1., 0., 0.]])
        appearing = torch.tensor([[0., 0., 1., 1.]])
        first = gate.condition_slots(features, disappearing, ordered=False)
        second = gate.condition_slots(features, appearing, ordered=False)
        # Check the whole input, including the mask channel and time codes,
        # before a learned/zero output projection could conceal leakage.
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first[..., -1], torch.full((1, 4), .5))
        torch.testing.assert_close(gate(features, disappearing, False),
                                   gate(features, appearing, False), rtol=0, atol=0)
        self.assertFalse(torch.allclose(gate.condition_slots(features, disappearing),
                                        gate.condition_slots(features, appearing)))
        self.assertFalse(torch.allclose(gate(features, disappearing), gate(features, appearing)))
        gate(features, disappearing).square().sum().backward()
        self.assertGreater(gate.time.grad.abs().sum().item(), 0)

    def test_all_joint_slot_permutations_preserve_complete_mean_input(self):
        torch.manual_seed(23)
        gate = OrderedAppearanceGate(256).double()
        # Nonconstant features, fractional masks and multiple target/query
        # prefixes exercise more than the binary constant-feature example.
        features = torch.randn(2, 3, 4, 256, dtype=torch.float64)
        masks = torch.rand(2, 3, 4, dtype=torch.float64)
        with torch.no_grad():
            reference_slots = gate.condition_slots(features, masks, ordered=False)
            reference_output = gate(features, masks, ordered=False)
            self.assertEqual(reference_slots.shape, (2, 3, 4, 257))
            for order in permutations(range(4)):
                index = list(order)
                with self.subTest(order=order):
                    permuted_features = features[..., index, :]
                    permuted_masks = masks[..., index]
                    torch.testing.assert_close(
                        gate.condition_slots(permuted_features, permuted_masks, False),
                        reference_slots, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(
                        gate(permuted_features, permuted_masks, False),
                        reference_output, rtol=1e-12, atol=1e-12)

    def test_mean_control_gradients_follow_pairs_not_original_slot_indices(self):
        torch.manual_seed(29)
        gate = OrderedAppearanceGate(8).double()
        features = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
        masks = torch.rand(2, 4, dtype=torch.float64, requires_grad=True)
        readout = torch.randn(2, 8, dtype=torch.float64)
        parameters = tuple(gate.parameters())
        loss = (gate(features, masks, ordered=False) * readout).sum()
        original = torch.autograd.grad(loss, (features, masks) + parameters)
        index = [2, 0, 3, 1]
        permuted_features = features.detach()[..., index, :].requires_grad_()
        permuted_masks = masks.detach()[..., index].requires_grad_()
        loss = (gate(permuted_features, permuted_masks, False) * readout).sum()
        permuted = torch.autograd.grad(loss, (permuted_features, permuted_masks) + parameters)
        torch.testing.assert_close(permuted[0], original[0][..., index, :], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(permuted[1], original[1][..., index], rtol=1e-12, atol=1e-12)
        for first, second in zip(original[2:], permuted[2:]):
            torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)

    def test_mean_control_preserves_feature_visibility_pairing(self):
        gate = OrderedAppearanceGate(8)
        features = torch.arange(1., 5.).reshape(1, 4, 1).expand(1, 4, 8)
        # Permuting ONLY masks changes which features are visible. This is
        # not the joint-permutation invariance required of the mean control.
        a = gate.condition_slots(features, torch.tensor([[1., 1., 0., 0.]]), False)
        b = gate.condition_slots(features, torch.tensor([[0., 0., 1., 1.]]), False)
        torch.testing.assert_close(b[..., :-1] - a[..., :-1], torch.ones(1, 4, 8))
        torch.testing.assert_close(a[..., -1], b[..., -1])

    def test_all_hidden_inputs_inject_zero_in_both_variants(self):
        gate = OrderedAppearanceGate(8)
        features = torch.randn(2, 3, 4, 8)
        masks = torch.zeros(2, 3, 4)
        for ordered in (False, True):
            with self.subTest(ordered=ordered):
                self.assertEqual(gate.condition_slots(features, masks, ordered).abs().sum().item(), 0)
                self.assertEqual(gate(features, masks, ordered).abs().sum().item(), 0)


class ExistingDynamicsTests(unittest.TestCase):
    def setUp(self):
        from experiments.paper_experiments.agcr_exp.future_prediction import JointDynamics
        statistics=dict(state=dict(mean=[0.]*11,std=[1.]*11),
                        acceleration=dict(mean=[0.]*3,std=[1.]*3),
                        class_velocity={str(k):dict(mean=[0.]*3,covariance=np.eye(3).tolist()) for k in range(1,5)})
        torch.manual_seed(17)
        self.model=JointDynamics(statistics).eval()
        self.tokens=torch.randn(1,1,256,requires_grad=True)
        self.scene=torch.randn(1,2,256,requires_grad=True)
        self.initial=torch.zeros(1,1,11,requires_grad=True)
        self.presence=torch.full((1,1),10.,requires_grad=True)

    def test_actual_dynamics_gradient_and_no_reference_interface(self):
        result=persistent_mode_rollouts(self.model,self.tokens,self.scene,self.initial,self.presence,
                                       torch.Generator().manual_seed(101),draws=2,horizon=2)
        self.assertEqual(result['state'].shape,(1,8,2,2,1,11))
        result['state'][...,:6].square().mean().backward()
        self.assertGreater(self.tokens.grad.abs().sum().item(),0)
        self.assertGreater(self.initial.grad.abs().sum().item(),0)
        self.assertGreater(self.model.acceleration.weight.grad.abs().sum().item(),0)

    def test_actual_discrete_sampler_repeatability_and_score_gradient(self):
        args=(self.model,self.tokens,self.scene,self.initial,self.presence)
        a=sample_shared_future(*args,torch.Generator().manual_seed(211),horizon=2)
        b=sample_shared_future(*args,torch.Generator().manual_seed(211),horizon=2)
        torch.testing.assert_close(a['state'],b['state'])
        # An absent target at the zero initialization still incurs reference
        # error. Using distance-to-zero here would give this draw zero reward
        # and incorrectly expect a nonzero score-function gradient.
        loss=(a['state'][...,:3]-1).square().mean((1,2,3))
        stochastic_feedback_surrogate(loss,a['discrete_log_prob'],0.).backward()
        self.assertGreater(self.model.mixture.weight.grad.abs().sum().item(),0)


def fake_family(root):
    records=[]
    for world in range(2):
        for camera in range(2):
            rgb=np.zeros((22,2,16,16,3),np.uint8)
            rgb[3:6,0,0,0]=world+1
            state=np.zeros((22,1,11)); state[3:,0,0]=world+1
            pose=np.broadcast_to(np.eye(4),(22,2,4,4)).copy(); pose[6:,:,0,3]=camera
            instance=np.ones((22,2,16,16),np.uint16); instance[6:,:,:,camera:camera+1]=0
            instance[3:6,1]=0
            visible=((instance==1).sum((-1,-2))/256).astype(np.float32)[...,None]
            arrays=dict(rgb=rgb,depth_mm=np.ones((22,2,16,16),np.float32)*1000,
                        valid=np.ones((22,2,16,16),bool),K=np.broadcast_to(np.eye(3),(22,2,3,3)).copy(),
                        camera_to_world=pose,time_s=(np.arange(22)-5)*.2,world_state=state,
                        present=np.ones((22,1),bool),instance=instance,visible_fraction=visible,
                        projection_area=np.ones((22,2,1),np.int32)*256,class_id=np.array([3]),
                        instance_id=np.array([1]),world_from_factory=np.eye(4))
            name='w%d_c%d.npz'%(world,camera); np.savez(root/name,**arrays)
            records.append(dict(path=name,world=world,camera=camera,family_id='f',scene_parent_id='p',
                                split='val',cohort='evidence',receiver=1,target_index=0,
                                environment_sha256='synthetic-test-only',nuisance_seed=17,passive_camera=True))
    (root/'manifest.json').write_text(json.dumps(dict(records=records)))
    return records


class DataAndMetricTests(unittest.TestCase):
    def test_future_labels_cannot_change_frontend_inputs(self):
        from .data import load_branch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fake_family(root);path=root/'w0_c0.npz'
            before=load_branch(path)
            with np.load(path) as file: arrays=dict(file)
            arrays['rgb'][6:]=123;arrays['depth_mm'][6:]=4321;arrays['world_state'][6:]+=9
            np.savez(path,**arrays)
            after=load_branch(path)
            for key in before['inputs']:
                torch.testing.assert_close(before['inputs'][key],after['inputs'][key])
            self.assertFalse(torch.equal(before['supervision']['state'],after['supervision']['state']))

    def test_exact_controls_and_background_leak_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fake_family(root)
            self.assertTrue(audit_manifest(root/'manifest.json')['passed'])
            path=root/'w1_c0.npz'
            with np.load(path) as f: arrays=dict(f)
            arrays['rgb'][4,1,15,15,0]=1 # an extra background/shadow pixel
            np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,'history/rgb'):
                audit_manifest(root/'manifest.json')

    def test_near_variant_split_leak_rejected_before_loading_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); records=fake_family(root)
            extra=dict(records[0],family_id='different-family',split='test',path='unused.npz')
            (root/'manifest.json').write_text(json.dumps(dict(records=records+[extra])))
            with self.assertRaisesRegex(ValueError,'lineage leakage'):
                audit_manifest(root/'manifest.json')

    def test_duplicate_histories_cannot_hide_behind_new_ancestry_names(self):
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            root=Path(directory); records=fake_family(root); duplicate=[]
            for item in records:
                path='copy_'+item['path']; shutil.copyfile(root/item['path'],root/path)
                duplicate.append(dict(item,path=path,family_id='f2',scene_parent_id='p2',split='test'))
            (root/'manifest.json').write_text(json.dumps(dict(records=records+duplicate)))
            with self.assertRaisesRegex(ValueError,'Identical complete history'):
                audit_manifest(root/'manifest.json')

    def test_reversed_visibility_order_has_distinct_transition_errors(self):
        late=transition_errors([True,True,False,False],[True,True,True,True],2,'disappearance')
        early=transition_errors([False,False,True,True],[True,True,True,True],2,'reappearance')
        self.assertEqual(late['false_visible_frames'],2)
        self.assertEqual(early['false_visible_frames'],2)

    def test_missing_predictions_not_renormalized_or_ignored(self):
        q=branch_diagnostics([[0,0,0,0,-1,-1,-1,-1]],[[.5,.5]])
        self.assertEqual(q['probability'][0,0],.5)
        self.assertEqual(q['missing_or_other'][0],.5)
        e=evidence_effect([[0]*8],[[1]*8]); self.assertEqual(e['aligned_effect'][0],1)
        ref=[False]*5+[True]*11
        t=reappearance_timing(ref,[False]*16)
        self.assertTrue(t['missing']); self.assertEqual(t['absolute_seconds'],3.2)
        leak=occlusion_violations(np.ones((1,2,2)),np.ones((1,2,2)),np.zeros((1,2,2)))
        self.assertEqual(leak['false_pixel_fraction'],1)

    def test_expected_brier_includes_irreducible_branch_uncertainty(self):
        calibrated=branch_diagnostics([[0]*4+[1]*4],[[.5,.5]])
        self.assertEqual(calibrated['brier'][0],.5)
        self.assertEqual(calibrated['probability_error'][0],0)
        deterministic=branch_diagnostics([[0]*8],[[.5,.5]])
        self.assertEqual(deterministic['brier'][0],1)


if __name__=='__main__':
    unittest.main()
