"""CPU boundary checks for the independent Base objectives; no model downloads."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments.base48.model import (Base48System, DEFAULT_VARIANT, focal_history_slot,
                                     persistent_target_rollouts, receiver_only_branch,
                                     free_prefix_presence_objective, evidence_scores)
from experiments.paired_interventions.objectives import candidate_log_density, evidence_preference
from experiments.paired_interventions.rollout import persistent_mode_rollouts


class ToyDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.acceleration = nn.Parameter(torch.tensor(.01))
        self.presence = nn.Parameter(torch.tensor([-.1, -2., .3]))
        self.scale = SimpleNamespace(acc_mean=torch.zeros(3), acc_std=torch.ones(3))

    def forward(self, tokens, scene, previous, mode_ids=None):
        batch, time, count = previous.shape[:3]
        modes = 8 if mode_ids is None else 1
        vector = (batch, modes, time, count, 3)
        scalar = vector[:-1]
        return dict(mode_logits=self.acceleration.expand(batch, 8),
                    acc_mean=self.acceleration.expand(vector), acc_scale=torch.ones(vector)*.1,
                    yaw_mean=torch.zeros(scalar), yaw_scale=torch.ones(scalar)*.1,
                    presence_logits=self.presence[None, None, None].expand(scalar))


class Base48ModelTest(unittest.TestCase):
    def test_evidence_mean_scores_reduce_gradient_scale_without_detaching_predictions(self):
        path = torch.zeros(1, 1, 1, 16, 6, requires_grad=True)
        references = torch.zeros(1, 2, 16, 6, requires_grad=True)
        with torch.no_grad():
            references[:, 0, :, 3] = -.6
            references[:, 1, :, 3] = .6
        scales = torch.tensor([8., 5., .8, .613, .9, .05])
        density = candidate_log_density(path, torch.zeros(1, 1), references, scales, .2,
                                       torch.zeros(1, 1, 1))
        raw = evidence_preference(evidence_scores(density, 96, 'sum'), torch.tensor([1]))
        normalized = evidence_preference(evidence_scores(density, 96, 'mean'), torch.tensor([1]))
        raw_gradient = torch.autograd.grad(raw, path, retain_graph=True)[0]
        gradient = torch.autograd.grad(normalized, path, retain_graph=True)[0]
        torch.testing.assert_close(gradient*96, raw_gradient)
        self.assertGreater(float(gradient.norm()), 0.)
        self.assertIsNone(torch.autograd.grad(normalized, references, allow_unused=True)[0])
        # Adding a shared survival penalty must not be advertised as absolute
        # persistence supervision: candidate preference cancels that factor.
        shifted = evidence_preference(evidence_scores(density-50, 96, 'mean'), torch.tensor([1]))
        torch.testing.assert_close(normalized, shifted)

    def test_invalid_evidence_calibration_fails_before_asset_loading(self):
        for scales in ([1.]*5, [1., 1., 1., 0., 1., 1.], [1., 1., 1., float('nan'), 1., 1.]):
            with self.assertRaisesRegex(ValueError, 'Evidence loss scales'):
                Base48System({}, torch.device('cpu'), 'state', {'evidence_loss_scales': scales})
        with self.assertRaisesRegex(ValueError, 'Evidence score reduction'):
            Base48System({}, torch.device('cpu'), 'state', {'evidence_score_reduction': 'max'})

    def test_free_prefix_presence_uses_targets_only_after_prediction_and_keeps_context_gradients(self):
        class ContextDynamics(ToyDynamics):
            def __init__(self):
                super().__init__()
                self.scored_prefixes = []
            def forward(self, tokens, scene, previous, mode_ids=None):
                output = super().forward(tokens, scene, previous, mode_ids)
                output['presence_logits'] = (output['presence_logits'] + .2*tokens.sum(-1)[:, None, None]
                    + .1*scene.sum((1, 2))[:, None, None, None] + .05*previous[..., 0][:, None])
                if torch.is_grad_enabled():
                    self.scored_prefixes.append(previous.detach().clone())
                return output
        model = ContextDynamics()
        history = dict(empty=False, tokens=torch.zeros(1, 3, 2, requires_grad=True),
            shared=torch.zeros(1, 2, 2, requires_grad=True), initial=torch.zeros(1, 3, 11, requires_grad=True),
            covariance=(torch.eye(6).repeat(1, 3, 1, 1)*.01).requires_grad_(),
            result=dict(tracks=[dict(birth=(0, 0, k), score=float(k), presence_logit=torch.tensor(0.)) for k in range(3)]))
        ids = torch.tensor([0, 0, 1]); eligible = torch.tensor([True, True])
        positive, stats = free_prefix_presence_objective(model, history, ids,
            torch.ones(16, 2, dtype=torch.bool), eligible, torch.Generator().manual_seed(17))
        self.assertEqual(stats['future_presence_supervised_targets'], 2)
        self.assertEqual(stats['future_presence_supervised_steps'], 32)
        positive.backward()
        self.assertLess(float(model.presence.grad[1]), 0.)
        self.assertLess(float(model.presence.grad[2]), 0.)
        self.assertGreater(float(history['tokens'].grad.abs().sum()), 0.)
        self.assertGreater(float(history['shared'].grad.abs().sum()), 0.)
        self.assertIsNone(history['initial'].grad)
        self.assertIsNone(history['covariance'].grad)
        self.assertIsNone(model.acceleration.grad)
        negative, stats = free_prefix_presence_objective(model, history, ids,
            torch.zeros(16, 2, dtype=torch.bool), eligible, torch.Generator().manual_seed(17))
        self.assertEqual(stats['future_presence_supervised_steps'], 2)
        self.assertTrue(torch.equal(model.scored_prefixes[0], model.scored_prefixes[1]))
        self.assertNotEqual(float(positive), float(negative))

    def test_receiver_only_removes_images_detections_scene_and_history_supervision(self):
        branch = dict(inputs=dict(rgb=torch.ones(6, 2, 2, 2, 3, dtype=torch.uint8),
                                  depth_mm=torch.ones(6, 2, 2, 2)*900,
                                  depth_valid=torch.ones(6, 2, 2, 2, dtype=torch.bool)),
                      supervision=dict(history_instance=torch.ones(6, 2, 2, 2, dtype=torch.long),
                          history_visible_fraction=torch.ones(6, 2, 2),
                          history_projection_area=torch.ones(6, 2, 2)*200,
                          history_isolated_support=torch.ones(6, 2, 2, 2, 2), state=torch.randn(16, 2, 11)))
        branch['_prefetched_replay'] = dict(offset=0, auxiliary={}, labels=dict(
            history_detectable=torch.ones(6, 2, 2, dtype=torch.bool),
            history_full_mask=torch.ones(6, 2, 2, 1, 1),
            history_visible_mask=torch.ones(6, 2, 2, 1, 1), history_eligible=torch.ones(2, dtype=torch.bool)))
        for receiver in (0, 1):
            actual = receiver_only_branch(branch, receiver)
            self.assertTrue(actual['inputs']['observation_available'][:, receiver].all())
            self.assertFalse(actual['inputs']['observation_available'][:, 1-receiver].any())
            for key in ('rgb', 'depth_mm', 'depth_valid'):
                self.assertEqual(int(actual['inputs'][key][:, 1-receiver].count_nonzero()), 0)
                self.assertTrue(torch.equal(actual['inputs'][key][:, receiver], branch['inputs'][key][:, receiver]))
            self.assertFalse(actual['_prefetched_replay']['labels']['history_detectable'][:, 1-receiver].any())
            self.assertTrue(torch.equal(actual['supervision']['state'], branch['supervision']['state']))
        self.assertTrue(branch['inputs']['depth_valid'].all())
        # The existing runtime explicitly suppresses unavailable detections,
        # removes their scene memory, and masks depth in static projection.
        from experiments.paired_interventions.upstream_runtime import RuntimeStageTwo
        import inspect
        source = inspect.getsource(RuntimeStageTwo.history)
        self.assertIn('torch.where(flat.reshape', source)
        self.assertIn('background[available]', source)
        self.assertIn("inputs['observation_available']", inspect.getsource(RuntimeStageTwo.project_history))

    def test_target_survival_and_paths(self):
        model = ToyDynamics()
        initial = torch.zeros(1, 3, 11)
        logits = torch.tensor([[0., -1., .2]])
        args = (model, torch.zeros(1, 3, 2), torch.zeros(1, 2), initial, logits)
        actual = persistent_target_rollouts(*args, torch.Generator().manual_seed(17), target_slot=1, draws=2, horizon=3)
        original = persistent_mode_rollouts(*args, torch.Generator().manual_seed(17), draws=2, horizon=3)
        self.assertTrue(torch.equal(actual['state'], original['state']))
        self.assertTrue(torch.equal(actual['mode_logits'], original['mode_logits']))
        expected = F.logsigmoid(logits[0, 1])+3*F.logsigmoid(model.presence[1])
        torch.testing.assert_close(actual['log_survival'], expected.expand(1, 8, 2))
        self.assertFalse(torch.equal(actual['log_survival'], original['log_survival']))
        (actual['state'][..., 1, :6].sum()+actual['log_survival'].sum()).backward()
        self.assertGreater(float(model.acceleration.grad.abs()), 0)
        self.assertNotEqual(float(model.presence.grad[1]), 0.)
        self.assertEqual(float(model.presence.grad[0]), 0.)

    def test_history_representative_is_confidence_based_and_permutation_invariant(self):
        tracks = [dict(score=score, presence_logit=torch.tensor(p), birth=(k, 0, k))
                  for k, (score, p) in enumerate([(1., 0.), (2., 0.), (2., 1.), (9., 0.)])]
        ids = torch.tensor([0, 0, 0, 1])
        history = dict(result=dict(tracks=tracks))
        slot, duplicate = focal_history_slot(history, ids, 0)
        self.assertEqual((slot, duplicate), (2, 2))
        permutation = [2, 3, 0, 1]
        changed = dict(result=dict(tracks=[tracks[k] for k in permutation]))
        selected, extra = focal_history_slot(changed, ids[permutation], 0)
        self.assertEqual(changed['result']['tracks'][selected]['birth'], tracks[slot]['birth'])
        self.assertEqual(extra, duplicate)
        self.assertEqual(focal_history_slot(history, ids, 8), (-1, 0))

    def test_condition_does_not_read_future_supervision(self):
        # Exercise the actual Base forward boundary with tiny stand-ins for
        # expensive encoders/decoders. Future references change the targets,
        # while only inputs reach predict_world/world_conditions.
        model = Base48System.__new__(Base48System)
        nn.Module.__init__(model)
        model.base_phase = 'video'
        model.variant = dict(DEFAULT_VARIANT)
        model.profile = False
        model.upstream = nn.Linear(1, 1)
        model.encoders = SimpleNamespace(history=lambda rgb: (rgb.float().sum(),))
        model.vae = SimpleNamespace(encode=lambda rgb: torch.zeros(2, 48, 5, 30, 46))
        conditions = []
        def forecast(upstream, inputs, encoded, generator):
            state = encoded[0]+torch.randn((), generator=generator)
            return dict(result=dict(tracks=[])), dict(state=state)
        def condition(upstream, history, draw, inputs):
            return dict(state=draw['state'], camera=inputs['planned_camera_to_world'].clone())
        def depth(branch, c, latent):
            conditions.append({k: v.clone() for k, v in c.items()})
            return branch['supervision']['depth_mm'].float().mean()
        model.clean_depth_objective = depth
        model.flow_objective = lambda branch, c, latent, generator: branch['supervision']['rgb'].float().mean()
        branch = dict(inputs=dict(rgb=torch.zeros(6, 2, 1, 1, 3, dtype=torch.uint8),
                                  planned_camera_to_world=torch.eye(4).repeat(16, 2, 1, 1)),
                      supervision=dict(rgb=torch.zeros(16, 2, 1, 1, 3, dtype=torch.uint8),
                                       state=torch.zeros(16, 2, 11), depth_mm=torch.ones(16, 2, 1, 1)))
        changed = copy.deepcopy(branch)
        for value in changed['supervision'].values():
            value.add_(11)
        with patch('experiments.base48.model.predict_world', side_effect=forecast), patch('experiments.base48.model.world_conditions', side_effect=condition):
            g1 = torch.Generator().manual_seed(17029)
            g2 = torch.Generator().manual_seed(17029)
            loss1, _ = model(branch, g1)
            loss2, _ = model(changed, g2)
            for key in conditions[0]:
                self.assertTrue(torch.equal(conditions[0][key], conditions[1][key]))
            self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))
            self.assertNotEqual(float(loss1), float(loss2))
            changed['inputs']['planned_camera_to_world'][..., 0, 3] += 1
            model(changed, torch.Generator().manual_seed(17029))
            self.assertTrue(torch.equal(conditions[0]['state'], conditions[2]['state']))
            self.assertFalse(torch.equal(conditions[0]['camera'], conditions[2]['camera']))
        with self.assertRaises(RuntimeError):
            model.reference_condition(branch, None, None)

    def test_video_train_mode_keeps_upstream_frozen(self):
        model = Base48System.__new__(Base48System)
        nn.Module.__init__(model)
        model.base_phase = 'video'
        model.upstream = nn.Linear(1, 1)
        model.reader = nn.Linear(1, 1)
        model.condition = nn.Linear(1, 1)
        model.depth = nn.Linear(1, 1)
        model.wan = nn.Linear(1, 1)
        model.set_phase('video')
        model.train()
        self.assertFalse(model.upstream.training)
        self.assertFalse(model.reader.training)
        self.assertFalse(any(p.requires_grad for p in model.upstream.parameters()))
        params = {id(p) for group in model.trainable_groups() for p in group['params']}
        self.assertNotIn(id(model.upstream.weight), params)
        with self.assertRaises(ValueError):
            model.set_phase('video', True)

    def test_misspelled_loss_weight_fails_before_asset_loading(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Base'):
            Base48System({}, torch.device('cpu'), 'state', {'free_presence_weight': 1.})


if __name__ == '__main__':
    unittest.main()
