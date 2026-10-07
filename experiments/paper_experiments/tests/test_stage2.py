import inspect
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from experiments.paper_experiments.agcr_exp.data import atomic_json
from experiments.paper_experiments.agcr_exp.future_prediction import JointDynamics, geometry_intrinsics, geometry_loss, project_voxels
from experiments.paper_experiments.agcr_exp.stage2_data import area8, depth8, training_statistics
from experiments.paper_experiments.agcr_exp.stage2_system import StageTwo, matched_state_supervision
from experiments.paper_experiments.agcr_exp.temporal_state import TemporalState, gaussian_loss


def statistics():
    return dict(state=dict(mean=[0.]*11, std=[2., 3., 4., .2, .3, .4, 1., 1., 1., 1., 1.]),
                acceleration=dict(mean=[0.]*3, std=[1.]*3),
                class_velocity={str(k):dict(mean=[1., 0., 0.], covariance=np.eye(3).tolist()) for k in range(1, 5)})


class StageTwoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(17)

    def test_area_preserves_support_and_excludes_padding(self):
        pixels = np.ones((2, 480, 720), np.float32)
        pooled = area8(pixels)
        self.assertEqual(pooled.shape, (2, 60, 92))
        np.testing.assert_equal(pooled[..., 1:-1], 1)
        np.testing.assert_equal(pooled[..., [0, -1]], 0)
        self.assertAlmostEqual(float(pooled.sum()*64), float(pixels.sum()))

    def test_depth_pooling_never_averages_invalid_zero_into_depth(self):
        d = np.zeros((480, 720), np.float32); d[0:4, 0:8] = 2000
        z, mask = depth8(d)
        self.assertEqual(z[0, 1], 2000)
        self.assertEqual(mask[0, 1], .5)
        self.assertEqual(z[0, 2], 0)

    def test_pixel_centers_and_video_padding_share_calibration(self):
        K = torch.tensor([[400., 0., 359.5], [0., 400., 239.5], [0., 0., 1.]])
        pooled = geometry_intrinsics(K)
        native = torch.tensor([19.5, 39.5, 1.])
        pixel = torch.tensor([(native[0]+.5)/8-.5+1, (native[1]+.5)/8-.5, 1.])
        torch.testing.assert_close(torch.linalg.inv(K)@native, torch.linalg.inv(pooled)@pixel)

    def test_initial_measurement_is_used_once_and_missing_evidence_retains_memory(self):
        tracker = TemporalState(statistics(), nn.Sequential(nn.Linear(515, 256), nn.GELU(), nn.Linear(256, 1)))
        def evidence(x):
            return dict(center=torch.tensor([x, 0., 0.]), covariance=torch.ones(3),
                        token=torch.ones(256), class_prob=torch.tensor([0., 1., 0., 0., 0.]), dimensions=torch.ones(3),
                        heading=torch.tensor([0., 1.]), weight=torch.tensor(1.), confidence=1., appearance=torch.ones(4, 256))
        state = tracker.update(dict(appearance=[]), [evidence(0.), evidence(4.)], initial=True)
        torch.testing.assert_close(state['mean'], torch.tensor([2., 0., 0., 1., 0., 0.]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state['covariance'][:3, :3], torch.eye(3)*.5, atol=1e-4, rtol=1e-4)
        memory = state['memory'].clone()
        predicted = tracker.update(tracker.predict(state, .2), [])
        torch.testing.assert_close(predicted['memory'], memory)
        self.assertAlmostEqual(float(predicted['age']), .2, places=5)
        self.assertAlmostEqual(float(predicted['mean'][0]), 2.2, places=4)
        self.assertGreater(float(torch.linalg.eigvalsh(predicted['covariance']).min()), 0)

    def test_covariance_loss_has_finite_uncertainty_gradients(self):
        parameter = torch.tensor([.1, .2, .3], requires_grad=True)
        C = torch.diag(parameter.exp())
        loss = gaussian_loss(torch.ones(3), C)
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(parameter.grad.norm()), 0)

    def test_dynamics_cannot_read_later_teacher_forced_states(self):
        model = JointDynamics(statistics()).eval()
        # Full six-layer causal computation, with three future steps in this unit test.
        previous = torch.randn(1, 3, 2, 11)
        tokens, scene = torch.randn(1, 2, 256), torch.randn(1, 4, 256)
        a = model(tokens, scene, previous)
        changed = previous.clone(); changed[:, 2] += 100
        b = model(tokens, scene, changed)
        torch.testing.assert_close(a['acc_mean'][:, :, :2], b['acc_mean'][:, :, :2])
        torch.testing.assert_close(a['mode_logits'], b['mode_logits'])

    def test_each_rollout_selects_one_scene_mode_without_8x8_expansion(self):
        model = JointDynamics(statistics()).eval()
        previous = torch.zeros(8, 1, 2, 11)
        out = model(torch.randn(8, 2, 256), torch.randn(8, 4, 256), previous, torch.arange(8))
        self.assertEqual(tuple(out['acc_mean'].shape), (8, 1, 1, 2, 3))

    def test_sampled_absence_is_absorbing_and_stops_continuous_motion(self):
        model = JointDynamics(statistics()).eval()
        initial = torch.randn(2, 11)
        out = model.sample(torch.randn(2, 256), torch.randn(1, 4, 256), initial, torch.zeros(2, dtype=torch.bool), torch.Generator().manual_seed(17))
        self.assertFalse(out['present'].any())
        torch.testing.assert_close(out['state'], initial[None].expand(16, -1, -1))
        self.assertIn(out['mode'], range(8))

    def test_background_only_objective_is_finite(self):
        prediction = dict(Z=torch.ones(1, 1, 60, 92)*1000, P=torch.ones(1, 1, 60, 92)*.9,
                          M=torch.ones(1, 1, 60, 92)*.9)
        zero = torch.zeros(1, 0, 60, 92)
        target = dict(static_depth_mm=torch.ones(1, 60, 92)*1000, static_support=torch.ones(1, 60, 92),
                      isolated_depth_mm=zero, full_mask=zero, visible_mask=zero)
        loss, terms = geometry_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(terms['silhouette']), 0)

    def test_voxel_projection_keeps_nearest_depth_and_padding_empty(self):
        voxels = torch.tensor([[0., 0., 1.], [0., 0., 2.]])
        K = torch.tensor([[50., 0., 45.5], [0., 50., 29.5], [0., 0., 1.]])
        result = project_voxels(voxels, K, torch.eye(4)[None])[0]
        self.assertAlmostEqual(float(result[0, 29, 45]), 950, places=3)
        self.assertEqual(float(result[1, :, [0, -1]].sum()), 0)

    def test_training_normalization_does_not_read_validation_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'episodes/train'; path.mkdir(parents=True)
            with h5py.File(path/'frames.h5', 'w') as h:
                states = np.zeros((22, 4, 11)); states[..., 8:] = 1
                states[..., 0] = np.arange(22)[:, None]
                h['state'] = states
                h['ego_to_world'] = np.tile(np.eye(4), (22, 2, 1, 1))
                h['class_id'] = np.array([3, 4, 13, 14])
            row = dict(role='train', history_covered=True, episode='train', start=0, anchor=5,
                       history_eligible=[True]*4, uid='train:000')
            # No validation H5 exists: touching it is a test failure.
            val = dict(row, role='val', episode='must_not_be_read', uid='val:000')
            index = root/'index.json'
            atomic_json(index, dict(dataset_root=str(root), split_sha256='test', windows=[row, val]))
            report = training_statistics(index, root/'stats.json')
            self.assertEqual(report['episodes'], 1)
            self.assertEqual(report['windows'], 1)
            self.assertEqual(report['state']['mean'][0], 10.5)

    def test_history_interface_has_no_reference_argument(self):
        self.assertEqual(list(inspect.signature(StageTwo.history).parameters), ['self', 'inputs', 'encoded'])

    def test_unmatched_tracks_never_teacher_force_placeholder_reference_state(self):
        initial = torch.randn(2, 11)
        ids = torch.tensor([-1, 1])
        labels = dict(history_eligible=torch.ones(2, dtype=torch.bool), state=torch.randn(16, 2, 11),
                      history_state=torch.randn(6, 2, 11), present=torch.ones(16, 2, dtype=torch.bool))
        first = matched_state_supervision(initial, ids, labels)
        changed = {k:v.clone() for k, v in labels.items()}
        changed['state'][:, 0] += 1000
        changed['history_state'][:, 0] -= 1000
        second = matched_state_supervision(initial, ids, changed)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(first[4][:, 0], initial[0:1].expand(16, -1))
        self.assertFalse(first[5][:, 0].any())


if __name__ == '__main__':
    unittest.main()
