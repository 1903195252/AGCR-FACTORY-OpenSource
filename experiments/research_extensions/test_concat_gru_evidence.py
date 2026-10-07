"""Check historical conditioning and actual free-prefix presence compatibility."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from .train_concat_gru_evidence import HISTORY_KEYS, evidence_loss
from .state_prediction_baselines import ConcatGRU


def statistics():
    return dict(state=dict(mean=[0.]*11, std=[1.]*11),
        acceleration=dict(mean=[0.]*3, std=[1.]*3),
        class_velocity={str(k): dict(mean=[0.]*3, covariance=torch.eye(3).tolist()) for k in range(1, 5)})


def history(value=0.):
    return dict(empty=False, tokens=torch.zeros(1, 1, 256), shared=torch.zeros(1, 256),
        separate=torch.full((2, 2, 256), value), initial=torch.zeros(1, 1, 11),
        covariance=torch.eye(6)[None, None]*.01, presence_logits=torch.ones(1, 1),
        result=dict(tracks=[dict(birth=(0, 0, 0), score=1., presence_logit=1.)]))


class ConcatEvidenceTests(unittest.TestCase):
    def test_compact_history_preserves_presence_selection(self):
        from experiments.base48.model import free_prefix_presence_objective
        torch.set_num_threads(2)
        h = {k: history()[k] for k in HISTORY_KEYS}
        model = ConcatGRU(statistics()); model.condition(h)
        loss, stats = free_prefix_presence_objective(model, h, torch.tensor([0]),
            torch.ones(16, 1, dtype=torch.bool), torch.tensor([True]), torch.Generator().manual_seed(17))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['future_presence_supervised_targets'], 1)
        loss.backward()
        self.assertGreater(float(model.presence.weight.grad.abs().sum()), 0.)

    def test_each_evidence_world_sets_own_history(self):
        model = ConcatGRU(statistics())
        pair = [dict(metadata=dict(world=w, cohort='evidence'), history=history(float(w)),
            focal_slot=0, focal_reference=0, future_state=torch.full((16, 1, 11), float(w))) for w in (0, 1)]
        seen = []
        def rollout(m, tokens, scene, initial, logits, generator, slot, **kwargs):
            seen.append(float(m._view_context.mean()))
            state = m.acceleration.weight.sum()*0 + initial.new_zeros(1, 8, 2, 16, 1, 11)
            return dict(state=state, mode_logits=initial.new_zeros(1, 8), log_survival=initial.new_zeros(1, 8, 2))
        def density(state, logits, refs, std, bandwidth, survival):
            # Only the loss receives the references; context was set before it.
            self.assertEqual(tuple(refs.shape), (1, 2, 16, 6))
            return state.sum()*0 + std.new_tensor([[1., 2.]])
        with patch('experiments.base48.model.persistent_target_rollouts', rollout), \
                patch('experiments.paired_interventions.objectives.candidate_log_density', density):
            loss = evidence_loss(model, pair, 0, 17017)
        self.assertEqual(seen, [0., 1.])
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__': unittest.main()
