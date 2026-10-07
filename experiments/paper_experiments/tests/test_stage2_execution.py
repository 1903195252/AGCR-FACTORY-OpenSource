import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.paper_experiments.agcr_exp.future_prediction import JointDynamics, project_voxels, unique_integer_rows
from experiments.paper_experiments.agcr_exp.stage2_migration import migration_lineage
from experiments.paper_experiments.agcr_exp.data import atomic_json
from experiments.paper_experiments.agcr_exp.train_observation import sha256


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_packing_is_exact_for_duplicates_negative_coordinates_and_empty_rows(self):
        g = torch.Generator().manual_seed(314)
        rows = torch.randint(-20000, 20000, (4000, 3), generator=g)
        for data in [rows, torch.cat([rows, rows[:300]]), rows[:0], rows[:1]]:
            self.assertTrue(torch.equal(unique_integer_rows(data), torch.unique(data, dim=0)))

    def test_packing_falls_back_before_int64_span_or_product_overflow(self):
        limits = torch.iinfo(torch.int64)
        for rows in [torch.tensor([[limits.min, 0, 0], [limits.max, 0, 0]]),
                     torch.tensor([[-2**30]*3, [2**30]*3, [0]*3])]:
            self.assertTrue(torch.equal(unique_integer_rows(rows), torch.unique(rows, dim=0)))

    def test_projection_chunk_size_preserves_front_depth_support_and_empty_inputs(self):
        g = torch.Generator().manual_seed(29)
        points = torch.randn(47, 3, generator=g)
        points[:, 2] = points[:, 2].abs()+.06
        K = torch.tensor([[50., 0., 45.5], [0., 50., 29.5], [0., 0., 1.]])
        cameras = torch.eye(4)[None].repeat(2, 1, 1)
        cameras[1, 0, 3] = .35
        for value in [points, points[:0]]:
            a = project_voxels(value, K, cameras, chunk_size=7)
            b = project_voxels(value, K, cameras, chunk_size=65536)
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(float(b[:, 1, :, [0, -1]].sum()), 0.)

    def test_large_track_sets_keep_activation_checkpointing(self):
        stats = dict(state=dict(mean=[0.]*11, std=[1.]*11), acceleration=dict(mean=[0.]*3, std=[1.]*3))
        stats['class_velocity'] = {str(k):dict(mean=[0.]*3, covariance=torch.eye(3).tolist()) for k in range(1, 5)}
        model = JointDynamics(stats).train()
        model.checkpoint_tracks_above = 8
        with patch('experiments.paper_experiments.agcr_exp.future_prediction.checkpoint', side_effect=lambda block,*args,**kw: block(*args)) as wrapper:
            for count in (2, 9):
                wrapper.reset_mock()
                model(torch.zeros(1, count, 256), torch.zeros(1, 4, 256), torch.zeros(1, 1, count, 11))
                self.assertEqual(wrapper.call_count, 0 if count == 2 else 6)

    def test_migration_requires_matching_weights_source_protocol_and_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, report = root/'weights.pt', root/'proof.json'
            checkpoint.write_bytes(b'original checkpoint identity')
            old = dict(config={'seed':17, 'history':6, 'future':16}, index_sha256='frozen-index', precision='float32',
                       source_sha256={'trainer.py':'old'}, execution={'workers':4, 'prefetch':8, 'geometry_chunk':8, 'clip_microbatch':1})
            new = copy.deepcopy(old)
            new['source_sha256'] = {'trainer.py':'new'}
            new['execution'].update(policy='optimized_v1', voxel_chunk=65536, dynamics_checkpoint_above_tracks=8)
            proof = dict(passed=True, scope='stage2_execution_migration', origin_step=90,
                         origin_checkpoint_sha256=sha256(checkpoint), origin_manifest=old, new_source_sha256=new['source_sha256'],
                         verified_features=['paired_batch16_equivalence', 'fixed_validation_equivalence', 'optimized_checkpoint_resume'])
            atomic_json(report, proof)
            self.assertEqual(migration_lineage(old, new, checkpoint, report)['origin_step'], 90)
            for key, value in [('index_sha256', 'another-index'), ('precision', 'bfloat16'),
                               ('source_sha256', {'trainer.py':'untested'})]:
                bad = copy.deepcopy(new); bad[key] = value
                with self.assertRaises(ValueError):
                    migration_lineage(old, bad, checkpoint, report)
            bad = copy.deepcopy(new); bad['execution']['workers'] = 12
            with self.assertRaises(ValueError):
                migration_lineage(old, bad, checkpoint, report)
            checkpoint.write_bytes(b'unchecked later checkpoint')
            with self.assertRaises(ValueError):
                migration_lineage(old, new, checkpoint, report)


if __name__ == '__main__':
    unittest.main()
