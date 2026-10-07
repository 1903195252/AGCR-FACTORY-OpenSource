import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.base48.train import (
    BudgetLedger, SAMPLE_SEEDS, SCHEMA, assert_base_paths, atomic_checkpoint,
    draw_plan, protocol_settings, registered_variant, validate_checkpoint_identity,
    source_hashes, audit_index_ancestry,
)
from experiments.paired_interventions.data import FamilySampler, file_sha256


class TrainingProtocolTests(unittest.TestCase):
    def protocol(self):
        return dict(schema='agcr-base48-protocol-1', stages=dict(state=dict(
            steps=100, batch=16, validate_every=20, validation_panel=4)),
            variants=dict(full=dict(paired_data=True)))

    def test_unregistered_or_changed_schedule_rejected(self):
        cfg = protocol_settings(self.protocol(), 'state')
        self.assertEqual(cfg['steps'], 100)
        for phase, override in [('video', {}), ('state', {'steps': 99})]:
            with self.assertRaises(ValueError): protocol_settings(self.protocol(), phase, override)
        p = self.protocol(); p['paired_sample_seeds'] = SAMPLE_SEEDS[:1]
        with self.assertRaises(ValueError): protocol_settings(p, 'state')
        self.assertTrue(registered_variant(self.protocol(), 'full')['paired_data'])

    def test_pro_paths_are_read_only(self):
        paths = {'upstream': '/data/upstream/last.pt'}
        for output, budget in [('artifacts/agcr-intervention-runs/x', '/tmp/base/budget.json'),
                               ('/tmp/base', 'artifacts/agcr-intervention-runs/budget.json'),
                               ('/data/upstream/base', '/tmp/budget.json')]:
            with self.assertRaises(ValueError): assert_base_paths(output, budget, paths)
        assert_base_paths('/tmp/base', '/tmp/base-budget.json', paths)

    def test_whole_family_balanced_between_ranks_and_exact_sampler_resume(self):
        fs, ds = FamilySampler(100, 3), FamilySampler(100, 9)
        plan, nf, nd = draw_plan(0, 16, True, fs, ds, world=2)
        self.assertEqual((nf, nd), (4, 12))
        for rank in range(2):
            self.assertEqual(sum(x[0] == 'family' for x in plan[rank::2]), 2)
        family_ids = [x[1] for x in plan if x[0] == 'family']
        self.assertEqual(family_ids, [fs.index(i) for i in range(4)])
        fs.advance(nf); ds.advance(nd)
        resumed = draw_plan(1, 16, True, FamilySampler(**fs.state_dict()),
                            FamilySampler(**ds.state_dict()), world=2)
        self.assertEqual(draw_plan(1, 16, True, fs, ds, world=2), resumed)
        self.assertEqual(draw_plan(0, 16, False, fs, ds, world=2)[1:], (0, 16))

    def test_budget_exact_billing_lock_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'budget.json'; clock = [0.]
            ledger = BudgetLedger(path, 24., 2, clock=lambda: clock[0])
            with self.assertRaises(RuntimeError): BudgetLedger(path, 24., 2)
            clock[0] = 1800.; ledger.commit('running')
            self.assertEqual(json.loads(path.read_text())['used_gpu_hours'], 1.)
            ledger.close(); ledger.close()  # no double accounting
            clock[0] = 10000.
            resumed = BudgetLedger(path, 24., 2, clock=lambda: clock[0])
            clock[0] += 900.; resumed.close()
            self.assertEqual(json.loads(path.read_text())['used_gpu_hours'], 1.5)
            with self.assertRaises(ValueError): BudgetLedger(path, 25., 2)

    def test_pro_warm_start_does_not_require_false_completion(self):
        identity = dict(schema=SCHEMA, phase='state', family_index_sha256='original', ordinary_index_sha256='ordinary')
        old = dict(identity=dict(development_only=False, phase='dynamics', family_index_sha256='original', ordinary_index_sha256='ordinary'), step=936, completed=False)
        validate_checkpoint_identity(old, identity, 'warm_start')
        self.assertFalse(old['completed'])
        with self.assertRaises(ValueError): validate_checkpoint_identity(old, dict(identity, phase='video'), 'warm_start')
        with self.assertRaises(ValueError): validate_checkpoint_identity(old, identity, 'init')

    def test_exact_resume_and_completed_predecessor(self):
        base = dict(schema=SCHEMA, phase='state', variant='full', seed=17,
            protocol_sha256='p', family_index_sha256='f', ordinary_index_sha256='o',
            parent_sha256='u', source_sha256={'train':'x'})
        prev = dict(identity=base, completed=True)
        validate_checkpoint_identity(prev, dict(base), 'resume')
        with self.assertRaises(ValueError): validate_checkpoint_identity(prev, dict(base, seed=29), 'resume')
        validate_checkpoint_identity(prev, dict(base, phase='depth'), 'init')
        with self.assertRaises(ValueError): validate_checkpoint_identity(dict(prev, completed=False), dict(base, phase='depth'), 'init')
        with self.assertRaises(ValueError): validate_checkpoint_identity(prev, dict(base, phase='depth', protocol_sha256='changed'), 'init')

    def test_only_real_base_and_performance_dependencies_affect_identity(self):
        actual = source_hashes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in actual:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
            before = source_hashes(root)
            for name in ('base48/sample.py', 'base48/runner.py', 'base48/downstream.py',
                         'performance/new_benchmark_tool.py'):
                (root/'experiments'/name).write_text('irrelevant tool')
            self.assertEqual(before, source_hashes(root))
            for name in ('base48/model.py', 'base48/train.py', 'performance/prefetched_replay.py'):
                path = root/'experiments'/name; old = path.read_text(); path.write_text(old+' changed')
                self.assertNotEqual(before, source_hashes(root))
                path.write_text(old)

    def test_index_ancestry_preserves_training_and_whole_test_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def family(name, split):
                return dict(family_id=name, split=split, records=[dict(world=w,camera=c,path=name+str(w)+str(c),sha256='same') for w in range(2) for c in range(2)])
            source = dict(schema='agcr-intervention-index-1', audit_passed=True, development_only=False,
                root=str(root/'data'), families=[family('train','train'), family('val','val'),
                                                family('test1','test'), family('test2','test')])
            path = root/'source.json'; path.write_text(json.dumps(source))
            subset = dict(source, families=source['families'][:3], source_index_path=str(path),
                          source_index_sha256=file_sha256(path))
            base = root/'base.json'; base.write_text(json.dumps(subset))
            audit = audit_index_ancestry(base)
            self.assertTrue(audit['train_val_identical']); self.assertEqual(audit['test_families'],1)
            new_identity = dict(schema=SCHEMA, phase='state', family_index_sha256=file_sha256(base),
                                family_index_ancestry=audit, ordinary_index_sha256='o')
            previous = dict(identity=dict(development_only=False,family_index_sha256=file_sha256(path),ordinary_index_sha256='o'),completed=False)
            validate_checkpoint_identity(previous,new_identity,'warm_start')
            with self.assertRaises(ValueError):
                validate_checkpoint_identity(previous,dict(new_identity,family_index_ancestry=None),'warm_start')
            with self.assertRaises(ValueError):validate_checkpoint_identity(previous,new_identity,'resume')
            for changed in [dict(subset,families=subset['families'][1:]),
                            dict(subset,root=str(root/'elsewhere'))]:
                base.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):audit_index_ancestry(base)
            changed=json.loads(json.dumps(subset));changed['families'][2]['records'].pop()
            base.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):audit_index_ancestry(base)

    def test_atomic_checkpoint_retains_unfinished_status_and_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'last.pt'
            source = dict(step=7, completed=False, model={'w':torch.arange(3)}, rank_rng=[{'torch':torch.get_rng_state()}])
            atomic_checkpoint(path, source)
            result = torch.load(path, weights_only=False)
            self.assertFalse(result['completed']); self.assertEqual(result['step'], 7)
            self.assertTrue(torch.equal(result['model']['w'], source['model']['w']))
            self.assertFalse(path.with_name('last.pt.tmp').exists())


if __name__ == '__main__': unittest.main()
