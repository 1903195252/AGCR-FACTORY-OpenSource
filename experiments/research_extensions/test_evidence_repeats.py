"""Critical-path checks for independent-seed, isolated-coefficient replication."""
import ast
import copy
import inspect
import unittest
from unittest.mock import patch
import torch
from . import train_future_no_evidence as reference
from . import train_evidence_repeats as repeat
from . import typed_evidence_repeat_test as scoring
from . import typed_no_evidence_test as original_scoring

class ReplicationTests(unittest.TestCase):
    def test_numerical_helpers_unchanged(self):
        for name in ('free_mode_means','project_centers','free_motion_projection_loss',
                     'evidence_pair_loss','geometry_surrogate','prepare_row','future_tensors'):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(reference,name)))),
                             ast.dump(ast.parse(inspect.getsource(getattr(repeat,name)))),name)

    def test_update_loop_preserves_rng_and_other_losses(self):
        def loop(module):
            s=inspect.getsource(module.run)
            return s[s.index('        while step < args.steps'):s.index("        if status.get('checkpoint_step')")]
        value=loop(repeat).replace('args.evidence_coefficient','0.')
        value=value.replace('# Identical forward and RNG stream in both coefficient arms.',
                            '# Evaluate the identical evidence branch/RNG stream, but contribute zero gradient.')
        self.assertEqual(value,loop(reference))

    def payload(self,seed,coefficient):
        return dict(schema=repeat.SCHEMA,parent=dict(sha256='p'),step=2048,target_steps=2048,completed=True,
            history_checkpoint=dict(path='history',sha256='h'),future_overlay={'dynamics.w':torch.ones(1)},
            identity=dict(configuration=dict(arm='paired_geometry',component_ablation='isolated_evidence_replication',
                evidence_forward_retained=True,stochastic_order_unchanged=True,seed=seed,steps=2048,lr=1e-5,
                weights=dict(evidence=coefficient,motion_projection=.5,common_geometry=.1,reappearance_geometry=.1),
                history_frozen=True,future_truth_conditions=False,deployment_sampler_changed=False,
                ambiguous_nll_and_presence_only=True),data=dict(split='train',test_used=False,
                selection_uses_validation_scores=False,index_sha256='i',branches=[dict(split='train',camera=0)])))

    def test_only_registered_seeds_and_complete_full_runs_admitted(self):
        with patch.object(repeat,'sha256',side_effect=lambda p:'h'if p=='history'else'p'):
            for seed in (17017,17029,17043):
                for weight in (0.,.1):repeat.validate_checkpoint(self.payload(seed,weight),'parent','i',True)
            for seed,weight in ((42,.1),(17017,.05)):
                with self.assertRaises(ValueError):repeat.validate_checkpoint(self.payload(seed,weight),'parent','i',True)
            p=self.payload(17017,.1);p.update(step=2047,completed=False)
            with self.assertRaises(ValueError):repeat.validate_checkpoint(p,'parent','i',True)

    def test_full_population_and_sampler_contract_retained(self):
        self.assertEqual(scoring.SEEDS,[101,211,307,401,503,601,701,809])
        for name in ('validate_population','validate_input_boundary','validate_family_rows','rows_sha'):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(scoring,name)))),
                             ast.dump(ast.parse(inspect.getsource(getattr(original_scoring,name)))),name)
        self.assertIn('sampling_stream_training_seed',inspect.getsource(scoring.validate_lock))

if __name__=='__main__':unittest.main()
