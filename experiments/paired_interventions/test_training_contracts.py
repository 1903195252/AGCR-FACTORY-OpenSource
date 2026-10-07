"""Mechanism and data-boundary regression tests; no benchmark assertions."""
import unittest
import numpy as np
import torch

from .data import CLASS_MAP,FamilySampler
from .video_losses import depth_loss,reference_for_queries
from .video_modules import CheckpointCausalResidual


class TrainingContracts(unittest.TestCase):
    def test_native_class_mapping_agrees_with_original_frontend(self):
        from experiments.paper_experiments.agcr_exp.data import CLASS_MAP as original
        self.assertEqual(CLASS_MAP,original)
        self.assertEqual(CLASS_MAP[4],2)

    def test_whole_family_rank_partition_and_exact_resume(self):
        sampler=FamilySampler(31,17)
        expected=[sampler.index(i) for i in range(16)]
        ranks=[sampler.rank_indices(16,r,2) for r in range(2)]
        interleaved=[ranks[i%2][i//2] for i in range(16)]
        self.assertEqual(interleaved,expected);self.assertEqual(len(set(expected)),16)
        sampler.advance(16);restored=FamilySampler(**sampler.state_dict())
        self.assertEqual([sampler.index(i) for i in range(64)],[restored.index(i) for i in range(64)])

    def test_intermittent_feedback_covers_the_entire_informative_cohort(self):
        # Real schedule: global batch 16, a feedback update every eight steps.
        # The former ordinary-cursor modulo schedule covered only 25%/12.5%
        # of these cohorts. Every family must occur once before epoch repeat.
        for count in (320,640):
            for ranks in (2,4):
                with self.subTest(families=count,world_size=ranks):
                    sampler=FamilySampler(count,526)
                    seen=[]
                    for update in range(1,8*(count//16)+1):
                        if update%8: continue
                        assigned=[sampler.rank_indices(16,rank,ranks) for rank in range(ranks)]
                        batch=[assigned[i%ranks][i//ranks] for i in range(16)]
                        self.assertEqual(len(set(batch)),16)
                        seen.extend(batch);sampler.advance(16)
                    self.assertEqual(len(seen),count)
                    self.assertEqual(set(seen),set(range(count)))
                    self.assertEqual(sampler.cursor,count)

    def test_feedback_stream_resume_is_independent_of_base_update_count(self):
        original=FamilySampler(320,526)
        original.advance(16*7)
        restored=FamilySampler(**original.state_dict())
        # Several intervening base updates do not consume this stream. A
        # replayed uncommitted feedback draw also must not consume it.
        first=[original.index(i) for i in range(16)]
        self.assertEqual(first,[restored.index(i) for i in range(16)])
        self.assertEqual(first,[restored.index(i) for i in range(16)])
        for _ in range(25):
            self.assertEqual([original.index(i) for i in range(16)],
                             [restored.index(i) for i in range(16)])
            original.advance(16);restored.advance(16)
        self.assertEqual(original.state_dict(),restored.state_dict())

    def test_depth_loss_uses_reference_validity_even_if_model_predicts_invalid(self):
        ref=torch.ones(1,3,4)*1000;valid=torch.ones_like(ref,dtype=torch.bool)
        a=torch.zeros_like(ref,requires_grad=True);logits=torch.ones_like(ref)*-50
        loss=depth_loss(a,logits,ref,valid);loss.backward()
        self.assertGreater(float(a.grad.abs().sum()),0)

    def test_missing_history_never_creates_a_truth_query(self):
        y=dict(state=torch.zeros(16,2,11))
        ref=reference_for_queries({'supervision':y},torch.empty(0,dtype=torch.long))
        self.assertEqual(ref['mask'].shape,(16,2,0,24,36))

    def test_causal_cache_checkpoint_matches_unwrapped_values_and_gradients(self):
        # Exercise the same two-write cache protocol across multiple chunks.
        class Residual(torch.nn.Module):
            def forward(self,x,feat_cache=None,feat_idx=None):
                i=feat_idx[0];old=feat_cache[i]
                y=torch.sin(x)+(0 if old is None else old*.3)
                feat_cache[i]=x;feat_idx[0]+=1
                j=feat_idx[0];old=feat_cache[j]
                z=y.square()+(0 if old is None else old*.2)
                feat_cache[j]=y;feat_idx[0]+=1
                return z+x
        def run(module,inputs):
            cache=[None,None];values=[]
            for x in inputs: values.append(module(x,cache,[0]))
            loss=sum(v.sum() for v in values)
            return torch.stack(values),torch.autograd.grad(loss,inputs)
        torch.manual_seed(5)
        xs=tuple(torch.randn(2,requires_grad=True) for _ in range(5))
        original,go=run(Residual(),xs)
        wrapped,gw=run(CheckpointCausalResidual(Residual()),xs)
        torch.testing.assert_close(wrapped,original)
        for a,b in zip(go,gw): torch.testing.assert_close(a,b)


if __name__=='__main__': unittest.main()
