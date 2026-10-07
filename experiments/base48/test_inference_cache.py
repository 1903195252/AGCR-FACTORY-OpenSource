import copy
import unittest

import torch
from torch import nn

from experiments.base48.inference_cache import inference_condition_cache
from experiments.paired_interventions.sampling import pure_noise_euler


class CountLinear(nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return super().forward(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = CountLinear()
        self.calls = 0
    def forward(self, condition):
        self.calls += 1
        return dict(layer=self.map(condition['geometry']))


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_key, self.geo_value, self.query = CountLinear(), CountLinear(), CountLinear()
    def forward(self, x, condition):
        key, value = self.geo_key(condition['layer']), self.geo_value(condition['layer'])
        return x+((self.query(x)@key.T).softmax(-1)@value)*.01


class Wan(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.adapters = nn.ModuleList([Adapter() for _ in range(10)])
        self.calls = 0
    def forward(self, x, condition):
        self.calls += 1
        shape = x.shape
        x = x.reshape(-1, 4)
        encoded = self.encoder(condition)
        for adapter in self.adapters:
            x = adapter(x, encoded)
        return x.reshape(shape)


class InferenceCacheTest(unittest.TestCase):
    @torch.no_grad()
    def test_actual_euler_detached_condition_aliases_hit_cache_at_all_fifty_steps(self):
        torch.manual_seed(71)
        original = Wan().eval()
        cached = copy.deepcopy(original)
        condition = dict(geometry=torch.randn(5, 4))
        noise, anchor = torch.randn(2, 4, 4), torch.randn(2, 1, 4)
        expected = pure_noise_euler(lambda z, t, c: original(z, c), noise, anchor, condition, 50, 2)
        with inference_condition_cache(cached) as stats:
            actual = pure_noise_euler(lambda z, t, c: cached(z, c), noise, anchor, condition, 50, 2)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(cached.calls, 50)
        self.assertEqual(cached.encoder.calls, 1)
        self.assertEqual(stats['hits'], 49*21)
        self.assertEqual(stats['misses'], 21)

    @torch.no_grad()
    def test_fifty_calls_and_queries_unchanged_exact_output_and_no_rng_changes(self):
        torch.manual_seed(17)
        original = Wan().eval()
        cached = copy.deepcopy(original)
        condition = dict(geometry=torch.randn(5, 4))
        old, new = torch.randn(3, 4), None
        new = old.clone()
        for _ in range(50): old = original(old, condition)
        rng = torch.get_rng_state().clone()
        with inference_condition_cache(cached) as stats:
            for _ in range(50): new = cached(new, condition)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(cached.calls, 50)
        self.assertEqual(cached.encoder.calls, 1)
        self.assertEqual(stats['hits'], 49*21)
        for adapter in cached.adapters:
            self.assertEqual(adapter.geo_key.calls, 1)
            self.assertEqual(adapter.geo_value.calls, 1)
            self.assertEqual(adapter.query.calls, 50)
            self.assertNotIn('forward', adapter.geo_key.__dict__)
        self.assertNotIn('forward', cached.encoder.__dict__)
        self.assertFalse(hasattr(cached, '_base_condition_cache_active'))

    @torch.no_grad()
    def test_input_weight_output_mutations_and_new_condition_invalidate(self):
        model = Wan().eval()
        condition = dict(geometry=torch.ones(5, 4))
        with inference_condition_cache(model):
            first = model.encoder(condition)
            self.assertIs(model.encoder(condition), first)
            condition['geometry'].add_(1)
            second = model.encoder(condition)
            self.assertIsNot(first, second)
            model.encoder.map.weight.add_(.1)
            third = model.encoder(condition)
            self.assertIsNot(second, third)
            third['layer'].zero_()
            fourth = model.encoder(condition)
            self.assertIsNot(third, fourth)
            other = dict(geometry=condition['geometry'].clone())
            self.assertIsNot(model.encoder(other), fourth)
        self.assertEqual(model.encoder.calls, 5)

    def test_training_grad_nested_and_exception_restoration(self):
        model = Wan().eval()
        with self.assertRaisesRegex(RuntimeError, 'no-grad'):
            with inference_condition_cache(model): pass
        with torch.no_grad():
            model.train()
            with self.assertRaisesRegex(RuntimeError, 'eval'):
                with inference_condition_cache(model): pass
            model.eval()
            with self.assertRaisesRegex(ValueError, 'abort'):
                with inference_condition_cache(model):
                    with self.assertRaisesRegex(RuntimeError, 'Nested'):
                        with inference_condition_cache(model): pass
                    raise ValueError('abort')
        self.assertNotIn('forward', model.encoder.__dict__)
        self.assertNotIn('forward', model.adapters[0].geo_key.__dict__)

    def test_inference_mode_unversioned_tensors_are_not_cached(self):
        model = Wan().eval()
        with torch.inference_mode():
            condition = dict(geometry=torch.ones(5, 4))
            with inference_condition_cache(model) as stats:
                first = model.encoder(condition)
                second = model.encoder(condition)
            self.assertTrue(torch.equal(first['layer'], second['layer']))
            self.assertEqual(stats['hits'], 0)
            self.assertEqual(model.encoder.calls, 2)


if __name__ == '__main__':
    unittest.main()
