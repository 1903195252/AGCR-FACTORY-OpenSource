import unittest
import numpy as np
from .predicted_box_mask import clip_numpy,clip_torch


class PredictedMaskModuleTest(unittest.TestCase):
    def test_preserves_probabilities_and_handles_fractional_bounds(self):
        source=np.arange(20,dtype=np.float32).reshape(4,5)/20
        cropped=clip_numpy(source,[1.2,.2,3.1,2.8])
        expected=np.zeros_like(source);expected[:3,1:3]=source[:3,1:3]
        np.testing.assert_array_equal(cropped,expected)
        self.assertEqual(cropped.dtype,source.dtype)
        np.testing.assert_array_equal(source,np.arange(20,dtype=np.float32).reshape(4,5)/20)

    def test_instance_localization_does_not_merge_masks_or_drop_empty_instances(self):
        source=np.ones((2,3,4),bool)
        cropped=clip_numpy(source,[[0,0,1,3],[8,8,10,10]])
        self.assertEqual(cropped.shape,source.shape)
        self.assertEqual(cropped[0].sum(),3)
        self.assertFalse(cropped[1].any())
        self.assertEqual(clip_numpy(source,[0,0,1,3]).sum(),6)

    def test_torch_matches_numpy_and_only_passes_inbox_mask_gradients(self):
        import torch
        source=torch.ones((2,3,4),dtype=torch.float32,requires_grad=True)
        boxes=np.array([[-.2,0,1.6,4],[2.5,.5,4,2.5]])
        cropped=clip_torch(source,boxes)
        expected=clip_numpy(np.ones((2,3,4),np.float32),boxes)
        np.testing.assert_array_equal(cropped.detach().numpy(),expected)
        cropped.sum().backward()
        np.testing.assert_array_equal(source.grad.numpy(),expected)
        self.assertEqual(cropped.dtype,source.dtype)

    def test_invalid_coordinates_and_wrong_instance_grid_fail(self):
        for box in ([2,0,1,2],[0,0,float('nan'),2],[0,0,2]):
            with self.assertRaises(ValueError):clip_numpy(np.ones((3,4)),box)
        with self.assertRaises(ValueError):clip_numpy(np.ones((2,3,4)),np.ones((3,4)))


if __name__=='__main__':unittest.main()
