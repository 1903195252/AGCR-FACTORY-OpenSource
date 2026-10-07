import unittest
import numpy as np
from .condition_adherence import support_boxes,soft_box_iou,assignment_values,analyze_frame


class ConditionAdherenceTest(unittest.TestCase):
    def test_native_crop_coordinates_and_soft_mask_overlap(self):
        masks=np.zeros((1,60,90));masks[0,2:4,3:5]=1
        boxes,valid=support_boxes(masks);np.testing.assert_array_equal(boxes,[[24,16,40,32]])
        self.assertTrue(valid[0]);self.assertAlmostEqual(soft_box_iou(masks,boxes)[0,0],1.)
        self.assertAlmostEqual(soft_box_iou(masks*.5,boxes)[0,0],.5)

    def test_one_image_detection_cannot_satisfy_duplicate_slots(self):
        scores=np.ones((2,1));values,assignment=assignment_values(scores,[1,1],[1],np.ones(2,bool))
        self.assertEqual((assignment>=0).sum(),1);self.assertEqual(values.sum(),1)

    def test_occluded_condition_is_not_an_expected_visible_detection(self):
        p=np.zeros((2,60,92));p[:,2:4,4:6]=1;m=np.zeros_like(p);m[1]=p[1]
        row=analyze_frame(p,m,np.full_like(p,1000),p.astype(bool),np.ones(2,bool),np.ones(2,bool),[1,1],[])
        self.assertEqual(len(row['slots']),2)
        self.assertEqual(row['counts']['supported_slot_views'],2)
        self.assertEqual(row['counts']['supported_zero_visible_mass_slot_views'],1)
        for value in row['sensitivity'].values():self.assertEqual(value['condition_visible_slot_views'],1)


if __name__=='__main__':unittest.main()
