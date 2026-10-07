import unittest
import numpy as np
import torch
from .inference import predict_pair,reader_implementation,POSTPROCESSING_VERSION
from experiments.formal_evaluation.test_clean_reference import FixedReader
from experiments.formal_evaluation.predicted_box_mask import clip_numpy


class PredictedBoxInferenceTest(unittest.TestCase):
    def run_reader(self,model):
        rgb=np.zeros((2,16,16,3),np.uint8);depth=np.zeros((2,16,16))
        K=np.array([[16.,0,8],[0,16,8],[0,0,1.]])
        poses=np.broadcast_to(np.eye(4),(2,4,4)).copy()
        return predict_pair(model,rgb,depth,depth.astype(bool),K,poses)

    def test_mask_is_clipped_after_resize_without_changing_scores_or_boxes(self):
        class Filled(FixedReader):
            def forward(self,**inputs):
                out=super().forward(**inputs);out['mask_logits']=torch.full_like(out['mask_logits'],2.)
                return out
        result=self.run_reader(Filled())
        expected=clip_numpy(np.full((16,16),torch.sigmoid(torch.tensor(2.)).item(),np.float32),[4,4,12,12])
        for view in ('air','ground'):
            self.assertEqual(len(result[view]),1);d=result[view][0]
            self.assertEqual(d['box_xyxy'],[4.,4.,12.,12.])
            self.assertEqual(d['class_id'],1)
            np.testing.assert_allclose(d['mask'],expected,rtol=1e-6)
            self.assertEqual(d['reader_postprocessing'],POSTPROCESSING_VERSION)
            self.assertGreater(d['score'],.9)

    def test_all_negative_mask_retains_image_detection_and_null_geometry(self):
        class Empty(FixedReader):
            def forward(self,**inputs):
                out=super().forward(**inputs);out['mask_logits']=torch.full_like(out['mask_logits'],-20.)
                return out
        result=self.run_reader(Empty())
        self.assertEqual(len(result['air']),1)
        self.assertFalse((result['air'][0]['mask']>=.5).any())
        self.assertIsNone(result['air'][0]['position_world_m'])

    def test_provenance_includes_actual_helper_source(self):
        from pathlib import Path
        from .data import sha256
        from experiments.formal_evaluation import predicted_box_mask
        result=reader_implementation()
        self.assertEqual(result['reader_source_sha256']['formal_evaluation/predicted_box_mask.py'],sha256(Path(predicted_box_mask.__file__)))
        self.assertEqual(result['reader_postprocessing'],'retain-2d-box-mask-and-exclude-floor-proxy-v4')


if __name__=='__main__':unittest.main()
