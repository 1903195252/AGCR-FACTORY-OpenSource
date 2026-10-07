import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .ranked_boxes import export_arrays,official_ranked,save_sidecar,load_sidecar,ranked_records


class RankedBoxTest(unittest.TestCase):
    def test_capture_detaches_without_changing_actual_output(self):
        boxes=torch.rand(2,300,4,requires_grad=True);logits=torch.randn(2,300,4,requires_grad=True)
        a=export_arrays(dict(boxes=boxes,logits=logits));a['boxes'][0,0]=0
        self.assertTrue(boxes.requires_grad);self.assertFalse(torch.equal(boxes[0,0],torch.zeros(4)))
        np.testing.assert_array_equal(a['logits'],logits.detach().numpy())

    def test_hf_focal_topk_allows_two_classes_for_one_query_and_no_clip(self):
        try:from transformers import RTDetrImageProcessor
        except ImportError:self.skipTest('Official HF backend checked in actual remote environment')
        boxes=np.array([[[.02,.5,.2,.2],[.8,.5,.1,.1],[.5,.5,.1,.1]]],np.float32)
        logits=np.array([[[3.,2.,-10.,-10.],[-5.,-5.,1.,-5.],[-8.,-8.,-8.,-8.]]],np.float32)
        output=official_ranked(boxes,logits,(480,720))[0]
        self.assertEqual([d['query_index'] for d in output],[0,0,1]);self.assertEqual([d['class_id'] for d in output],[1,2,3])
        self.assertLess(output[0]['box_xyxy'][0],0)
        ref=RTDetrImageProcessor().post_process_object_detection(SimpleNamespace(logits=torch.tensor(logits),pred_boxes=torch.tensor(boxes)),threshold=-1,target_sizes=[(480,720)],use_focal_loss=True)[0]
        np.testing.assert_array_equal([d['box_xyxy'] for d in output],ref['boxes'].numpy())

    def test_all_below_tracking_threshold_survive_and_history_is_not_scored(self):
        try:from transformers import RTDetrImageProcessor
        except ImportError:self.skipTest('Official HF backend checked remotely')
        b=np.full((2,2,300,4),.2,np.float32);l=np.full_like(b,-4.)
        rows=ranked_records(b,l,dict(episode='e'),np.array([0.,.2]),score_from_frame=1)
        self.assertEqual(len(rows),2);self.assertTrue(all(r['frame']==1 for r in rows))
        self.assertEqual(len(rows[0]['detections']),300);self.assertLess(rows[0]['detections'][0]['score'],.5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.npz';save_sidecar(path,b,l,(480,720),dict(pixel_sha='fixed',reader_sha='reader'))
            restored,_=load_sidecar(path);np.testing.assert_array_equal(restored['logits'],l)
            with self.assertRaises(ValueError):load_sidecar(path,dict(pixel_sha='wrong'))
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):load_sidecar(path)


if __name__=='__main__':unittest.main()
