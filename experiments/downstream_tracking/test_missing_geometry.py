"""Valid image detections must survive unavailable ground-plane geometry."""
import unittest
import numpy as np
from .inference import predict_pair
from .association import CooperativeTracker
from experiments.formal_evaluation.test_clean_reference import FixedReader


class MissingGeometryTest(unittest.TestCase):
    def test_above_horizon_box_is_retained_without_inventing_depth(self):
        reader=FixedReader();rgb=np.zeros((2,16,16,3),np.uint8);z=np.zeros((2,16,16))
        K=np.array([[16.,0,8],[0,16,8],[0,0,1.]])
        T=np.broadcast_to(np.eye(4),(2,4,4)).copy()
        T[:,:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];T[:,2,3]=1
        result=predict_pair(reader,rgb,z,z.astype(bool),K,T)
        self.assertEqual(len(result['air']),1)
        self.assertIsNone(result['air'][0]['position_world_m'])
        self.assertEqual(result['air'][0]['localization_scope'],'unavailable')
        self.assertEqual(result['air'][0]['box_xyxy'],[4.,4.,12.,12.])
        self.assertTrue(np.isfinite(result['ground'][0]['position_world_m']).all())

    def test_unknown_position_stays_null_and_cross_view_id_is_appearance_based(self):
        tracker=CooperativeTracker()
        d=dict(class_id=4,score=.9,box_xyxy=[1,2,5,8],position_world_m=None,
            center_covariance_world_m2=None,identity_embedding=[1.,0.],localization_scope='unavailable')
        a=tracker.update({'air':[d]},0)['air'][0]
        b=tracker.update({'ground':[d]},.2)['ground'][0]
        self.assertEqual(a['track_id'],b['track_id'])
        for row in (a,b):
            self.assertIsNone(row['position_world_m'])
            self.assertIsNone(row['filtered_position_world_m'])
            self.assertIsNone(row['velocity_world_mps'])
        other=tracker.update({'air':[dict(d,identity_embedding=[0.,1.])]},.4)['air'][0]
        self.assertNotEqual(a['track_id'],other['track_id'])

    def test_geometry_can_become_available_without_resetting_identity(self):
        tracker=CooperativeTracker()
        d=dict(class_id=1,score=.9,box_xyxy=[1,2,5,8],position_world_m=None,identity_embedding=[1.,0.])
        first=tracker.update({'air':[d]},0)['air'][0]
        second=tracker.update({'ground':[dict(d,position_world_m=[2.,3.,1.])]},.2)['ground'][0]
        self.assertEqual(first['track_id'],second['track_id'])
        np.testing.assert_allclose(second['filtered_position_world_m'],[2,3,1])
        third=tracker.update({'air':[d]},.4)['air'][0]
        self.assertIsNone(third['position_world_m'])
        self.assertEqual(third['track_id'],first['track_id'])

    def test_airborne_class_never_gets_fictitious_floor_center(self):
        import torch
        class UAVReader(FixedReader):
            def forward(self,**inputs):
                out=super().forward(**inputs)
                out['logits']=torch.tensor([[[-9.,-9.,-9.,9.]],[[-9.,-9.,-9.,9.]]])
                return out
        rgb=np.zeros((2,16,16,3),np.uint8);z=np.zeros((2,16,16));T=np.broadcast_to(np.eye(4),(2,4,4)).copy()
        T[:,:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];T[:,2,3]=1
        out=predict_pair(UAVReader(),rgb,z,z.astype(bool),np.array([[16.,0,8],[0,16,8],[0,0,1.]]),T)
        self.assertEqual(len(out['ground']),1)
        self.assertIsNone(out['ground'][0]['position_world_m'])
        self.assertEqual(out['ground'][0]['geometry_missing_reason'],'airborne_class_has_no_floor_contact')

    def test_missing_or_proxy_geometry_never_enters_metric_center_error(self):
        from experiments.formal_evaluation.tracking import event_identity_metrics
        from experiments.formal_evaluation.metrics import metric_position
        from experiments.formal_evaluation.downstream import zone_events
        ref=[];pred=[]
        for view in ('air','ground'):
            meta=dict(episode='e',frame=0,view=view,time_s=0)
            target=dict(track_id=1,class_id=1,box_xyxy=[0,0,10,10],position_world_m=[1,2,3])
            ref.append(dict(meta,detections=[target]))
            pred.append(dict(meta,detections=[dict(target,track_id=7,position_world_m=None,localization_scope='unavailable')]))
        result=event_identity_metrics(ref,pred,[])
        self.assertEqual(result['jointly_detected_count'],1)
        self.assertIsNone(result['cross_view_center_error_m'])
        self.assertEqual(result['center_count'],0)
        self.assertIsNone(result['R3'])
        self.assertEqual(zone_events(pred,[dict(id='z',polygon=[[0,0],[4,0],[4,4],[0,4]])]),[])
        self.assertIsNone(metric_position(dict(position_world_m=[0,0,0],localization_scope='floor_projection')))

    def test_same_class_unknown_geometry_keeps_multiple_targets_one_to_one(self):
        tracker=CooperativeTracker()
        one=dict(class_id=4,score=.9,position_world_m=None,identity_embedding=[1.,0.])
        two=dict(one,identity_embedding=[0.,1.])
        out=tracker.update({'air':[one,two],'ground':[two,one]},0)
        self.assertEqual(len({r['track_id'] for r in out['air']}),2)
        self.assertEqual(len({r['track_id'] for r in out['ground']}),2)
        self.assertEqual(out['air'][0]['track_id'],out['ground'][1]['track_id'])
        self.assertEqual(out['air'][1]['track_id'],out['ground'][0]['track_id'])
        ambiguous=CooperativeTracker().update({'air':[one,one],'ground':[one,one]},0)
        # Exact descriptor ties have no physical identity proof, but must not
        # collapse two detections in one image into one artificial world point.
        self.assertEqual(len({r['track_id'] for r in ambiguous['ground']}),2)
        self.assertTrue(all(r['position_world_m'] is None for r in ambiguous['ground']))


if __name__=='__main__':unittest.main()
