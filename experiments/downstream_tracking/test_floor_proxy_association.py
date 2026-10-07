import unittest
import numpy as np
from .association import CooperativeTracker,DEFAULT


class FloorProxyAssociationTest(unittest.TestCase):
    def detection(self,point,scope='floor_projection'):
        return dict(class_id=1,score=.9,box_xyxy=[0,0,10,10],identity_embedding=[1.,0.],
            position_world_m=point,center_covariance_world_m2=(np.eye(3)*.25).tolist(),localization_scope=scope)

    def test_proxy_never_gates_or_updates_metric_track_and_is_retained(self):
        tracker=CooperativeTracker();a=self.detection([0.,0.,0.]);b=self.detection([100.,0.,0.])
        out=tracker.update({'air':[a],'ground':[b]},0)
        self.assertEqual(out['air'][0]['track_id'],out['ground'][0]['track_id'])
        self.assertEqual(out['ground'][0]['position_world_m'],[100.,0.,0.])
        self.assertEqual(out['ground'][0]['center_covariance_world_m2'],b['center_covariance_world_m2'])
        self.assertFalse(out['ground'][0]['association_geometry_available'])
        self.assertIsNone(out['ground'][0]['filtered_position_world_m'])
        self.assertEqual(tracker.config,DEFAULT)

    def test_valid_metric_geometry_still_rejects_implausible_pair(self):
        tracker=CooperativeTracker()
        out=tracker.update({'air':[self.detection([0.,0.,1.],'predicted_center_3d')],
                            'ground':[self.detection([100.,0.,1.],'predicted_center_3d')]},0)
        self.assertNotEqual(out['air'][0]['track_id'],out['ground'][0]['track_id'])
        self.assertTrue(out['ground'][0]['association_geometry_available'])

    def test_proxy_does_not_pull_existing_reliable_world_state(self):
        tracker=CooperativeTracker();valid=self.detection([2.,3.,1.],'predicted_center_3d')
        first=tracker.update({'air':[valid]},0)['air'][0]
        second=tracker.update({'ground':[self.detection([100.,0.,0.])]},.2)['ground'][0]
        self.assertEqual(first['track_id'],second['track_id'])
        np.testing.assert_allclose(second['filtered_position_world_m'],[2.,3.,1.])
        self.assertFalse(second['association_geometry_available'])


if __name__=='__main__':unittest.main()
