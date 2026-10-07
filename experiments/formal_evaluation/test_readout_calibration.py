import unittest
import numpy as np
from .readout import evaluation_geometry


class FixedMeasurementCalibration(unittest.TestCase):
    def setUp(self):
        self.shape=(22,2,4,6,3)
        self.K=np.broadcast_to(np.eye(3),(22,2,3,3)).copy()
        self.poses=np.broadcast_to(np.eye(4),(22,2,4,4)).copy()
        self.times=np.arange(22)/5.
        self.generated=dict(rgb=np.zeros((16,2,4,6,3),np.uint8),camera_to_world=self.poses[6:].copy())

    def check(self):
        return evaluation_geometry(self.generated,self.shape,self.K,self.poses,self.times)

    def test_conditioning_calibration_error_does_not_change_measurement(self):
        self.generated['camera_to_world'][...,0,3]+=10.
        self.generated['K']=self.K[6:]*5.
        K,poses=self.check()
        np.testing.assert_array_equal(K,self.K)
        np.testing.assert_array_equal(poses,self.poses)
        # No future reference RGB, labels or state are provided to this API.

    def test_misaligned_generated_frames_views_or_resolution_rejected(self):
        for shape in ((15,2,4,6,3),(16,1,4,6,3),(16,2,5,6,3)):
            self.generated['rgb']=np.zeros(shape,np.uint8)
            with self.assertRaisesRegex(ValueError,'axes'):self.check()

    def test_static_intrinsics_broadcast_and_invalid_sensor_axes_rejected(self):
        self.K=np.eye(3)
        self.assertEqual(self.check()[0].shape,(22,2,3,3))
        self.generated['camera_to_world']=np.zeros((16,4,4))
        with self.assertRaisesRegex(ValueError,'axes'):self.check()

    def test_disordered_reference_time_rejected(self):
        self.times[7]=self.times[6]
        with self.assertRaisesRegex(ValueError,'timestamps'):self.check()


if __name__=='__main__':unittest.main()
