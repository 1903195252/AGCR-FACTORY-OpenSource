import unittest
import torch
from .perturbations import degrade_history

class HistoryPerturbationTests(unittest.TestCase):
    def inputs(self):
        return dict(rgb=torch.arange(6*2*8*8*3).reshape(6,2,8,8,3).float(),depth_mm=torch.full((6,2,8,8),1000.),depth_valid=torch.ones(6,2,8,8,dtype=torch.bool),
            K=torch.eye(3),camera_to_world=torch.eye(4).expand(6,2,4,4).clone(),time_s=torch.arange(6).float()/5-1,
            planned_camera_to_world=torch.eye(4).expand(16,2,4,4).clone(),future_time_s=torch.arange(1,17).float()/5)
    def test_delay_preserves_real_timestamps_and_masks_late_frames(self):
        x=self.inputs();y=degrade_history(x,{'kind':'partner_delay','seconds':.6},17,partner=0)
        self.assertTrue(torch.equal(x['time_s'],y['time_s']));self.assertEqual(y['observation_available'][:,0].tolist(),[True,True,True,False,False,False])
        self.assertTrue(y['observation_available'][:,1].all());self.assertTrue(x['depth_valid'].all())
    def test_missing_depth_exact_reference_fraction(self):
        x=self.inputs();y=degrade_history(x,{'kind':'depth_missing','fraction':.5},17)
        self.assertTrue(torch.equal(y['depth_valid'].sum((-1,-2)),torch.full((6,2),32)))
        z=degrade_history(x,{'kind':'depth_missing','fraction':.5},17);self.assertTrue(torch.equal(y['depth_valid'],z['depth_valid']))
    def test_noise_never_changes_future_condition(self):
        x=self.inputs();y=degrade_history(x,{'kind':'depth_noise','beta':.02},17)
        self.assertTrue(torch.equal(x['planned_camera_to_world'],y['planned_camera_to_world']));self.assertFalse(torch.equal(x['depth_mm'],y['depth_mm']))
    def test_reference_payload_rejected(self):
        x=self.inputs();x['future_rgb']=torch.zeros(1)
        with self.assertRaises(ValueError):degrade_history(x,{'kind':'clean'},17)
    def test_range_removes_only_unmeasurable_history(self):
        x=self.inputs();x['depth_mm'][0,0,0,0]=15001
        y=degrade_history(x,{'kind':'depth_range','maximum_mm':15000},17)
        self.assertEqual(y['depth_mm'][0,0,0,0],0);self.assertEqual(int(y['depth_valid'].sum()),int(x['depth_valid'].sum())-1)
        self.assertTrue(torch.equal(y['planned_camera_to_world'],x['planned_camera_to_world']))
    def test_misregistration_uses_only_older_evidence_and_keeps_declared_times(self):
        x=self.inputs();y=degrade_history(x,{'kind':'partner_time_misalignment','frames':1},17,partner=1)
        self.assertTrue(torch.equal(y['rgb'][1:,1],x['rgb'][:-1,1]))
        self.assertTrue(torch.equal(y['rgb'][:,0],x['rgb'][:,0]))
        self.assertTrue(torch.equal(y['time_s'],x['time_s']));self.assertTrue(y['observation_available'].all())
    def test_extrinsic_bias_is_rigid_repeatable_and_shared_with_planned_path(self):
        x=self.inputs();config={'kind':'extrinsic_calibration','translation_std_m':.02,'rotation_std_deg':1.}
        y=degrade_history(x,config,17);z=degrade_history(x,config,17)
        self.assertTrue(torch.equal(y['camera_to_world'],z['camera_to_world']))
        torch.testing.assert_close(y['camera_to_world'][0],y['planned_camera_to_world'][0])
        rotation=y['camera_to_world'][0,:,:3,:3]
        torch.testing.assert_close(rotation.transpose(-1,-2)@rotation,torch.eye(3).expand(2,3,3))
        torch.testing.assert_close(torch.linalg.det(rotation),torch.ones(2))
        self.assertTrue(torch.equal(y['rgb'],x['rgb']));self.assertTrue(torch.equal(y['depth_mm'],x['depth_mm']))
        self.assertTrue(torch.equal(x['camera_to_world'],torch.eye(4).expand(6,2,4,4)))
    def test_changed_perturbation_source_cannot_reuse_registered_config(self):
        with self.assertRaisesRegex(ValueError,'implementation differs'):
            degrade_history(self.inputs(),{'kind':'clean','implementation_sha256':'outdated'},17)
if __name__=='__main__':unittest.main()
