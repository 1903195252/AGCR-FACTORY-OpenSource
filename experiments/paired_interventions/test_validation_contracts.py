"""Ablation training switches must not change held-out checkpoint ranking."""
import unittest
import torch
from .validation import validate


class SelectionContracts(unittest.TestCase):
    def test_camera_training_switch_does_not_change_selection_score(self):
        class Model:
            phase='video'
            def __init__(self,camera):
                self.camera=camera;self.upstream=torch.nn.Identity();self.reader=torch.nn.Identity()
            def feedback(self,family,generator,world,baseline):
                reappearance=torch.tensor(2.);camera=torch.tensor(3.)
                training_loss=.1*reappearance+.1*camera*float(self.camera)
                return training_loss,dict(reappearance=reappearance,camera=camera)
        dataset=[dict(metadata=dict(family_id='fixed_validation_family'))]
        reports=[validate(Model(camera),dataset,[0],'cpu',42000) for camera in (False,True)]
        self.assertAlmostEqual(reports[0]['mean'],.5)
        self.assertEqual(reports[0]['mean'],reports[1]['mean'])
        self.assertEqual(reports[0]['rows'][0]['scope'],'fixed_reappearance_camera_mean_of_8')


if __name__=='__main__':unittest.main()
