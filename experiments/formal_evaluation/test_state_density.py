import unittest
import torch
from .state_likelihood import innovation_nll
from experiments.paper_experiments.agcr_exp.future_prediction import JointDynamics

class StateDensityTests(unittest.TestCase):
    def test_density_matches_existing_mixture_likelihood(self):
        old=torch.get_num_threads();torch.set_num_threads(1)
        try:
            stats=dict(state=dict(mean=[0.]*11,std=[1.]*11),acceleration=dict(mean=[0.]*3,std=[1.]*3),class_velocity={str(k):dict(mean=[0.,0,0],covariance=torch.eye(3).tolist()) for k in range(1,5)})
            torch.manual_seed(17);m=JointDynamics(stats).eval();tokens=torch.randn(2,256);scene=torch.randn(1,4,256)
            initial=torch.zeros(2,11);initial[:,8:]=1;reference=initial[None].repeat(16,1,1);reference[:,:,3]=.2
            eligible=torch.tensor([True,False]);present=eligible[None].expand(16,-1)
            with torch.no_grad():
                _,terms,_=m.objective(tokens,scene,initial,initial,reference,eligible,present)
                ours=innovation_nll(m,tokens,scene,initial,reference,eligible,present)
            self.assertAlmostEqual(ours['per_active_step_NLL'],float(terms['dynamics_nll']),places=5)
            self.assertEqual(ours['active_steps'],16)
        finally:torch.set_num_threads(old)
if __name__=='__main__':unittest.main()
