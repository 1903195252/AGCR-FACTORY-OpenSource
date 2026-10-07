import unittest
import numpy as np
from .metrics import depth_all,exit_assignment,reference_correspondence_error,matched_indices,visibility_groups
from .statistics import holm,paired_cluster_bootstrap,mean_reducer
from .tracking import historical_identity_map,event_identity_metrics
from .video import frechet


def row(frame,view='air',track=1,box=(1,1,20,20),visible=True):
    return dict(episode='ep',frame=frame,view=view,canvas_size=[24,24],detections=[dict(track_id=track,class_id=1,box_xyxy=list(box),score=1.,visible=visible,present=True,position_world_m=[0,0,1])])


class ReferencePopulationTests(unittest.TestCase):
    def test_invalid_prediction_does_not_shrink_depth(self):
        r=np.full((2,3),1000.);v=np.ones_like(r,bool);p=np.full_like(r,1100.);q=np.zeros_like(r)
        x=depth_all(p,q,r,v)
        self.assertEqual(x['depth_MAE_mm'],100.);self.assertEqual(x['sums']['depth_count'],6)
        p[0,0]=np.nan;x=depth_all(p,q,r,v)
        self.assertAlmostEqual(x['depth_MAE_mm'],(59900+500)/6)
    def test_reference_correspondences_do_not_use_pred_validity(self):
        d=np.full((1,2,8,8),1000.);valid=np.ones_like(d,bool);K=np.array([[10,0,4],[0,10,4],[0,0,1.]])
        C=np.broadcast_to(np.eye(4),(1,2,4,4));pred=d.copy()
        score=reference_correspondence_error(pred,d,valid,K,C)
        self.assertEqual(score['surface_correspondence_error_m'],0.);self.assertGreater(score['reference_correspondence_count'],0)
        pred[:,0]=np.nan;bad=reference_correspondence_error(pred,d,valid,K,C)
        self.assertEqual(bad['reference_correspondence_count'],score['reference_correspondence_count']);self.assertAlmostEqual(bad['surface_correspondence_error_m'],59.9)
    def test_history_mapping_does_not_use_future(self):
        refs=[];pred=[]
        for t in range(8):
            for v in ('air','ground'):refs.append(row(t,v));pred.append(row(t,v,track=5 if t<6 else 9))
        self.assertEqual(historical_identity_map(refs,pred),{'1':'5'})
    def test_R3_missing_preocclusion_identity_counts_failure(self):
        refs=[];pred=[]
        for t in range(8):
            for v in ('air','ground'):
                refs.append(row(t,v,visible=t<2 or t>=5));p=row(t,v,track=5)
                if t<5:p['detections']=[]
                pred.append(p)
        e=[dict(episode='ep',view='ground',track_id=1,hidden_start=2,reappearance=5)]
        score=event_identity_metrics(refs,pred,e);self.assertEqual(score['R3'],0.);self.assertEqual(score['R3_events'],1)
    def test_R3_global_identity_from_partner_is_valid(self):
        refs=[];pred=[]
        for t in range(8):
            for v in ('air','ground'):
                refs.append(row(t,v,visible=t<2 or t>=5));p=row(t,v,track=7)
                if 2<=t<5:p['detections']=[]
                pred.append(p)
        e=[dict(episode='ep',view='ground',track_id=1,hidden_start=2,reappearance=5)]
        self.assertEqual(event_identity_metrics(refs,pred,e)['R3'],1.)
    def test_generated_cross_view_scores_exclude_conditioning_history(self):
        refs=[];pred=[]
        for t in range(8):
            for v in ('air','ground'):
                refs.append(row(t,v));pred.append(row(t,v,track=1 if t<6 or v=='air' else 2))
        score=event_identity_metrics(refs,pred,[],score_from_frame=6)
        self.assertEqual(score['cross_view_ID_agreement'],0.);self.assertEqual(score['jointly_visible_count'],2)
    def test_exit_missing_is_other_no_nearest_future(self):
        regions=[dict(min_xyz_m=[-2,-1,-1],max_xyz_m=[-1,1,1]),dict(min_xyz_m=[1,-1,-1],max_xyz_m=[2,1,1])]
        self.assertEqual(exit_assignment([0,0,0],regions),-1);self.assertEqual(exit_assignment(None,regions),-1);self.assertEqual(exit_assignment([1.5,0,0],regions),1)
    def test_class_mismatch_cannot_match(self):
        a=row(0)['detections'];b=[dict(a[0],class_id=2)]
        self.assertEqual(matched_indices(a,b)[0],[])
    def test_dense_matching_prioritizes_cardinality(self):
        from unittest.mock import patch
        matrix=np.eye(7)*.51+np.eye(7,k=1)
        detections=[dict(row(0)['detections'][0],track_id=k) for k in range(7)]
        with patch('experiments.formal_evaluation.metrics.box_iou',return_value=matrix):
            matched,_=matched_indices(detections,detections)
        self.assertEqual(len(matched),7)
    def test_four_visibility_groups_partition(self):
        v=np.array([[[1,1,0,0],[1,0,1,0]]],bool)
        groups=visibility_groups(v);self.assertTrue(np.all(sum(groups.values())==1))
    def test_holm_monotonic_adjustment(self):self.assertTrue(np.allclose(holm([.01,.04,.03]),[.03,.06,.06]))
    def test_cluster_bootstrap_keeps_all_seeds(self):
        a=[];b=[]
        for layout in (0,1):
            for parent in (0,1):
                for seed in (17,29,43):
                    r=dict(layout=layout,scene_parent_id=str(parent),unit_id='u',training_seed=seed,metrics={'x':1.})
                    a.append(r);b.append(dict(r,metrics={'x':2.}))
        q=paired_cluster_bootstrap(a,b,mean_reducer('x'),40);self.assertEqual(q['difference'],-1.);self.assertEqual(q['ci95'],[-1.,-1.])
        with self.assertRaises(ValueError):paired_cluster_bootstrap(a,b[:-1],mean_reducer('x'),40)
    def test_frechet_singular_features(self):
        x=np.array([[1.,2,3],[1,2,3],[1,2,3]]);self.assertEqual(frechet(x,x),0.)
        self.assertAlmostEqual(frechet(x,x+1),3.)
    def test_reference_tf1_resize_origin_and_scale(self):
        import torch
        from .video import reference_resize224
        x=torch.arange(12,dtype=torch.float32).reshape(1,1,3,4,1).expand(1,1,3,4,3)
        y=reference_resize224(x)
        # At output (112,112), TF1 maps to input (1.5,2), value 8.
        self.assertAlmostEqual(float((y[0,0,0,112,112]+1)*127.5),8.,places=4)

if __name__=='__main__':unittest.main()
