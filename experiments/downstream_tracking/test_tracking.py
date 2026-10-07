import inspect,math,types,unittest
import numpy as np
import torch
from torch import nn
from .association import CooperativeTracker
from .data import make_targets
from .model import CooperativeDetector,head_losses,roi_grid
from .train import index_at


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__();self.config=types.SimpleNamespace(d_model=16,num_queries=4)
        self.features=nn.Linear(3,16);self.cls=nn.Linear(16,4);self.box=nn.Linear(16,4)
        self.enc_cls=nn.Linear(16,4);self.enc_box=nn.Linear(16,4)
    def forward(self,pixel_values,labels=None,return_dict=True):
        q=self.features(pixel_values.mean((-1,-2)))[:,None].expand(-1,4,-1)
        logits=self.cls(q);boxes=self.box(q).sigmoid()
        return types.SimpleNamespace(last_hidden_state=q,logits=logits,pred_boxes=boxes,
            enc_topk_logits=self.enc_cls(q),enc_topk_bboxes=self.enc_box(q).sigmoid(),
            loss=logits.square().mean()+boxes.square().mean())


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    @staticmethod
    def isolated_head_inputs(frames=1):
        def leaf(value):
            return torch.tensor(value,dtype=torch.float32).repeat(frames,1,1).requires_grad_()
        prediction=dict(boxes=leaf([[[.5,.5,.25,.25]]]),logits=leaf([[[0.,1.,0.,0.]]]),
            contact=leaf([[[.2,.4]]]),identity=leaf([[[1.,0.]]]),
            mask_logits=torch.zeros(frames,1,2,2,requires_grad=True),
            center_camera_m=leaf([[[.2,.4,.6]]]),dimensions_m=leaf([[[1.,1.,1.]]]),
            heading_camera=leaf([[[1.,0.,0.]]]),center_var_m2=leaf([[[1.,1.,1.]]]))
        targets=[dict(boxes=torch.tensor([[.5,.5,.25,.25]]),class_labels=torch.tensor([1]),
            contact=torch.zeros(1,2),masks=torch.zeros(1,2,2),identity_keys=['episode:'+str(i)],
            center_camera_m=torch.zeros(1,3),dimensions_m=torch.ones(1,3),
            heading_camera=torch.tensor([[1.,0.,0.]])) for i in range(frames)]
        return prediction,targets

    def test_head_components_and_zero_detector_default_are_independent(self):
        # Different numbers of identical observations must preserve per-target losses.
        expected=dict(contact=.1,mask=math.log(2)+2/3,identity=0.,box3d=.28,covariance=.28)
        expected_total=expected['contact']+expected['mask']+expected['box3d']+.1*expected['covariance']
        for frames in (1,3):
            with self.subTest(frames=frames):
                prediction,targets=self.isolated_head_inputs(frames)
                losses=head_losses(prediction,targets)
                for name,value in expected.items():
                    self.assertAlmostEqual(float(losses[name]),value,places=6,msg=name)
                self.assertAlmostEqual(float(losses['total']),expected_total,places=6)
                prediction['detector_loss']=torch.tensor(2.,requires_grad=True)
                with_detector=head_losses(prediction,targets)
                self.assertAlmostEqual(float(with_detector['total']),expected_total+2,places=6)
                for name in expected:
                    self.assertEqual(float(with_detector[name]),float(losses[name]))

    def test_head_gradients_preserve_individual_weights(self):
        prediction,targets=self.isolated_head_inputs(3)
        losses=head_losses(prediction,targets)
        for name in ('mask','identity','box3d','covariance'):
            gradient=torch.autograd.grad(losses[name],prediction['contact'],retain_graph=True)[0]
            self.assertEqual(float(gradient.abs().sum()),0.,msg=name)
        losses['total'].backward()
        torch.testing.assert_close(prediction['contact'].grad,prediction['contact'].detach()/3)
        torch.testing.assert_close(prediction['center_camera_m'].grad,
                                   1.1*prediction['center_camera_m'].detach()/3)
        self.assertGreater(float(prediction['mask_logits'].grad.abs().sum()),0.)

    def test_image_only_forward_retains_encoder_candidates_for_supervision(self):
        model=CooperativeDetector(None,detector=TinyDetector())
        prediction=model(torch.zeros(1,3,16,16),torch.ones(1,16,16)*1000,
                         torch.ones(1,16,16,dtype=torch.bool),torch.eye(3)[None])
        self.assertNotIn('detector_loss',prediction)
        self.assertEqual(prediction['encoder_logits'].shape,(1,4,4))
        self.assertEqual(prediction['encoder_boxes'].shape,(1,4,4))
        self.assertTrue(prediction['encoder_logits'].requires_grad)
        self.assertTrue(prediction['encoder_boxes'].requires_grad)

    def test_vectorized_association_matches_pairwise_cost(self):
        rng=np.random.default_rng(123);tracker=CooperativeTracker()
        rows=[]
        for k in range(19):
            embedding=rng.normal(size=128);embedding/=np.linalg.norm(embedding)
            d=dict(class_id=k%3+1,position_world_m=rng.normal(size=3),identity_embedding=embedding,score=.9)
            if k<11:tracker._new(d,0)
            rows.append(d)
        keys=list(tracker.tracks);expected=np.full((len(rows),len(keys)),1e6)
        for i,d in enumerate(rows):
            z,R=tracker._measure(d)
            for j,key in enumerate(keys):
                tr=tracker.tracks[key]
                if d['class_id']!=tr['class_id']:continue
                delta=z-tr['mean'][:3];maha=delta@np.linalg.solve(tr['cov'][:3,:3]+R,delta)
                appearance=1-float(np.dot(d['identity_embedding'],tr['descriptor']))
                if maha<=tracker.config['mahalanobis_gate'] and appearance<=tracker.config['appearance_gate']:
                    w=tracker.config['appearance_weight'];expected[i,j]=w*appearance+(1-w)*maha/tracker.config['mahalanobis_gate']
        np.testing.assert_allclose(tracker._cost(rows,keys),expected,rtol=1e-12,atol=1e-12)

    def test_reader_job_cache_resume_and_input_identity(self):
        import tempfile,signal
        from pathlib import Path
        from .runtime import ReaderRuntime
        previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'manifest.json';source.write_text('{}')
                first=ReaderRuntime(root/'out.json',1,[source],{'role':'annotation'})
                first.put('clip1',{'done':True});first.stopped=True
                self.assertTrue(first.pause());self.assertFalse((root/'out.json').exists())
                resumed=ReaderRuntime(root/'out.json',1,[source],{'role':'annotation'})
                self.assertEqual(resumed.get('clip1'),{'done':True})
                source.write_text('{"changed":true}')
                with self.assertRaises(ValueError):ReaderRuntime(root/'out.json',1,[source],{'role':'annotation'})
        finally:
            for sig,value in previous.items():signal.signal(sig,value)

    def test_forward_no_reference_inputs_and_head_gradients(self):
        self.assertNotIn('state',inspect.signature(CooperativeDetector.forward).parameters)
        torch.manual_seed(3);m=CooperativeDetector(None,detector=TinyDetector())
        images=torch.randint(255,(2,3,24,32),dtype=torch.uint8);K=torch.tensor([[30.,0,16],[0,30,12],[0,0,1]])[None].repeat(2,1,1)
        targets=[]
        for i in range(2):
            targets.append(dict(boxes=torch.tensor([[.5,.5,.5,.5]]),class_labels=torch.tensor([1]),contact=torch.tensor([[.5,.7]]),
                masks=torch.ones(1,24,32),identity_keys=['episode:1'],center_camera_m=torch.ones(1,3),dimensions_m=torch.ones(1,3),
                heading_camera=torch.tensor([[1.,0,0]])))
        output=m(images,torch.ones(2,24,32)*1000,torch.ones(2,24,32,dtype=torch.bool),K,labels=targets)
        loss=head_losses(output,targets)['total'];loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(m.depth_head[-1].weight.grad.abs().sum(),0)
        self.assertGreater(m.mask_query.weight.grad.abs().sum(),0)
        self.assertEqual(output['mask_logits'].shape,(2,4,6,8))
    def test_empty_labels_are_valid_negative_detection_training(self):
        m=CooperativeDetector(None,detector=TinyDetector());K=torch.eye(3)[None]
        out=m(torch.zeros(1,3,16,16),torch.zeros(1,16,16),torch.zeros(1,16,16),K,labels=[dict(boxes=torch.zeros(0,4),class_labels=torch.zeros(0,dtype=torch.long))])
        loss=head_losses(out,[dict(boxes=torch.zeros(0,4),class_labels=torch.zeros(0,dtype=torch.long))])['total'];loss.backward()
        self.assertTrue(torch.isfinite(loss))
    def test_cross_view_handover_and_class_gate(self):
        d=dict(class_id=2,score=.9,position_world_m=[0,0,1],identity_embedding=[1.,0])
        tr=CooperativeTracker();a=tr.update({'air':[d]},0)['air'][0]
        b=tr.update({'ground':[dict(d,position_world_m=[.1,0,1])]},.2)['ground'][0]
        self.assertEqual(a['track_id'],b['track_id'])
        other=tr.update({'air':[dict(d,class_id=1)]},.4)['air'][0]
        self.assertNotEqual(other['track_id'],b['track_id'])
    def test_camera_detections_are_one_to_one(self):
        d=dict(class_id=2,score=.9,position_world_m=[0,0,1],identity_embedding=[1.,0])
        tr=CooperativeTracker();out=tr.update({'air':[d,d],'ground':[d,d]},0)
        self.assertEqual(len({r['track_id'] for r in out['air']}),2)
        self.assertEqual(len({r['track_id'] for r in out['ground']}),2)
    def test_targets_use_visible_pixels_only(self):
        rgb=np.zeros((1,2,12,16,3),np.uint8);inst=np.zeros((1,2,12,16),int);inst[0,0,3:5,4:7]=1
        s=np.array([0,0,3,0,0,0,0,0,1,1,1],np.float32)[None,None]
        target=make_targets(dict(rgb=rgb,K=np.eye(3),camera_to_world=np.tile(np.eye(4),(1,2,1,1))),
            dict(instance=inst,state=s,class_id=np.array([2]),track_id=np.array([1])),'e')
        self.assertEqual(len(target[0]['boxes']),1);self.assertEqual(len(target[1]['boxes']),0)
        self.assertEqual(target[0]['identity_keys'],['e:1'])
    def test_epoch_sampler_restarts_exactly(self):
        full=[index_at(7,17,i) for i in range(30)]
        self.assertEqual(full[:14]+[index_at(7,17,i) for i in range(14,30)],full)
        self.assertEqual(set(full[:7]),set(range(7)))
    def test_mixed_batch_has_exact_source_fraction(self):
        from .train import batch_indices
        class Rows:
            rows=[dict(source='rendered') for _ in range(10)]+[dict(source='generated') for _ in range(10)]
            def __len__(self):return len(self.rows)
        d=Rows()
        for step in range(5):
            selected=batch_indices(d,17,step,16)
            self.assertEqual(sum(d.rows[i]['source']=='generated' for i in selected),8)
            self.assertEqual(len(selected[0::2]),8)
    def test_rgb_retention_never_uses_another_future(self):
        import copy
        from .rgb_acceptance import fit,accept
        rows=[dict(role='val',correctly_associated=True,annotation_confidence=.8,identity_similarity=.9,
                   paired_center_error_m=.1,contact_reprojection_error_px=2.,jointly_visible=True)]
        threshold=fit(rows,'split','reader')
        proposal=dict(uid='x',role='train',annotation_protocol='independent_rgb_trajectory_free_v1',split_sha256='split',
            annotation_reader_sha256='reader',observations=[dict(rows[0])],unmatched=[])
        first=accept([proposal],threshold)
        altered=copy.deepcopy(proposal);altered['internal_state']=[[9999]*11];altered['predicted_visible_layers']=[0]
        second=accept([altered],threshold)
        self.assertEqual(first['report'],second['report']);self.assertEqual(first['report']['accepted_count'],1)
    def test_rgb_empty_scene_is_not_silently_accepted(self):
        from .rgb_acceptance import fit,accept
        threshold=fit([dict(role='val',correctly_associated=True,annotation_confidence=.8,identity_similarity=.9,
                   paired_center_error_m=.1,contact_reprojection_error_px=2.,jointly_visible=True)],'s','r')
        row=dict(uid='x',role='train',annotation_protocol='independent_rgb_trajectory_free_v1',split_sha256='s',annotation_reader_sha256='r',observations=[],unmatched=[])
        self.assertEqual(accept([row],threshold)['report']['accepted_count'],0)
    def test_dag_dependencies_and_budget_completion(self):
        from .matrix import build_plan
        plan=build_plan('index','assets','output',[dict(method='m',seed=17,modality='rgb',manifest='manifest',completion='done')])
        ids={j['id'] for j in plan['jobs']}
        self.assertEqual(len(ids),len(plan['jobs']))
        for j in plan['jobs']:
            self.assertTrue(set(j['dependencies'])<=ids)
            self.assertIn('predicate',j['completion'])
            if 'checkpoint' in j:self.assertEqual(j['completion']['predicate'],{'completed':True})

    def test_roi_uses_seven_sample_centers(self):
        grid=roi_grid(torch.tensor([[[.5,.5,1.,1.]]]))
        self.assertEqual(grid.shape,(1,1,7,7,2));self.assertAlmostEqual(float(grid[0,0,3,3,0]),0.)

if __name__=='__main__':unittest.main()
