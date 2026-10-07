"""Reader diagnosis preserves targets and isolates postprocessing from images."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
import torch
from .clean_reference import select_families, infer_clean,check_reader_policy,paired_source_policy,infer_generated


class FixedReader(torch.nn.Module):
    modality='rgb'
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.calls=[]
    def forward(self,**inputs):
        self.calls.append(set(inputs))
        # View zero points above its horizon; both image detections are valid.
        return dict(logits=torch.tensor([[[9.,-9.,-9.,-9.]],[[9.,-9.,-9.,-9.]]]),
            boxes=torch.tensor([[[.5,.5,.5,.5]],[[.5,.5,.5,.5]]]),
            contact=torch.tensor([[[.5,.25]],[[.5,.75]]]),
            identity=torch.tensor([[[1.,0.]],[[1.,0.]]]),
            mask_logits=torch.ones(2,1,2,2))


class CleanReferenceTest(unittest.TestCase):
    def test_metadata_panel_is_order_invariant_and_test_fixed(self):
        rows=[dict(family_id=str(k),split='val',cohort='evidence',receiver=k%2) for k in range(8)]
        self.assertEqual(select_families({'families':rows},'val'),select_families({'families':list(reversed(rows))},'val'))
        with self.assertRaises(ValueError):select_families({'families':rows},'test')
        test=[dict(r,split='test') for r in rows]
        got=select_families({'families':test},'test',paired_manifest={'split':'test','selected_family_ids':['5','2']})
        self.assertEqual([x['family_id'] for x in got],['5','2'])

    def test_new_reader_cannot_compare_against_old_readouts_as_same_implementation(self):
        from .compare import validate_shared_reader
        old=dict(checkpoint_sha256='same',association_sha256='same')
        new=dict(old,reader_postprocessing='v2',reader_source_sha256={'inference.py':'new'})
        with self.assertRaises(ValueError):check_reader_policy(old,new)
        with self.assertRaises(ValueError):validate_shared_reader(old,new)
        check_reader_policy(old,new,rerun_generated=True)
        validate_shared_reader(new,dict(new))

    def test_instrumentation_is_label_free_and_preserves_valid_2d_box(self):
        model=FixedReader();n=7;H=W=16
        pose=np.broadcast_to(np.eye(4),(n,2,4,4)).copy()
        # Camera x right, y down, z forward; camera is one metre above floor.
        pose[:,:,:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];pose[:,:,2,3]=1
        arrays=dict(rgb=np.zeros((n,2,H,W,3),np.uint8),depth_mm=np.zeros((n,2,H,W)),
            valid=np.zeros((n,2,H,W),bool),K=np.array([[16.,0,8],[0,16,8],[0,0,1]]),
            camera_to_world=pose,time_s=np.arange(n)*.2)
        meta=dict(episode='x',layout=1,scene_parent_id='x')
        ref=[dict(meta,frame=t,view=v,detections=[dict(class_id=1,track_id=77,box_xyxy=[4,4,12,12])])
             for t in range(n) for v in ('air','ground')]
        with patch('experiments.formal_evaluation.clean_reference.mask_rle',return_value={'size':[H,W],'counts':'mock'}):
            predictions,trace=infer_clean(model,{},arrays,meta,ref,.25)
        self.assertEqual(len(model.calls),n)
        self.assertTrue(all(keys=={'images','depth_mm','valid','K'} for keys in model.calls))
        self.assertEqual(trace['stages']['history:air:raw_at_fixed_association_threshold']['recall'],1)
        self.assertEqual(trace['stages']['history:air:after_geometry_filter']['recall'],1)
        self.assertEqual(trace['stages']['future:ground:after_tracking']['recall'],1)
        self.assertEqual(len(trace['geometry_dropped_queries']),0)
        self.assertTrue(all(d['position_world_m'] is None for r in predictions if r['view']=='air' for d in r['detections']))
        self.assertTrue(all(d['track_id']!=77 for r in predictions for d in r['detections']))

    def test_new_reader_rereads_pixels_but_cannot_reuse_old_reader_predictions(self):
        from .common import write_json,sha256
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pred=root/'pred';old=root/'old'
            write_json(pred/'manifest.json',dict(index_sha256='index',split='test',selected_family_ids=['f']))
            saved=dict(generator_manifest_sha256=sha256(pred/'manifest.json'),checkpoint_sha256='parent',association_sha256='old-params',
                       reader_postprocessing='v2',reader_source_sha256={'inference.py':'old'})
            write_json(old/'manifest.json',saved)
            new=dict(checkpoint_sha256='candidate',association_sha256='new-params',reader_postprocessing='v4',reader_source_sha256={'inference.py':'new'})
            with self.assertRaises(ValueError):paired_source_policy(pred,old,False,new,'index')
            with self.assertRaises(ValueError):paired_source_policy(pred,None,False,new,'index')
            with self.assertRaises(ValueError):paired_source_policy(None,old,True,new,'index')
            pair,provenance=paired_source_policy(pred,old,True,new,'index')
            self.assertFalse(provenance['reused_predictions']);self.assertEqual(provenance['checkpoint_sha256'],'parent')
            self.assertEqual(paired_source_policy(pred,None,True,new,'index'),(pair,None))
            _,reused=paired_source_policy(pred,old,False,saved,'index');self.assertTrue(reused['reused_predictions'])
            write_json(old/'manifest.json',dict(saved,generator_manifest_sha256='different-corpus'))
            with self.assertRaises(ValueError):paired_source_policy(pred,old,True,new,'index')

    def test_rerun_supplies_same_model_parameters_and_generated_pixels_without_internal_states(self):
        model=FixedReader();config={'confidence':.5};refs=[];meta={'episode':'x'}
        arrays=dict(rgb=np.ones((22,2,2,2,3),np.uint8),depth_mm=np.ones((22,2,2,2)),valid=np.ones((22,2,2,2),bool),
            K=np.eye(3),camera_to_world=np.broadcast_to(np.eye(4),(22,2,4,4)),time_s=np.arange(22)*.2)
        generated=dict(rgb=np.full((16,2,2,2,3),7,np.uint8),internal_state='never an input',historical_reference_ids='never an input')
        with patch('experiments.formal_evaluation.clean_reference.evaluation_geometry',return_value=(arrays['K'],arrays['camera_to_world'])), \
             patch('experiments.formal_evaluation.clean_reference.infer_clean',return_value=(['new predictions'],{'new trace':True})) as infer:
            for _ in range(2):self.assertEqual(infer_generated(model,config,arrays,generated,meta,refs,None)[0],['new predictions'])
        self.assertEqual(infer.call_count,2)
        for call in infer.call_args_list:
            args=call.args;self.assertIs(args[0],model);self.assertIs(args[1],config)
            observed=args[2];self.assertEqual(set(observed),{'rgb','depth_mm','valid','K','camera_to_world','time_s'})
            self.assertTrue((observed['rgb'][:6]==1).all());self.assertTrue((observed['rgb'][6:]==7).all())
            self.assertFalse(observed['valid'][6:].any())


if __name__=='__main__':unittest.main()
