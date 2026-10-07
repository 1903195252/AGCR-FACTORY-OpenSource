import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from experiments.base48 import sample
from experiments.paired_interventions.data import file_sha256
from experiments.paired_interventions.train import atomic_json


class SampleTests(unittest.TestCase):
    def test_real_base_eval_lifecycle_after_releasing_training_reader(self):
        from experiments.base48.model import Base48System
        # Keep the actual train/eval override; only replace expensive assets.
        model=Base48System.__new__(Base48System);torch.nn.Module.__init__(model)
        model.base_phase='video'
        model.upstream=torch.nn.Module()
        model.upstream.reader=torch.nn.Linear(2,2)
        model.upstream.motion=torch.nn.Linear(2,2)
        model.condition=torch.nn.Linear(2,2)
        model.wan=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5))
        model.reader=torch.nn.Linear(2,2)
        released=weakref.ref(model.reader)
        renderer=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()})
        original={k:v.clone() for k,v in renderer['model'].items()}
        state=dict(model={k:torch.full_like(v,7) for k,v in original.items() if k.startswith('upstream.')})
        rng=torch.get_rng_state().clone()
        sample.install_checkpoints(model,renderer,state)
        self.assertIsNone(released())
        self.assertEqual(list(model.reader.parameters()),[])
        self.assertFalse(any(k.startswith('reader.') for k in model.state_dict()))
        with self.assertRaisesRegex(RuntimeError,'training reader'):
            model.reader(torch.zeros(1,2))
        # eval() is a public lifecycle call; it must remain safe after install.
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.upstream.training)
        self.assertFalse(model.reader.training)
        model.eval().requires_grad_(False)
        model.eval()
        self.assertFalse(any(module.training for module in model.modules()))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        for key,value in model.state_dict().items():
            expected=state['model'][key] if key.startswith('upstream.') else original[key]
            self.assertTrue(torch.equal(value,expected),key)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue(all(torch.equal(renderer['model'][k],v) for k,v in original.items()))

    def test_nonfinite_rgb_is_rejected_before_uint8_cast_and_decoder_restored(self):
        class Decoder(torch.nn.Module):
            def decode(self, *args):
                return dict(rgb=torch.tensor(float('nan')), depth_mm=torch.tensor(1.),
                            validity_logits=torch.tensor(0.))
        model=Decoder()
        with self.assertRaises(FloatingPointError):sample.checked_decode(model,None,None)
        self.assertNotIn('decode',model.__dict__)
        with self.assertRaises(FloatingPointError):
            with sample.finite_decoder(model):model.decode(None,None)
        self.assertNotIn('decode',model.__dict__)

    def protocol(self):
        return dict(schema='agcr-base48-protocol-1',paired_sample_seeds=sample.SAMPLE_SEEDS,
                    sampling=dict(augmentation_family_ids=['train']))

    def arrays(self):
        return dict(rgb=np.zeros((16,2,2,3,3),np.uint8),depth_mm=np.ones((16,2,2,3),np.float32),
            validity_probability=np.ones((16,2,2,3),np.float32),internal_state=np.zeros((16,3,11),np.float32),
            state_present=np.ones((16,3),bool),historical_reference_ids=np.array([0,-1,-1]),
            historical_reference_ids_raw=np.array([0,0,-1]),historical_duplicate_slot_mask=np.array([False,True,False]),
            historical_unassociated_slot_mask=np.array([False,False,True]),
            predicted_visible_layers=np.zeros((16,2,4,2,3)),predicted_full_layers=np.zeros((16,2,4,2,3)),
            predicted_surface_depth_mm=np.zeros((16,2,4,2,3)),initial_state=np.zeros((3,11)),class_id=np.ones(3),
            time_s=np.arange(1,17)/5,K=np.eye(3),camera_to_world=np.tile(np.eye(4),(16,2,1,1)))

    def test_registered_test_population_only_no_arbitrary_limit(self):
        index=dict(families=[dict(family_id='train',split='train'),dict(family_id='test',split='test')])
        self.assertEqual(sample.selected_families(index,self.protocol(),'test'),['test'])
        with self.assertRaises(ValueError):sample.selected_families(index,self.protocol(),'augmentation')
        p=self.protocol();p['sampling']['augmentation_family_ids']=['test']
        with self.assertRaises(ValueError):sample.selected_families(index,p,'augmentation')
        p=self.protocol();p['paired_sample_seeds']=sample.SAMPLE_SEEDS[:1]
        with self.assertRaises(ValueError):sample.selected_families(index,p,'test')

    def test_future_supervision_is_not_transferred_to_sampling(self):
        past={key:torch.zeros(1) for key in sample.HISTORY_LABELS}
        family=dict(metadata={'family_id':'x'},branches={cell:dict(inputs={'rgb':torch.zeros(6,2)},
                supervision=dict(past,state=object(),rgb=object(),depth_mm=object())) for cell in sample.CELLS})
        clean=sample.inference_family(family)
        for branch in clean['branches'].values():self.assertEqual(set(branch['supervision']),set(sample.HISTORY_LABELS))

    def test_unique_ids_without_deleting_duplicate_or_unassociated_slots(self):
        arrays=self.arrays()
        with patch.object(sample,'RGB_SHAPE',(16,2,2,3,3)),patch.object(sample,'DEPTH_SHAPE',(16,2,2,3)):
            sample.validate_arrays(arrays)
            self.assertEqual(arrays['internal_state'].shape[1],3)
            arrays['historical_reference_ids']=np.array([0,0,-1])
            with self.assertRaises(ValueError):sample.validate_arrays(arrays)

    def test_all_four_cells_required_for_atomic_draw_marker(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sample,'RGB_SHAPE',(16,2,2,3,3)),patch.object(sample,'DEPTH_SHAPE',(16,2,2,3)):
            outputs={cell:self.arrays() for cell in sample.CELLS}
            sample.publish_draw(tmp,0,101,outputs,{'family_id':'x'},{})
            sample.validate_draw(tmp,0,101)
            path=Path(tmp)/sample.file_name(1,1,0);path.write_bytes(b'changed')
            with self.assertRaises(ValueError):sample.validate_draw(tmp,0,101)
            with self.assertRaises(ValueError):sample.publish_draw(tmp,1,211,{(0,0):self.arrays()},{},{})
            self.assertFalse((Path(tmp)/'draw_01.json').exists())

    def test_cpu_merge_rejects_partial_shard_and_checks_all_eight_draws(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sample,'RGB_SHAPE',(16,2,2,3,3)),patch.object(sample,'DEPTH_SHAPE',(16,2,2,3)):
            root=Path(tmp);protocol=root/'protocol.json';protocol.write_text(json.dumps(self.protocol()))
            index=root/'index.json';index.write_text(json.dumps(dict(families=[dict(family_id='x',split='test')])))
            manifest=dict(index_sha256=file_sha256(index),protocol_sha256=file_sha256(protocol),
                sample_seeds=sample.SAMPLE_SEEDS,selected_family_ids=['x'],corpus_role='test',num_shards=1,
                source_sha256=sample.sampler_source_hashes())
            digest=sample.register_manifest(root,manifest)
            shard=root/'shards/shard_0000';shard.mkdir(parents=True)
            atomic_json(shard/'completion.json',dict(completed=True,whole_corpus_completed=False,manifest_sha256=digest,family_ids=['x']))
            self.assertFalse((root/'completion.json').exists())
            for draw,seed in enumerate(sample.SAMPLE_SEEDS[:7]):
                sample.publish_draw(root/'x',draw,seed,{cell:self.arrays() for cell in sample.CELLS},{},{})
            with self.assertRaises(ValueError):sample.merge(root,protocol,index)
            self.assertFalse(json.loads((root/'completion.json').read_text())['completed'])
            sample.publish_draw(root/'x',7,809,{cell:self.arrays() for cell in sample.CELLS},{},{})
            report=sample.merge(root,protocol,index)
            self.assertTrue(report['completed']);self.assertEqual(report['generated_cells'],32)
            marker=root/'x/draw_00.json';row=json.loads(marker.read_text());row['files'].pop(sample.file_name(0,0,0));marker.write_text(json.dumps(row))
            with self.assertRaises(ValueError):sample.merge(root,protocol,index)
            self.assertFalse(json.loads((root/'completion.json').read_text())['completed'])

    def test_renderer_and_state_checkpoint_composition_changes_only_upstream(self):
        model=torch.nn.Module();model.upstream=torch.nn.Linear(2,2);model.condition=torch.nn.Linear(2,2);model.reader=torch.nn.Linear(2,2)
        renderer=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()})
        state=dict(model={k:torch.full_like(v,7) for k,v in renderer['model'].items() if k.startswith('upstream.')})
        original={k:v.clone() for k,v in renderer['model'].items()}
        sample.install_checkpoints(model,renderer,state)
        self.assertTrue(torch.equal(model.upstream.weight,state['model']['upstream.weight']))
        self.assertTrue(torch.equal(model.condition.weight,original['condition.weight']))
        self.assertIsInstance(model.reader,sample._ReleasedTrainingReader)
        self.assertEqual(list(model.reader.parameters()),[])
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(torch.equal(renderer['model'][k],v) for k,v in original.items()))

    def test_real_fifty_step_path_reuses_world_state_without_auxiliary_reader(self):
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            class Encoders:
                def history(self,rgb):return None,torch.zeros(6,2,48,30,46)
            class FakeModel:
                def __init__(self):self.encoders=Encoders();self.upstream=object();self.flow_calls=0
                def wan(self,z,t,c):self.flow_calls+=1;return torch.zeros_like(z)
                def video_readout(self,*_):raise AssertionError('Auxiliary training reader must not run')
                def decode(self,z,c):
                    return dict(rgb=torch.zeros(16,2,3,2,3),depth_mm=torch.ones(16,2,2,3),validity_logits=torch.zeros(16,2,2,3))
            def branch(world,camera):
                x=dict(rgb=torch.full((6,2,2,3,3),world),depth_mm=torch.ones(6,2,2,3),depth_valid=torch.ones(6,2,2,3,dtype=torch.bool),
                    K=torch.eye(3),camera_to_world=torch.eye(4).repeat(6,2,1,1),time_s=torch.arange(-5,1)/5,
                    future_time_s=torch.arange(1,17)/5,planned_camera_to_world=torch.eye(4).repeat(16,2,1,1))
                x['planned_camera_to_world'][:,:,:3,3]+=camera
                y=dict(history_state=torch.zeros(6,2,11),class_id=torch.ones(2,dtype=torch.long),
                    history_visible_fraction=torch.ones(6,2,2),history_projection_area=torch.full((6,2,2),200))
                return dict(inputs=x,supervision=y)
            family=dict(metadata={'family_id':'f'},branches={cell:branch(*cell) for cell in sample.CELLS})
            tracks=[dict(birth=(i,0,0),score=torch.tensor(1.-i*.1),presence_logit=torch.tensor(1.),class_prob=torch.tensor([0.,1.])) for i in range(3)]
            predicts=[];conditions=[]
            def predict(upstream,x,encoded,generator):
                history=dict(result=dict(tracks=tracks,initial_state=torch.zeros(3,11)))
                draw=dict(state=torch.full((1,16,3,11),float(x['rgb'].float().mean())),present=torch.ones(1,16,3,dtype=torch.bool))
                predicts.append(draw);return history,draw
            def condition(upstream,history,draw,x):
                conditions.append(draw)
                return dict(M=torch.zeros(16,2,4,2,3),P=torch.zeros(16,2,4,2,3),Z=torch.zeros(16,2,4,2,3))
            model=FakeModel()
            with patch.object(sample,'predict_world',side_effect=predict),patch.object(sample,'world_conditions',side_effect=condition),patch.object(sample,'query_labels',return_value=torch.tensor([0,0,-1])):
                outputs,diagnostics=sample.generate_family(model,family,101,17,'cpu')
            self.assertEqual(model.flow_calls,200)
            self.assertEqual(len(predicts),2)
            self.assertIs(conditions[0],conditions[1]);self.assertIs(conditions[2],conditions[3])
            self.assertTrue(np.array_equal(outputs[0,0]['internal_state'],outputs[0,1]['internal_state']))
            self.assertEqual(outputs[0,0]['internal_state'].shape[1],3)
            self.assertEqual(outputs[0,0]['historical_reference_ids'].tolist(),[0,-1,-1])
            self.assertEqual(outputs[0,0]['historical_reference_ids_raw'].tolist(),[0,0,-1])
            self.assertEqual(diagnostics['0']['duplicate_history_slots'],1)
            self.assertEqual(diagnostics['0']['unassociated_history_slots'],1)
            streamed={};boundaries=[]
            def publish(w,c,arrays):
                streamed[w,c]=arrays
                return (w,c)
            with patch.object(sample,'predict_world',side_effect=predict),patch.object(sample,'world_conditions',side_effect=condition),patch.object(sample,'query_labels',return_value=torch.tensor([0,0,-1])):
                handles,stream_diagnostics=sample.generate_family(model,family,101,17,'cpu',
                    publish_cell=publish,before_cell=lambda:boundaries.append(True))
            self.assertEqual(model.flow_calls,400)
            self.assertEqual(len(boundaries),4)
            self.assertEqual(handles,{cell:cell for cell in sample.CELLS})
            self.assertEqual(stream_diagnostics,diagnostics)
            for cell in sample.CELLS:
                for key,value in outputs[cell].items():
                    np.testing.assert_array_equal(value,streamed[cell][key])
        finally:torch.set_num_threads(old_threads)

    def test_history_camera_equality_prevents_false_shared_world(self):
        x={key:torch.zeros(1) for key in ('rgb','depth_mm','depth_valid','K','camera_to_world','time_s','future_time_s')}
        family=dict(branches={cell:{'inputs':{k:v.clone() for k,v in x.items()}} for cell in sample.CELLS})
        sample.history_agreement(family)
        family['branches'][1,1]['inputs']['rgb']+=1
        with self.assertRaises(ValueError):sample.history_agreement(family)


if __name__=='__main__':unittest.main()
