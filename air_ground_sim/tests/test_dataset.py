"""Projection and segmentation correctness on a small independent native scene."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import air_ground_sim
import unittest
import json
import numpy as np
import mujoco
from air_ground_sim.dataset.geometry import camera_rotation,quaternion,rotation,box_corners,projected_edges,yaw_rotation
from air_ground_sim.dataset.rig import FiveCameraRig


class DatasetGeometryTests(unittest.TestCase):
    def test_all_authored_fixture_groups_have_semantic_class(self):
        from air_ground_sim.dataset.catalog import body_class
        from air_ground_sim.paths import CONFIG
        layout=json.loads((CONFIG/'factory_industrial_layout.json').read_text())
        for item in layout['modules']:
            cls,instance=body_class(item['name'])
            self.assertGreater(cls,0,item['name'])
            if item['kind']=='pallet_staging':
                self.assertEqual(cls,12)
                self.assertTrue(instance)

    def test_axes_and_nuscenes_width_length_order(self):
        r=camera_rotation(0,0)
        np.testing.assert_allclose(r[:,2],[1,0,0],atol=1e-12)
        np.testing.assert_allclose(r[:,0],[0,-1,0],atol=1e-12)
        np.testing.assert_allclose(r[:,1],[0,0,-1],atol=1e-12)
        np.testing.assert_allclose(rotation(quaternion(r)),r,atol=1e-12)
        corners=box_corners([10,20,30],[2,6,4],[1,0,0,0])
        np.testing.assert_allclose(np.ptp(corners,axis=0),[6,2,4])

    def test_native_projection_and_occlusion(self):
        # FRONT looks down 45 degrees from the origin: sphere [2,-.5,-2].
        xml='''<mujoco><visual><global offwidth="320" offheight="180"/><map znear=".001" zfar="20"/></visual>
        <worldbody><geom name="target" type="sphere" pos="2 -.5 -2" size=".08" rgba="1 0 0 1"/>
        <geom name="hidden" type="sphere" pos="4 -1 -4" size=".08" rgba="0 1 0 1"/></worldbody></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        rig=FiveCameraRig(m,320,180,90,shadow_size=128)
        try:
            rig.update(d,np.zeros(3));rgb,seg,p,r=rig.view(0,np.zeros(3),np.eye(3))
            yy,xx=np.nonzero(seg[:,:,0]==m.geom('target').id)
            self.assertGreater(len(xx),30)
            point=np.array([2,-.5,-2])@r;uv=rig.K@point;uv=uv[:2]/uv[2]
            np.testing.assert_allclose([xx.mean(),yy.mean()],uv,atol=.6)
            self.assertFalse(np.any(seg[:,:,0]==m.geom('hidden').id))
            self.assertEqual(int(np.argmax(rgb[round(uv[1]),round(uv[0])])),0)
            rig.renderer.enable_segmentation_rendering()
            expected=rig.renderer.render().copy()
            rig.renderer.disable_segmentation_rendering()
            np.testing.assert_array_equal(seg,expected)
        finally:rig.close()

    def test_near_plane_edge_clipping(self):
        corners=box_corners([0,0,.1],[1,1,1],[1,0,0,0])
        segments=list(projected_edges(corners,np.zeros(3),np.eye(3),np.eye(3),near=.05))
        self.assertTrue(segments)
        self.assertTrue(all(np.isfinite(s).all() for s in segments))

    def test_all_five_views_with_translated_rotated_ego(self):
        position=np.array([3.,2.,8.]);ego=yaw_rotation(.9)
        angles=[(0,-45),(180,-45),(90,-45),(-90,-45),(0,-90)]
        target=np.array([.33,-.21,3.])
        geoms=[]
        for i,(yaw,pitch) in enumerate(angles):
            r=ego@camera_rotation(np.deg2rad(yaw),np.deg2rad(pitch))
            p=position+r@target
            geoms.append(f'<geom name="t{i}" type="sphere" pos="{p[0]} {p[1]} {p[2]}" size=".06"/>')
        m=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="320" offheight="180"/></visual><worldbody>'+''.join(geoms)+'</worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d);rig=FiveCameraRig(m,320,180,shadow_size=128)
        try:
            rig.update(d,position)
            for i in range(5):
                _,seg,_,_=rig.view(i,position,ego)
                yy,xx=np.nonzero(seg[:,:,0]==m.geom(f't{i}').id)
                self.assertGreater(len(xx),10)
                uv=rig.K@target
                np.testing.assert_allclose([xx.mean(),yy.mean()],uv[:2]/uv[2],atol=.6)
        finally:rig.close()

if __name__=='__main__':unittest.main()
