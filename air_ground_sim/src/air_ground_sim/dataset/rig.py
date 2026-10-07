"""Five ideal, synchronous pinhole cameras using the complete native scene."""
import numpy as np
import mujoco
from .geometry import camera_rotation,intrinsics,quaternion

CHANNELS=('CAM_FRONT','CAM_BACK','CAM_LEFT','CAM_RIGHT','CAM_BOTTOM')


class FiveCameraRig:
    def __init__(self,model,width=800,height=450,hfov=90.,appearance=None,shadow_size=1024):
        self.model=model;self.width=width;self.height=height;self.appearance=appearance
        self.K=intrinsics(width,height,hfov)
        self.rotations=[camera_rotation(np.deg2rad(y),np.deg2rad(p)) for y,p in [(0,-45),(180,-45),(90,-45),(-90,-45),(0,-90)]]
        # Co-located ideal sensors on a virtual survey rig, explicitly recorded.
        self.translations=[np.zeros(3) for _ in CHANNELS]
        model.vis.global_.offwidth=max(width,model.vis.global_.offwidth)
        model.vis.global_.offheight=max(height,model.vis.global_.offheight)
        model.vis.quality.shadowsize=shadow_size
        model.vis.quality.offsamples=0  # exact discrete instance IDs at edges
        self.renderer=mujoco.Renderer(model,width=width,height=height,max_geom=model.ngeom+2048)
        self.option=mujoco.MjvOption();self.option.geomgroup[:]=[1,1,1,0,1,0]
        self.option.sitegroup[:]=0
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance=1.

    def calibration(self):
        return [dict(channel=c,translation=t.tolist(),rotation=quaternion(r),camera_intrinsic=self.K.tolist())
                for c,t,r in zip(CHANNELS,self.translations,self.rotations)]

    def update(self,data,position):
        if self.appearance:
            self.appearance.prepare(self.model,position,45.,self.renderer)
        mujoco.mj_camlight(self.model,data)
        self.camera.lookat[:]=position
        self.renderer.update_scene(data,camera=self.camera,scene_option=self.option)
        # One lookup per frozen scene, shared by its five simultaneous views.
        self._seg_lookup=np.full((self.renderer.scene.ngeom+1,2),-1,np.int32)
        for g in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
            if g.segid>=0:self._seg_lookup[g.segid+1]=[g.objid,g.objtype]

    def segmentation(self):
        renderer=self.renderer;scene=renderer.scene;flags=scene.flags.copy()
        raw=np.empty((self.height,self.width,3),np.uint8)
        if renderer._gl_context:renderer._gl_context.make_current()
        try:
            scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT]=True
            scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR]=True
            mujoco.mjr_render(renderer._rect,scene,renderer._mjr_context)
            mujoco.mjr_readPixels(raw,None,renderer._rect,renderer._mjr_context)
        finally:scene.flags[:]=flags
        if renderer._gl_context:raw=raw[::-1]
        raw=raw.astype(np.uint32)
        ids=raw[:,:,0]+256*raw[:,:,1]+65536*raw[:,:,2]
        return self._seg_lookup[ids]

    def view(self,index,position,ego_rotation):
        p=position+ego_rotation@self.translations[index]
        r=ego_rotation@self.rotations[index]
        near=float(self.model.vis.map.znear*self.model.stat.extent)
        far=float(self.model.vis.map.zfar*self.model.stat.extent)
        halfheight=near*self.height/(2*self.K[1,1])
        for cam in self.renderer.scene.camera:
            cam.pos[:]=p;cam.forward[:]=r[:,2];cam.up[:]=-r[:,1]
            cam.frustum_near=near;cam.frustum_far=far
            cam.frustum_bottom=-halfheight;cam.frustum_top=halfheight
            # 0 lets Classic infer the horizontal half-width from the viewport.
            cam.frustum_center=0;cam.frustum_width=0
            cam.orthographic=0
        rgb=self.renderer.render().copy()
        seg=self.segmentation()
        return rgb,seg,p,r

    def close(self):self.renderer.close()
