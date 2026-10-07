from __future__ import annotations
import math
import numpy as np
import mujoco
_RAY_NORMAL_ARGUMENT = {'normal': None} if 'normal:' in (mujoco.mj_multiRay.__doc__ or '') else {}

def visual_option():
    opt=mujoco.MjvOption()
    opt.geomgroup[:]=[1,1,1,0,1,0]
    return opt

class SensorBridge:
    def __init__(self,model,width=320,height=240,range_m=60.,appearance=None):
        if not any(model.geom(i).name.startswith(('visual_batch_', 'owned_visual_')) for i in range(model.ngeom)):
            raise ValueError('RGB-D/LiDAR observations require the complete authored visual meshes; contact-only models are ineligible.')
        self.model=model;self.width=width;self.height=height;self.range_m=range_m
        self.appearance=appearance
        self.renderer=mujoco.Renderer(model,width=width,height=height,max_geom=int(model.ngeom)+2048)
        self.option=visual_option()
        self.option.geomgroup[4]=1
        # Group 0 floor, 1 robots, 2 fine meshes. No invisible coarse contacts.
        self.ray_groups=np.array([1,1,1,0,1,0],np.uint8)

    def rgbd(self,data,robot):
        camera=self.model.camera(robot+'_rgbd').id
        if self.appearance:
            self.appearance.prepare(self.model,data.cam_xpos[camera],self.range_m,self.renderer)
            mujoco.mj_camlight(self.model,data)
        self.renderer.update_scene(data,camera=camera,scene_option=self.option)
        rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering();depth=self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        valid=np.isfinite(depth)&(depth>0)&(depth<=self.range_m)
        depth[~valid]=0
        fy=.5*self.height/np.tan(np.deg2rad(self.model.cam_fovy[camera])*.5)
        K=np.array([[fy,0,(self.width-1)/2],[0,fy,(self.height-1)/2],[0,0,1.]])
        return dict(rgb=rgb,depth_mm=(depth*1000).astype(np.float32),depth_unit='mm',
                    depth_m=depth.astype(np.float32),valid=valid,  # legacy compatibility field
                    K=K,timestamp_s=float(data.time),camera_position=data.cam_xpos[camera].copy(),
                    camera_rotation=data.cam_xmat[camera].reshape(3,3)@np.diag([1.,-1.,-1.]),
                    camera_axes='CV: X right, Y down, Z forward; rotation maps camera to world')

    def lidar(self,data,robot,rings=16,azimuth_samples=180):
        cid=self.model.camera(robot+'_rgbd').id
        origin=data.cam_xpos[cid].copy()
        body=int(self.model.cam_bodyid[cid])
        # LiDAR axes follow body XYZ (X forward, Z up), not the overview camera.
        rotation=data.xmat[body].reshape(3,3)
        az,el=np.meshgrid(np.linspace(-np.pi,np.pi,azimuth_samples,endpoint=False),
                          np.deg2rad(np.linspace(-15,15,rings)))
        local=np.column_stack([(np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()])
        directions=np.ascontiguousarray(local@rotation.T,dtype=np.float64)
        count=len(directions);ids=np.full(count,-1,np.int32);distance=np.full(count,-1.,np.float64)
        mujoco.mj_multiRay(self.model,data,origin,directions.ravel(),self.ray_groups,1,body,
                          ids,distance,nray=count,cutoff=self.range_m,**_RAY_NORMAL_ARGUMENT)
        valid=(distance>.05)&(distance<=self.range_m)
        own={body}
        for b in range(body+1,self.model.nbody):
            if int(self.model.body_parentid[b]) in own:own.add(b)
        self_hit=np.array([g>=0 and int(self.model.geom_bodyid[g]) in own for g in ids])
        valid &= ~self_hit
        xyz=origin+directions[valid]*distance[valid,None]
        distance[~valid]=0
        return dict(xyz_world=xyz.astype(np.float32),ranges_m=distance.astype(np.float32),valid=valid,
                    origin=origin,timestamp_s=float(data.time))

    def close(self):self.renderer.close()
