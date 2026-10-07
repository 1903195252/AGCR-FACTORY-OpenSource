"""Right-handed camera/ego transforms and nuScenes box conventions."""
import itertools
import numpy as np
import mujoco

SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))
EDGES = [(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(SIGNS[i]!=SIGNS[j])==1]


def quaternion(rotation):
    q=np.empty(4)
    mujoco.mju_mat2Quat(q,np.asarray(rotation,dtype=float).ravel())
    return (q if q[0]>=0 else -q).tolist()


def rotation(q):
    r=np.empty(9)
    mujoco.mju_quat2Mat(r,np.asarray(q,dtype=float))
    return r.reshape(3,3)


def yaw_rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def camera_rotation(yaw,pitch):
    """Camera CV XYZ (right, down, forward) -> ego FLU XYZ."""
    forward=np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
    right=np.array([np.sin(yaw),-np.cos(yaw),0.])
    down=np.cross(forward,right)
    return np.column_stack((right,down,forward))


def intrinsics(width,height,hfov):
    f=.5*width/np.tan(np.deg2rad(hfov)/2)
    return np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])


def box_corners(translation,size_wlh,quaternion_wxyz):
    return SIGNS*np.asarray(size_wlh)[[1,0,2]]/2 @ rotation(quaternion_wxyz).T+translation


def projected_edges(corners,position,cam_to_world,K,near=.05):
    """Clip each edge against the near plane before perspective division."""
    camera=(corners-position)@cam_to_world
    for i,j in EDGES:
        a,b=camera[i].copy(),camera[j].copy()
        if max(a[2],b[2])<near:continue
        if a[2]<near:a+=(b-a)*(near-a[2])/(b[2]-a[2])
        if b[2]<near:b+=(a-b)*(near-b[2])/(a[2]-b[2])
        uv=np.stack((a,b))@K.T
        yield uv[:,:2]/uv[:,2,None]


def geom_local_corners(model,gid):
    """Conservative bounds of a visual geom; mesh bounds use actual vertices."""
    typ=int(model.geom_type[gid]);s=model.geom_size[gid]
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(model.geom_dataid[gid]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
        v=model.mesh_vert[a:a+n];lo=v.min(0);hi=v.max(0)
        return (lo+hi)/2+SIGNS*(hi-lo)/2
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):extent=np.repeat(s[0],3)
    elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):extent=np.array([s[0],s[0],s[0]+s[1]])
    elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):extent=np.array([s[0],s[0],s[1]])
    elif typ in (int(mujoco.mjtGeom.mjGEOM_BOX),int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):extent=s
    else:raise ValueError('Unbounded or unsupported instance geom: '+model.geom(gid).name)
    return SIGNS*extent
