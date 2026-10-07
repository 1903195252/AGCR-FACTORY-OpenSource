"""MuJoCo FK/Jacobian queries; do not mutate integrated generalized coordinates."""
import mujoco
import numpy as np


def site_pose(model,data,name):
    sid=model.site(name).id
    return data.site_xpos[sid].copy(),data.site_xmat[sid].reshape(3,3).copy()


def site_jacobian(model,data,name):
    linear,angular=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,data,linear,angular,model.site(name).id)
    return linear,angular


def body_twist(model,data,name,local=False):
    """[wx,wy,wz,vx,vy,vz], using CAD body axes rather than principal inertia axes."""
    mujoco.mj_comVel(model,data)
    result=np.empty(6)
    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,model.body(name).id,result,int(local))
    return result


def contacts(model,data):
    """Solved contact forces refer to the last dynamics solve, not a future pose."""
    rows=[]
    for i,contact in enumerate(data.contact):
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
        rows.append(dict(geoms=(model.geom(contact.geom1).name,model.geom(contact.geom2).name),
                         distance_m=float(contact.dist),position_world_m=contact.pos.copy(),
                         frame=contact.frame.reshape(3,3).copy(),wrench_contact_frame=wrench))
    return rows
