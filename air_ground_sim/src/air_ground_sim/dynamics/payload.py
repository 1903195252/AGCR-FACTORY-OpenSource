"""Known assembly inertia for the unchanged CFNTSM/GFA control equations."""
from types import FunctionType, MethodType
import numpy as np


def bind_inertia_schedule(controller, inertia):
    """Inject nominal parameters without modifying the release or its globals.

    The release compute function reads J_fast/Jinv_fast as module globals.
    A private namespace binds those two dependencies. Its bytecode, gains,
    observer equations, saturation and allocation algorithm remain unchanged.
    """
    original=controller.compute.__func__
    if not {'J_fast','Jinv_fast'}.issubset(original.__code__.co_names):
        raise ValueError('Release controller inertia interface changed')
    namespace=dict(original.__globals__,J_fast=inertia,Jinv_fast=lambda alpha:np.linalg.inv(inertia(alpha)))
    compute=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    controller.compute=MethodType(compute,controller)
    assert controller.compute.__func__.__code__ is original.__code__


def subtree_ids(model, root):
    selected={root}
    for body in range(root+1,model.nbody):
        if int(model.body_parentid[body]) in selected:
            selected.add(body)
    return np.array(sorted(selected),dtype=int)


def rigid_assembly_inertia(model,data,bodies,frame):
    """Mass properties from known CAD and measured joint poses, in frame axes."""
    masses=model.body_mass[bodies]
    positions=data.xipos[bodies]
    center=np.average(positions,axis=0,weights=masses)
    rotations=data.ximat[bodies].reshape(-1,3,3)
    intrinsic=np.einsum('nik,nk,njk->nij',rotations,model.body_inertia[bodies],rotations).sum(axis=0)
    offset=positions-center
    parallel=(np.einsum('n,ni,ni->',masses,offset,offset)*np.eye(3)
              -np.einsum('n,ni,nj->ij',masses,offset,offset))
    rotation=data.xmat[frame].reshape(3,3)
    return rotation.T@(intrinsic+parallel)@rotation


class PayloadInertia:
    def __init__(self,model,data,air,dog,connection,legacy_inertia):
        self.model,self.data,self.air,self.dog,self.connection=model,data,air,dog,connection
        self.legacy_inertia=legacy_inertia
        self.bodies=np.concatenate((subtree_ids(model,air.root),subtree_ids(model,dog.body)))
        self.air_mass=float(model.body_subtreemass[air.root])
        self.dog_mass=float(model.body_subtreemass[dog.body])
        self.cache_key,self.cache=None,None

    def __call__(self,alpha):
        fraction=float(np.clip((self.air.controller.mass-self.air_mass)/self.dog_mass,0.,1.)) if self.connection.active else 0.
        if fraction==0.:
            return self.legacy_inertia(alpha)
        key=(float(self.data.time),fraction,float(alpha))
        if key!=self.cache_key:
            assembly=rigid_assembly_inertia(self.model,self.data,self.bodies,self.air.root)
            self.cache=(1.-fraction)*self.legacy_inertia(alpha)+fraction*assembly
            self.cache_key=key
        return self.cache.copy()
