from __future__ import annotations
import math
import numpy as np
import mujoco
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class AttachmentConfig:
    lateral_tolerance_m: float = .012
    normal_tolerance_m: float = .004
    angle_tolerance_rad: float = math.radians(7.)
    relative_speed_limit_mps: float = .06
    contact_dwell_s: float = .1
    force_limit_N: float = 75.
    moment_limit_Nm: float = 2.
    constraint_time_constant_s: float = .02
    torque_scale_m: float = .1

class MagneticAttachment:
    def __init__(self, model, data, config=AttachmentConfig(), surface_spec=None):
        self.model, self.data, self.config = model, data, config
        self.eq = model.equality("gate_magnetic_attachment").id
        self.dog = model.body("dog_mobile_base").id
        self.air = model.body("uav_fuav_root").id
        self.dog_site = model.site("gate_dog_attachment").id
        self.air_site = model.site("gate_air_attachment").id
        self.surface_spec = surface_spec
        dog_geom = model.geom("dog_chassis_collision").id
        if surface_spec is None:
            air_geoms = [model.geom("uav_root_collision_proxy").id]
        else:
            prefix = surface_spec['air_contact_prefix']
            air_geoms = [g for g in range(model.ngeom) if model.geom(g).name.startswith(prefix)]
            if not air_geoms or any(model.geom_bodyid[g] != self.air for g in air_geoms):
                raise ValueError('Registered mating contacts must belong to the UAV root')
        self.pairs = {frozenset((dog_geom, g)) for g in air_geoms}
        # Retain the legacy single-pair attribute for unchanged callers.
        self.pair = {dog_geom, air_geoms[0]}
        joint = int(model.body_jntadr[self.dog])
        self.dof = int(model.jnt_dofadr[joint])
        self.contact_since = None
        self.broken = False
        self.peak_force, self.peak_moment = 0., 0.
        if model.site_bodyid[self.dog_site]!=self.dog or model.site_bodyid[self.air_site]!=self.air:
            raise ValueError("Attachment frame belongs to another rigid body")
        # Check against compiled geometry, including all fixed CAD parent frames.
        surfaces = [(self.dog,self.dog_site,"dog_chassis_collision",1)]
        if surface_spec is None:
            surfaces.append((self.air,self.air_site,"uav_root_collision_proxy",-1))
        else:
            rotation = data.xmat[self.air].reshape(3, 3)
            lower_faces = []
            for gid in air_geoms:
                local_rotation = rotation.T@data.geom_xmat[gid].reshape(3, 3)
                center = rotation.T@(data.geom_xpos[gid]-data.xpos[self.air])
                center += local_rotation@model.geom_aabb[gid, :3]
                lower_faces.append(center[2]-np.abs(local_rotation[2])@model.geom_aabb[gid, 3:])
            if abs(min(lower_faces)-model.site_pos[self.air_site, 2]) > 1e-8:
                raise ValueError('Registered UAV mating frame differs from its actual bottom sheet')
        for body,site,geom,sign in surfaces:
            gid=model.geom(geom).id
            R=data.xmat[body].reshape(3,3)
            p=R.T@(data.geom_xpos[gid]-data.xpos[body])
            local_rotation=R.T@data.geom_xmat[gid].reshape(3,3)
            expected=p[2]+sign*np.abs(local_rotation[2])@model.geom_size[gid]
            if abs(model.site_pos[site,2]-expected)>1e-9:
                raise ValueError("Attachment site disagrees with compiled external surface")

    @property
    def active(self):
        return bool(self.data.eq_active[self.eq])

    def is_mating_contact(self, geom_ids):
        return frozenset(int(g) for g in geom_ids) in self.pairs

    def geometry(self):
        m, d = self.model, self.data
        Rd, Ra = d.xmat[self.dog].reshape(3,3), d.xmat[self.air].reshape(3,3)
        dog_anchor = d.site_xpos[self.dog_site]
        air_anchor = d.site_xpos[self.air_site]
        delta = Rd.T@(air_anchor-dog_anchor)
        velocities = []
        for bid, point in ((self.dog,dog_anchor),(self.air,air_anchor)):
            jacp, jacr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            mujoco.mj_jac(m,d,jacp,jacr,point,bid)
            velocities.append(jacp@d.qvel)
        touching = any(self.is_mating_contact((c.geom1,c.geom2)) and c.dist <= 0 for c in d.contact)
        return dict(lateral_error_m=float(np.linalg.norm(delta[:2])), normal_gap_m=float(delta[2]),
                    angle_error_rad=float(np.arccos(np.clip((np.trace(Rd.T@Ra)-1)/2.,-1.,1.))),
                    relative_speed_mps=float(np.linalg.norm(velocities[1]-velocities[0])),
                    plate_contact=bool(touching), dog_anchor=dog_anchor, air_anchor=air_anchor)

    def try_lock(self, now):
        if self.active or self.broken:
            return False
        g, c = self.geometry(), self.config
        eligible = (g["plate_contact"] and g["lateral_error_m"] <= c.lateral_tolerance_m
                    and abs(g["normal_gap_m"]) <= c.normal_tolerance_m
                    and g["angle_error_rad"] <= c.angle_tolerance_rad
                    and g["relative_speed_mps"] <= c.relative_speed_limit_mps)
        if not eligible:
            self.contact_since = None
            return False
        if self.contact_since is None:
            self.contact_since = now
        if now-self.contact_since < c.contact_dwell_s:
            return False
        self.data.eq_active[self.eq] = True
        return True

    def solved_wrench(self):
        m,d = self.model,self.data
        rows = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == self.eq)
        if not self.active or not rows.any():
            return dict(force_N=0.,moment_Nm=0.,overload=False)
        constraint_force = np.zeros(d.nefc)
        constraint_force[rows] = d.efc_force[rows]
        generalized = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m,d,generalized,constraint_force)
        force = generalized[self.dof:self.dof+3]
        rotation = d.xmat[self.dog].reshape(3,3)
        moment_at_root = rotation@generalized[self.dof+3:self.dof+6]
        anchor_offset = d.site_xpos[self.dog_site]-d.xpos[self.dog]
        moment_at_connection = moment_at_root-np.cross(anchor_offset,force)
        fn,mn = float(np.linalg.norm(force)),float(np.linalg.norm(moment_at_connection))
        self.peak_force,self.peak_moment = max(self.peak_force,fn),max(self.peak_moment,mn)
        overload = fn > self.config.force_limit_N or mn > self.config.moment_limit_Nm
        if overload:
            self.data.eq_active[self.eq] = False
            self.broken = True
        return dict(force_N=fn,moment_Nm=mn,overload=bool(overload))

    def release(self, ground_supported):
        if not self.active or not ground_supported:
            return False
        self.data.eq_active[self.eq] = False
        self.contact_since = None
        return True

    def specification(self):
        return dict(**asdict(self.config), model="Contact-gated soft weld with solved-wrench breakaway",
                    calibration="Declared development values; not measured magnet characteristics",
                    surface_assumption=("Registered source UAV bottom-sheet convex pieces to labelled nominal dog backplate"
                        if self.surface_spec is not None else
                        "CAD-verified external UAV bottom sheet to existing upper chassis face; buried legacy dock marker retained"),
                    collisions_disabled=False, mass_or_inertia_changed=False,
                    registered_surface_spec=self.surface_spec, mating_pair_count=len(self.pairs))
