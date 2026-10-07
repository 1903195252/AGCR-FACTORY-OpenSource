from __future__ import annotations
import math
import numpy as np
import mujoco
class PolylineRoute:
    def __init__(self, points) -> None:
        self.points = tuple(points)
        self.segments = []
        self.length = 0.0
        for start, end in zip(self.points, self.points[1:] + self.points[:1]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            self.segments.append((start, end, segment_length))
            self.length += segment_length

    def sample(self, distance: float) -> tuple[float, float, float]:
        distance %= self.length
        for start, end, segment_length in self.segments:
            if distance <= segment_length:
                ratio = distance / max(segment_length, 1e-9)
                x = start[0] + ratio * (end[0] - start[0])
                y = start[1] + ratio * (end[1] - start[1])
                return x, y, math.atan2(end[1] - start[1], end[0] - start[0])
            distance -= segment_length
        start, end, _ = self.segments[-1]
        return end[0], end[1], math.atan2(end[1] - start[1], end[0] - start[0])

class DynamicActorController:
    """Moves mocap workers and forklifts along segregated factory traffic routes."""

    def __init__(self, model: mujoco.MjModel, route_variant='baseline') -> None:
        west=-188 if route_variant=='industrial' else -185
        logistics = PolylineRoute(((west, -138), (185, -138), (185, -84), (west, -84)))
        cross_south = PolylineRoute(((-185, -86), (185, -86), (185, -78), (-185, -78)))
        cross_center = PolylineRoute(((-185, -4), (185, -4), (185, 4), (-185, 4)))
        central_left = PolylineRoute(((-4.2, -138), (-4.2, 88), (-1.2, 88), (-1.2, -138)))
        central_right = PolylineRoute(((1.2, -138), (1.2, 88), (4.2, 88), (4.2, -138)))
        utility_south=86.3 if route_variant=='industrial' else 87
        utility = PolylineRoute(((-185, utility_south), (185, utility_south), (185, 93), (-185, 93)))
        south,north=(-76.3,84.5) if route_variant=='industrial' else (-75,78)
        worker_west = PolylineRoute(((-182, south), (-18, south), (-18, north), (-182, north)))
        east_inner=11 if route_variant=='industrial' else 18
        worker_east = PolylineRoute(((east_inner, south), (182, south), (182, north), (east_inner, north)))
        walk_x=6.8 if route_variant=='industrial' else 8
        worker_center = PolylineRoute(((-walk_x, -135), (-walk_x, 85), (walk_x, 85), (walk_x, -135)))
        warehouse_south=-142 if route_variant=='industrial' else -140
        worker_warehouse = PolylineRoute(((-180, warehouse_south), (-20, warehouse_south), (-20, -88), (-180, -88)))
        utility_x,utility_y=(182.5,90) if route_variant=='industrial' else (180,88)
        worker_utility = PolylineRoute(((-utility_x, utility_y), (utility_x, utility_y), (utility_x, 138), (-utility_x, 138)))

        forklift_specs = (
            ("mobile_robot_1", logistics, 4.2, 0.02),
            ("forklift_dynamic_00", logistics, 3.6, 0.18),
            ("forklift_dynamic_01", cross_south, 4.8, 0.35),
            ("forklift_dynamic_02", cross_south, 4.0, 0.66),
            ("forklift_dynamic_03", cross_center, 4.5, 0.12),
            ("forklift_dynamic_04", central_left, 3.4, 0.32),
            ("forklift_dynamic_05", central_right, 3.8, 0.58),
            ("forklift_dynamic_06", utility, 3.2, 0.22),
            ("forklift_dynamic_07", utility, 3.7, 0.72),
        )
        worker_routes = (worker_west, worker_east, worker_center, worker_warehouse, worker_utility)
        worker_names = ["pedestrian_1"] + [f"worker_dynamic_{index:02d}" for index in range(12)]
        worker_specs = tuple(
            (name, worker_routes[index % len(worker_routes)], 1.05 + 0.11 * (index % 5),
             (0.071 + index * 0.137) % 1.0)
            for index, name in enumerate(worker_names)
        )
        self.actors = []
        for kind, specs, height in (("forklift", forklift_specs, 0.0),
                                    ("worker", worker_specs, 0.0)):
            for name, route, speed, phase in specs:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    continue
                mocap_id = int(model.body_mocapid[body_id])
                if mocap_id >= 0:
                    self.actors.append((kind, mocap_id, route, speed, phase, height))

    @staticmethod
    def quaternion(yaw: float) -> tuple[float, float, float, float]:
        return math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)

    def update(self, data: mujoco.MjData, scene_time: float) -> None:
        for kind, mocap_id, route, speed, phase, height in self.actors:
            x, y, yaw = route.sample(phase * route.length + scene_time * speed)
            bob = 0.025 * abs(math.sin(scene_time * speed * 2.8 + phase * math.tau)) \
                if kind == "worker" else 0.0
            data.mocap_pos[mocap_id] = (x, y, height + bob)
            data.mocap_quat[mocap_id] = self.quaternion(yaw)

class SmoothActors:
    """Prescribed mocap obstacles, with bounded translation and yaw per step.

    Pose initialization happens before the first physics step. Rotation at
    polyline corners is rate-limited; no 20 Hz jumps of collision geometry.
    These actors have infinite effective mass, with no momentum response.
    """
    def __init__(self,model,data,route_variant='baseline'):
        self.model=model
        self.driver=DynamicActorController(model,route_variant)
        self.scratch=mujoco.MjData(model)
        self.driver.update(data,0.)
        self.last_time=0.
        self.max_displacement=0.
        self.max_yaw_step=0.

    def update(self,data):
        dt=float(self.model.opt.timestep)
        self.driver.update(self.scratch,float(data.time)+dt)
        for kind,i,route,speed,phase,height in self.driver.actors:
            delta=self.scratch.mocap_pos[i]-data.mocap_pos[i]
            distance=np.linalg.norm(delta)
            step=delta*min(1.,speed*1.1*dt/max(distance,1e-12))
            data.mocap_pos[i]+=step
            yaw=2*math.atan2(data.mocap_quat[i,3],data.mocap_quat[i,0])
            target=2*math.atan2(self.scratch.mocap_quat[i,3],self.scratch.mocap_quat[i,0])
            error=math.atan2(math.sin(target-yaw),math.cos(target-yaw))
            rate=1.2 if kind=='forklift' else 2.
            turn=float(np.clip(error,-rate*dt,rate*dt))
            data.mocap_quat[i]=self.driver.quaternion(yaw+turn)
            self.max_displacement=max(self.max_displacement,float(np.linalg.norm(step)))
            self.max_yaw_step=max(self.max_yaw_step,abs(turn))
