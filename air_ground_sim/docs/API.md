# Simulation API and coordinate contract

`Simulator("factory_industrial")` verifies the precompiled model and native
runtime hashes before loading. `reset()` clears physical and controller state.
`command(air, ground, duration, fold_deg)` accepts UAV world `[x,y,z,yaw]`, ground
world `[x,y,yaw]`, seconds, and the mechanical arm angle in degrees.

`step(n)` advances both robots by `n` native physics ticks. The model timestep is
0.0005 s, with an integer-tick flight control clock at 0.004 s. A custom
`actuator_callback(model, data, sim)` can replace both low-level controllers for
an episode; it must supply rotor forces and all ground/folding actuator commands.
The reference controller and custom actuator callback cannot be mixed without a
reset.

`state()` returns copies of pose, velocity, actuator and attachment data.
`observe()` returns synchronous RGB, metric optical-Z depth, calibration and
geometric LiDAR. World Z is up; quaternions use `wxyz`. CV camera axes are X right,
Y down and Z forward. `camera_rotation` maps CV vectors to the world frame.

- RGB: H×W×3, uint8.
- `depth_mm`: H×W, float32, millimetres; invalid values are accompanied by `valid`.
- `depth_m`: compatibility field only; new export uses `depth_mm`.
- Intrinsics: 3×3 `K`; camera/world translation and LiDAR points: metres.
- Backprojection: `p_world = R @ ((depth_mm/1000) * inv(K) @ [u,v,1]) + t`.
- Contact wrench: force followed by torque in the contact frame.

Depth is rendered from MuJoCo geometry, never estimated from RGB. Sensors have
no default noise, glass multipath or real-device intensity calibration.
`render()` is a presentation camera; roof hiding affects visualization only.

`try_attach()` requires mating contact, pose/rate tolerances and dwell. `release()`
requires ground support. A finite load/torque threshold releases the attachment;
this is not an electromagnetic field model. The robots use gravity, joint,
contact and rotor dynamics. Prescribed worker/forklift actors are kinematic and
do not simulate a freely moving body after impact.

Physics-state NPZ files contain MuJoCo integration state and model identity.
They support same-model replay; they do not serialize Python observers, reference
segments, attachment dwell history or controller clocks.
