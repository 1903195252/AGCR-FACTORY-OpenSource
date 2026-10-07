# Air–Ground Factory Simulator

A MuJoCo factory environment with a folding quadrotor, a wheeled quadruped,
industrial equipment, moving actors, synchronized RGB-D cameras and geometric
LiDAR. The source release contains the current `factory_industrial` profile.
Model geometry and precompiled assets are distributed separately from Git.

## Installation

The bundled engine targets **Linux x86_64 and CPython 3.11**. The native MuJoCo
version is 3.12.1, with a large-model I/O extension and origin-relative shared
libraries. Install the provided runtime rather than a different MuJoCo wheel.

From the repository root, extract the simulation-assets release archive; its
`air_ground_sim/assets/` and `air_ground_sim/authoring/` directories merge with
this directory. Then run:

```bash
python3.11 -m venv .venv-sim
.venv-sim/bin/python -m pip install -e ./air_ground_sim
MUJOCO_GL=osmesa .venv-sim/bin/python air_ground_sim/run.py demo \
  --profile factory_industrial --seconds 2 --output outputs/factory_demo
```

Use `MUJOCO_GL=egl` on an NVIDIA server. OSMesa needs the system OSMesa library;
interactive viewing needs GLFW and a display. The full binary model is about
3.26 GB and requires several GiB of additional memory. Use a machine with at
least 16 GiB available memory; recompiling the full MJCF requires substantially
more. Keep generated output outside the source and use a new output directory.

## API

```python
from air_ground_sim import Simulator

with Simulator("factory_industrial") as sim:
    observation = sim.observe(width=160, height=120)
    sim.command(air=[-180, -82, 1.5, 0], ground=[-179.5, -81.5, 0], duration=4)
    for _ in range(4000):
        state = sim.step()
```

Positions and LiDAR use metres; yaw uses radians. RGB-D observations expose
optical-axis depth in **millimetres**, together with validity and camera
calibration. The simulator integrates both robots through their actuators.
`command()` supplies a continuous reference segment, not collision-free planning.

See [the API contract](docs/API.md), [capture protocol](docs/CAPTURE.md), and
[native runtime/build notes](runtime/BUILD.md).

## Source layout

| Directory | Contents |
| --- | --- |
| `src/air_ground_sim` | Physics, controllers, folding kinematics, attachment, sensors and exporters |
| `configs` | Robot dimensions, profile registry, industrial layout and capture specification |
| `tools/paper_dataset` | Scripted visibility events, scene construction, paired RGB-D capture and labels |
| `tests` | Deterministic controller, geometry and serialization checks |
| `runtime` | Native engine, Python bindings, licenses, engine patch and upstream build source |
| `assets` | Release asset payload: MJCF, deduplicated meshes/textures and precompiled industrial model |
| `authoring` | Release asset payload: editable Blender robot, fixture and industrial-scene sources |

## Checks and export

```bash
.venv-sim/bin/python air_ground_sim/tests/test_core.py
MUJOCO_GL=osmesa .venv-sim/bin/python air_ground_sim/tests/test_dataset.py
.venv-sim/bin/python air_ground_sim/tools/test_paper_dataset.py
.venv-sim/bin/python air_ground_sim/run.py verify --profile factory_industrial
.venv-sim/bin/python air_ground_sim/run.py export-scene \
  --profile factory_industrial --component all --output outputs/editable_scene
```

`export-scene` writes the actual MJCF and its referenced assets. Blender sources
are editable geometry references; an appearance edit does not automatically
update inertia, collision geometry or a precompiled model.

The default ground controller uses wheels and suspension on a flat floor;
moving worker/vehicle actors follow prescribed kinematic routes. Camera depth
and LiDAR are ideal geometry measurements. Attachment uses contact-gated finite
forces and torques. These modeling choices are documented in [the API](docs/API.md).

## Licenses

MuJoCo's Apache-2.0 license, GCC runtime notices and the GCC Runtime Library
Exception remain in `runtime/`. Upstream copyright notices are retained in the
vendored source. Project-authored software and configuration use Apache License 2.0;
non-software assets retain their existing terms. See the repository license and notices.
