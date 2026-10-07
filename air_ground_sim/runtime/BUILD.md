# Native runtime and engine source

The bundled engine is MuJoCo 3.12.1 at upstream commit
`707f4735e248872a38c5040a21b510a5bd386459`. Complete engine, Python-binding,
plugin, sample and test build source is provided in `source/mujoco/`; generated
builds, install trees and unrelated example model collections are excluded.
`native_changes.patch` records the local engine changes already present in this
source tree; do not apply it a second time.

The changes extend file serialization/loading beyond signed-int MJB buffer sizes
without changing the public buffer ABI. An optional conservative broad-phase
bound is enabled only by `GATE_BVH_BROADPHASE=1`; the default is unchanged.
The runtime libraries use `$ORIGIN` search paths. Source filenames used in error
diagnostics are relative names, and no executable sections were changed while
removing build-machine identifiers. `identity.json` stores the exported engine
hash; `sanitization.json` records this diagnostic-only cleanup.

Rebuild with a recent GCC/C++ toolchain, CMake, Ninja and CPython 3.11 development
headers. The CMake dependency fetches need network access and are pinned by the
upstream build files:

```bash
cmake -S air_ground_sim/runtime/source/mujoco -B /tmp/mujoco-build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=OFF -DCMAKE_INSTALL_PREFIX=/tmp/mujoco-install
cmake --build /tmp/mujoco-build --parallel
cmake --install /tmp/mujoco-build
```

Python binding build and wheel instructions are in
[source/mujoco/python/README.md](source/mujoco/python/README.md). Use the rebuilt
engine and matching headers when building bindings; do not substitute an
unmodified downloaded engine. Install the resulting bindings and shared
libraries into `runtime/python/mujoco`, update the engine hash in `identity.json`,
and rerun the CPU checks and profile verification. The supplied precompiled
runtime is the tested path; rebuilding with another toolchain produces a new
binary identity and requires validation.

Required system libraries include glibc and the selected OpenGL/EGL/OSMesa/GLFW
backend. Check dependencies with `ldd runtime/python/mujoco/_functions*.so`.
MuJoCo and GCC license texts and runtime exceptions are retained alongside the
native files.
