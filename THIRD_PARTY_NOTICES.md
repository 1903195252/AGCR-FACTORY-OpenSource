# Licenses and upstream components

## Project code and assets

Project-authored software and configuration in V1.0 are distributed under
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

Illustrations, photographs, model geometry and other non-software assets retain
their existing rights. This code license does not relicense external model
weights, datasets or third-party materials.

## Included third-party materials

- **MuJoCo**: the upstream Apache-2.0 license is retained at
  [`air_ground_sim/runtime/MUJOCO_LICENSE`](air_ground_sim/runtime/MUJOCO_LICENSE)
  and in the vendored engine source. Source copyright notices are preserved.
- **GCC runtime libraries**: applicable GPL and GCC Runtime Library Exception
  notices are retained in
  [`air_ground_sim/runtime/GCC_LICENSES`](air_ground_sim/runtime/GCC_LICENSES).
- Embedded upstream components retain their license files in the source tree.

## External code and weights

DINOv2, Wan2.2, SynCamMaster, ShareVerse and metric dependencies are external
dependencies. Their model weights and upstream repositories are not bundled in
the project source archive. Download and use them under their respective terms.
The package requirements do not change those terms.

Upstream attribution in license files and vendored source is intentionally
retained.
