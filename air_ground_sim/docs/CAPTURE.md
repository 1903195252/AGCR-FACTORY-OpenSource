# Synchronized capture and event labels

Two capture paths are retained. `tools/generate_nuscenes.py` exports the legacy
five-camera aerial rig and semantic/instance extensions. The paper RGB-D path
uses paired `CAM_AIR` / `CAM_GROUND` observations with scripted visibility events.
The formats are distinct and should not be mixed.

The paper configuration fixes 720×480, 5 Hz, six history frames and sixteen
future frames (3.2 s). Valid depth is optical-Z in millimetres. It defines 20
training, five validation and five test layouts, with family-level splits.
Events cover both-visible, aerial-only, ground-only, neither-visible and four
visibility transitions. Occluders include racks, manipulators, suspended loads,
vehicles and ground structures; targets include forklift, worker, AGV and UAV.

Install the capture dependency in the simulator environment:

```bash
.venv-sim/bin/python -m pip install 'h5py>=3.10,<4'
.venv-sim/bin/python air_ground_sim/tools/pack_paper_assets.py \
  --source air_ground_sim/assets/factory_industrial/authoring/model.xml \
  --output outputs/capture_assets.zip
.venv-sim/bin/python air_ground_sim/tools/test_paper_dataset.py
.venv-sim/bin/python air_ground_sim/tools/collect_paper_dataset.py \
  --config air_ground_sim/configs/dataset_paper_rgbd.json \
  --assets outputs/capture_assets.zip --out outputs/capture_pilot \
  --pilot --indices 0,1
```

`--prepare-only` constructs plans/layouts without rendering. Parallel workers use
`--worker i --workers n` with the same output root; atomic completion records and
quality checks guard each sequence. `capture_scheduler.py` and
`capture_task_queue.py` support shared paired-group dispatch and restart recovery.
`validate_paper_dataset.py` performs reference-label and split checks.

Capture includes synchronized RGB-D, target state, segmentation, isolated-target
and static depths, visible masks and camera transforms. Historical input and
future supervision are distinct fields. Visibility scripts are checked against
rendered projected/visible regions rather than accepted by their names alone.
The capture trajectories are prescribed kinematics; the full simulator's
actuator-driven robot control is a separate entry point.

The full dataset and training outputs are intentionally outside the source
archive. Capture code is included for reproducibility and for new configurations.
