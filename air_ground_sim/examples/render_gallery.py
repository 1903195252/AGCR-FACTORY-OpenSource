"""Capture the complete factory and physically simulated robots with MuJoCo.

Run with the bundle's Python 3.11 dependencies and MUJOCO_GL=egl or osmesa.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from air_ground_sim import Simulator
from air_ground_sim.io import save_physics_state, restore_physics_state, export_observations
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="factory_industrial")
    parser.add_argument("--appearance", choices=['industrial','legacy'], default='industrial')
    parser.add_argument("--sensors", action="store_true", help="Also export both robots' RGB-D and LiDAR")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--replay", type=Path, help="Render a saved state instead of repeating physics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    print("Loading verified full factory model...", flush=True)
    with Simulator(args.profile,appearance=args.appearance) as sim:
        print(f"Loaded {sim.model.ngeom} geoms, {sim.model.nmesh} meshes", flush=True)
        sim.model.vis.global_.offwidth = args.width
        sim.model.vis.global_.offheight = args.height
        sim.model.vis.map.zfar = 4.
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        with mujoco.Renderer(sim.model, width=args.width, height=args.height,
                             max_geom=sim.model.ngeom + 2048) as renderer:
            def capture(filename, title, lookat, distance, azimuth, elevation):
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(camera)
                camera.lookat[:] = lookat
                camera.distance = distance
                camera.azimuth = azimuth
                camera.elevation = elevation
                sim.prepare_visuals(lookat,distance,renderer)
                mujoco.mj_forward(sim.model, sim.data)
                renderer.update_scene(sim.data, camera=camera, scene_option=option)
                pixels = renderer.render().copy()
                Image.fromarray(pixels).save(args.output / filename)
                records.append(dict(file=filename, title=title, time_s=sim.time,
                    lookat=np.asarray(lookat).tolist(), distance=distance,
                    azimuth=azimuth, elevation=elevation))
                print(f"Saved {filename} at t={sim.time:.3f}s", flush=True)

            capture("01_factory_overview.png", "Factory overview", [0, 0, 0], 620, 130, -53)
            capture("02_factory_top.png", "Factory top view", [0, 0, 0], 420, 90, -89.5)
            initial = sim.state()
            capture("03_team_on_ground.png", "Air-ground team / initial state",
                    [-180, -81.75, .16], 1.5, 135, -29)
            air = initial["air"]["position_m"].copy()
            air[2] = 1.5
            ground = initial["ground"]["position_m"][:2] + [.5, 0]
            sim.command(air=np.r_[air, 0.], ground=np.r_[ground, 0.], duration=4.)
            trajectory = []
            if args.replay:
                restore_physics_state(sim.model, sim.data, args.replay, sim.model_spec["model_sha256"])
                state = sim.state()
                trajectory.append([sim.time, *state["air"]["position_m"],
                                   *state["ground"]["position_m"]])
                print(f"Restored physical state at t={sim.time:.3f}s", flush=True)
            else:
                print("Simulating 2 seconds with original flight and wheel controllers...", flush=True)
                for _ in range(20):
                    state = sim.step(round(.1 / sim.timestep))
                    trajectory.append([sim.time, *state["air"]["position_m"],
                                       *state["ground"]["position_m"]])
                    if len(trajectory) % 5 == 0:
                        print(f"Physics t={sim.time:.1f}s", flush=True)
            air = state["air"]["position_m"]
            dog = state["ground"]["position_m"]
            capture("04_team_in_motion.png", f"Air-ground team / t = {sim.time:.1f} s",
                    (air + dog) / 2, 1.8, 135, -18)
            capture("05_uav_closeup.png", f"Folding quadrotor / t = {sim.time:.1f} s",
                    air, 1.05, 135, -27)
            capture("06_robot_dog_closeup.png", f"Wheeled quadruped / t = {sim.time:.1f} s",
                    dog + [0, 0, -.025], .65, 140, -23)
            save_physics_state(sim, args.output / "physics_state.npz")
            if args.sensors:
                export_observations(sim.observe(width=320,height=240,rings=4,azimuth_samples=60),
                                    args.output / "sensors")
            np.savetxt(args.output / "trajectory.csv", trajectory, delimiter=",",
                       header="time_s,air_x,air_y,air_z,ground_x,ground_y,ground_z", comments="")
            report = dict(profile=sim.profile, model_sha256=sim.model_spec["model_sha256"],
                          appearance=sim.appearance_name,
                          resolution=[args.width, args.height], mujoco_version=mujoco.__version__,
                          physics_steps_this_run=sim.steps, replay_source=str(args.replay) if args.replay else None,
                          warnings=[int(w.number) for w in sim.data.warning],
                          air_displacement_m=(air-initial["air"]["position_m"]).tolist(),
                          ground_displacement_m=(dog-initial["ground"]["position_m"]).tolist(),
                          images=records, elapsed_wall_s=time.monotonic()-started)
            (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    sheet = Image.new("RGB", (1200, 3 * 405), "#edf0f3")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        x, y = (i % 2) * 600, (i // 2) * 405
        with Image.open(args.output / record["file"]) as frame:
            frame.thumbnail((592, 370), Image.Resampling.LANCZOS)
            sheet.paste(frame, (x+4, y+4))
        draw.text((x+12, y+380), record["title"], fill="#192d43")
    sheet.save(args.output / "gallery_preview.jpg", quality=94)
    print(f"Finished: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
