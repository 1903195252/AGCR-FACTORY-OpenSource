"""Small deterministic checks; full-model integration is exercised by run.py demo."""
import json
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from air_ground_sim import _native_runtime
_native_runtime()
from air_ground_sim.paths import CONFIG
from air_ground_sim.control.references import References
from air_ground_sim.control.native_air import PhysicsTickClock
from air_ground_sim.control.ground_rate import YawRateFeedback
from air_ground_sim.kinematics.folding import FourBarGeometry
from air_ground_sim.dynamics.rotors import RotorParameters


class TestCore(unittest.TestCase):
    def test_reference_endpoints_and_replanning(self):
        refs=References([0,0,1,0],[0,0,0])
        refs.goto('air',[1,2,3,.5],0,4)
        np.testing.assert_allclose(refs.reference('air',4)[0],[1,2,3,.5],atol=1e-12)
        np.testing.assert_allclose(refs.reference('air',4)[1:],0)
        before=refs.reference('air',2)
        refs.goto('air',[2,1,2,1],2,3)
        for a,b in zip(before,refs.reference('air',2)):np.testing.assert_allclose(a,b,atol=1e-12)

    def test_heading_uses_short_arc(self):
        refs=References([0,0,1,3.1],[0,0,0])
        refs.goto('air',[0,0,1,-3.1],0,1)
        self.assertLess(abs(refs.reference('air',1)[0][-1]-3.1),.1)

    def test_bad_commands(self):
        refs=References([0,0,1,0],[0,0,0])
        for goal,duration in [([1,2],1),([0,0,float('nan'),0],1),([0,0,1,0],0)]:
            with self.assertRaises(ValueError):refs.goto('air',goal,0,duration)

    def test_integer_control_clock(self):
        clock=PhysicsTickClock(.0005,.004,0)
        self.assertEqual(sum(clock.take_update(i*.0005) for i in range(16001)),2001)
        with self.assertRaises(ValueError):clock.take_update(9.)

    def test_mechanism_inverse(self):
        geom=FourBarGeometry(json.loads((CONFIG/'hardware_spec.json').read_text()))
        for alpha in np.linspace(geom.alpha_fold,geom.alpha_deploy,30):
            self.assertAlmostEqual(geom.configuration(geom.inverse(alpha))[2],alpha,places=10)

    def test_ground_rate_and_rotor_validation(self):
        servo=YawRateFeedback()
        for _ in range(5000):self.assertLessEqual(abs(servo.update(.25,0,.0005)),2.)
        RotorParameters().validate()
        with self.assertRaises(ValueError):RotorParameters(tau=-1).validate()

    def test_no_learning_import(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('transformers',sys.modules)
        self.assertFalse(any(name.startswith('experiments.') for name in sys.modules))

    def test_native_state_roundtrip(self):
        import mujoco
        from air_ground_sim.io import save_physics_state,restore_physics_state
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model)
        for _ in range(10):mujoco.mj_step(model,data)
        sim=SimpleNamespace(model=model,data=data,time=float(data.time),model_spec={'model_sha256':'test-model'})
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'state.npz';save_physics_state(sim,file)
            recovered=mujoco.MjData(model);restore_physics_state(model,recovered,file,'test-model')
            np.testing.assert_array_equal(data.qpos,recovered.qpos)
            np.testing.assert_array_equal(data.qvel,recovered.qvel)
            self.assertEqual(data.time,recovered.time)
            with self.assertRaises(ValueError):restore_physics_state(model,recovered,file,'wrong-model')


if __name__=='__main__':unittest.main()
