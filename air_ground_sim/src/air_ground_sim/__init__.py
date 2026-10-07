"""Public, policy-independent API. MuJoCo is loaded lazily when Simulator is used."""
import os
import platform
import sys
from .paths import BUNDLE

__version__ = '0.1.0'


def _native_runtime():
    if sys.version_info[:2] != (3, 11) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Bundled native runtime requires CPython 3.11 on Linux x86_64; see runtime/BUILD.md.')
    runtime = BUNDLE/'runtime/python'
    if not (runtime/'mujoco').is_dir():
        raise RuntimeError('Missing bundled runtime; set AIR_GROUND_SIM_ROOT to the full extracted bundle.')
    if 'mujoco' in sys.modules and PathOf(sys.modules['mujoco']) != (runtime/'mujoco').resolve():
        raise RuntimeError('A different MuJoCo is already imported. Import air_ground_sim before mujoco.')
    if str(runtime) not in sys.path: sys.path.insert(0, str(runtime))
    # Load the private C++ runtime before NumPy/other extensions can pin an old
    # system SONAME on older Linux distributions. The supported NumPy is locked.
    import ctypes
    for name in ('libgcc_s.so.1','libstdc++.so.6'):
        ctypes.CDLL(str(runtime/'mujoco'/name),mode=ctypes.RTLD_GLOBAL)
    os.environ.setdefault('MUJOCO_GL', 'glfw' if os.environ.get('DISPLAY') else 'osmesa')


def PathOf(module):
    from pathlib import Path
    return Path(module.__file__).resolve().parent


def __getattr__(name):
    if name == 'Simulator':
        _native_runtime()
        from .environment import Simulator
        return Simulator
    raise AttributeError(name)


__all__ = ['Simulator']

# Configure the private extension search path for direct submodule imports too.
# This does not load a model, create a GL context or import MuJoCo itself.
_native_runtime()
