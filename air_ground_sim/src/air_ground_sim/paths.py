"""All runtime paths are relative to this independent bundle, or one explicit root."""
import os
from pathlib import Path

BUNDLE = Path(os.environ.get('AIR_GROUND_SIM_ROOT', Path(__file__).resolve().parents[2])).resolve()
CONFIG = BUNDLE / 'configs'
