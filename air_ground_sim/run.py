"""Launch from any working directory using Python 3.11; no installation needed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from air_ground_sim.cli import main
if __name__=='__main__':main()
