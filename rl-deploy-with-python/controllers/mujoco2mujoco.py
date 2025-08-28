
# import mujoco
# import mujoco_viewer
# import numpy as np
# import time
# from mujoco_playground import wrapper
# from mujoco_playground import registry

import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# Graphics and plotting.
print("Installing mediapy:")
import subprocess
import sys

# Check if ffmpeg is installed, install if needed
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    print("ffmpeg is already installed")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Installing ffmpeg...")
    subprocess.run(['sudo', 'apt', 'update'], check=True)
    subprocess.run(['sudo', 'apt', 'install', '-y', 'ffmpeg'], check=True)

# Install mediapy if not already installed
try:
    import mediapy as media
    print("mediapy is already installed")
except ImportError:
    print("Installing mediapy...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mediapy'], check=True)
    import mediapy as media

import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp