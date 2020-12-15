# %% Import commonly used args
import numpy as np

# %% Define environment
from env2048 import Game2048Env


def eval_state(x): return 2 * np.log2(x.ravel()) + np.sum(x.ravel())
env = Game2048Env(eval_state)
