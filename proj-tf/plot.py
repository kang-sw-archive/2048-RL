# %% Load log paths
import matplotlib.pyplot as plt
import os
from fnmatch import fnmatch

LOG_DIR = 'log/old-model'
log_paths = [os.path.join(LOG_DIR, s) for s in os.listdir(LOG_DIR) if fnmatch(s, "log-*.txt")]

# %% Load csv variables
import csv
tuples = []
colnames = []

for path in log_paths:
    with open(path, 'r') as fp:
        table = csv.reader(fp)
        colnames, _ = next(table), next(table)  # discard second row, which is filled with 0
        for row in table:
            tuples.append([float(e) if '.' in e else int(e) for e in row])

values = [e for e in zip(*tuples)]
steps = values[0]


# %% Find category and divide
import numpy as np
from scipy.signal import lfilter
from scipy.signal import savgol_filter

REWARD_COL_NAME = 'AverageReturnMetric'
LENGTH_COL_NAME = 'AverageEpisodeLengthMetric'

try:
    catidxs = []
    for name in (REWARD_COL_NAME, LENGTH_COL_NAME):
        catidxs += [[i for i in range(len(colnames)) if name in colnames[i]][0]]

    rewards, lengths = catidxs
    rewards, lengths = values[rewards], values[lengths]

    # Sort values by step order
    steps, rewards, lengths = [np.array(i) for i in zip(*sorted(zip(steps, rewards, lengths)))]

    # Optimize
    rate = 1
    if rate < 1:
        idxs = np.sort(np.random.choice(len(steps), int(len(steps) * rate)))
        steps = steps[idxs]
        rewards = rewards[idxs]
        lengths = lengths[idxs]

except IndexError:
    print(f'Element "{name}" does not exists within given list {colnames}')

# %%

try:
    def lpf(x): return savgol_filter(x, 56 | 1, 2)
    plt.plot(steps, lpf(rewards))
    plt.plot(steps, lpf(lengths))
    plt.scatter(steps, rewards, s=1, c='#ccccdd')
    plt.legend(['Avereage Reward', 'Average Episode Length', 'Raw Reward', ])
    plt.xlabel('Training Steps')
    plt.ylabel('Value')
    plt.show()
except NameError:
    pass

# %%
