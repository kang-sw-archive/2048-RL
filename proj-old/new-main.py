# %%
import env2048 as g2048
import numpy as np
import keras

from enum import Enum, auto
from rl.agents import DQNAgent

# %% Constants
GRID_SIZE = 4
N_ACTIONS = g2048.NUM_ACTIONS()
g2048.GRID_SIZE = GRID_SIZE


class ModelType:
    DNN, CNN0, CNN1 = 'DNN', 'CNN0', 'CNN1'


class ScoreEvalType(Enum):
    def SUM_SIMPLE(st): return np.sum(np.sum(st))
    def MAX(st): return np.max(np.max(st))
    def MAX_LOG2(st): return np.log2(np.max(np.max(st)))
    def SCORE(st): return g2048.calc_score(st)


MODEL_TYPE = ModelType.DNN
SCORING_METHOD = 'SUM_SIMPLE'

# Serialization arguments
MODEL_DIR = 'models'
MODEL_PREFIX = 'dqn_2048'
MODEL_REVNUM = 'rev0'
TOTALS_PATH = 'model-record.pkl'

# DQN parameters
N_STEPS_ANNEALED = int(1e5)
N_STEPS_WARMUP = int(5000)
N_STEPS_TRAINING = int(5e6)
N_TARGET_MODEL_UPDATE_PERIOD = 1000

N_MAX_EPISODE_STEPS = 20000
N_LOG_INTERVAL = 20000

MEMORY_LIMIT = 6000

WINDOW_LENGTH = 1

# MODES
PERFORM_TRAIN = False
PERFORM_TEST = False

# %% Miscellaneous definitions
import os
FILENAME_PREFIX = f'{MODEL_PREFIX}-{MODEL_TYPE}-{MODEL_REVNUM}-{SCORING_METHOD}'
PATH_PREFIX = os.path.join(MODEL_DIR, FILENAME_PREFIX)

FILEPATH_MODEL = PATH_PREFIX + '_model.h5'
FILENAME_WEIGHT = PATH_PREFIX + '_weights_{step}.h5'
FILENAME_MEMORY = PATH_PREFIX + '_memory_{step}.h5'

# %% Miscellaneous utilities to stabilize operation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


def in_ipynb():
    try:
        cfg = get_ipython().config
        return True
    except NameError:
        return False


def show_plot():
    import matplotlib.pyplot as plt
    if in_ipynb():
        plt.show()


# %% Model utils
import train


def concat_layers(input, layers):
    hidden = input
    for layer in layers:
        hidden = layer(hidden)
    return hidden


def concat_parallel_layers(input, ll_layers):
    output_layers = []
    for layers in ll_layers:
        o_layer = concat_layers(input, layers)
        output_layers.append(o_layer)
    return keras.layers.concatenate(output_layers, axis=3)


def create_model(model_type):
    INPUT_SHAPE = [GRID_SIZE, GRID_SIZE, 1]
    input_state = keras.layers.Input(shape=INPUT_SHAPE, name="state_input")
    output = None  # suppress warning
    from keras import layers

    if model_type == ModelType.DNN:
        print('Creating DNN model ...')

        DNN_ACTIVATION = 'relu'
        DNN_OUTPUT_ACTIVATION = 'linear'

        hidden = concat_layers(
            input_state, [
                layers.Flatten(),
                layers.Dense(1024, activation=DNN_ACTIVATION),
                layers.Dense(512, activation=DNN_ACTIVATION),
                layers.Dense(256, activation=DNN_ACTIVATION),
            ]
        )

        output = layers.Dense(N_ACTIONS, activation=DNN_OUTPUT_ACTIVATION)(hidden)
    elif model_type == ModelType.CNN0:
        print('Creating CNN0 model ...')

        CNN0_CONV_ACTIVATION = 'relu'
        CNN0_DENSE_ACTIVATION = 'relu'
        CNN0_OUTPUT_ACTIVATION = 'linear'

        output = input_state
    else:
        raise 'Invalid model type specified'

    return keras.Model(inputs=input_state, outputs=output)


# %% Load previous session variables
import pickle

if os.path.exists(TOTALS_PATH):
    with open(TOTALS_PATH, 'rb') as fp:
        total = pickle.load(fp)
    print(TOTALS_PATH, 'loaded')
else:
    total = {
        'iterations': [0],
        'rewards': [0],
        'losses': [0],
        'steps': [0],
        'evaluations': [(0, 0., 0.)],
    }


def store_total():
    with open(TOTALS_PATH, 'rb') as fp:
        pickle.dump(total, fp)
    print(TOTALS_PATH, 'stored')


total_iters = total['iterations']
total_rewards = total['rewards']
total_losses = total['losses']
total_steps = total['steps']
total_evals = total['evaluations']

ITER_OFFSET = total_iters[len(total_rewards) - 1]

# %% Try load existing model. Else, create new
import pickle
from rl.policy import EpsGreedyQPolicy
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory
from fnmatch import fnmatch


def checkpoint(step=0):
    pickle.dump(memory, open(PATH_PREFIX + f'_memory_{step}.pkl', 'wb'))
    pickle.dump(model.get_weights(), open(PATH_PREFIX + f'_model_{step}.pkl', 'wb'))
    print(f'<checkpoint {step:-8}>')


def load_latest(pat):
    return pickle.load(open(fnmatch(os.listdir(MODEL_DIR), FILENAME_PREFIX + pat)[-1], 'rb'))


try:
    model = keras.models.load_model(FILEPATH_MODEL)
    memory = load_latest('_memory_*.pkl')
    weights = load_latest('_weights_*.pkl')

    print(f'Model successfully loaded ... {memory}')
except:
    model = create_model(MODEL_TYPE)
    memory = SequentialMemory(MEMORY_LIMIT, window_length=WINDOW_LENGTH)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    model.save(FILEPATH_MODEL)
    checkpoint()

    print('New model created')

# Keras solver
model.output.__len__ = lambda _: 1
model.output._keras_shape = model.output.shape


# %% Visualize loaded/created model
from keras.utils.vis_utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% instantiate DQN Agent

dqn = DQNAgent(
    model,
    nb_actions=N_ACTIONS,
    test_policy=EpsGreedyQPolicy(eps=.01),
    policy=LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0.01, nb_steps=N_STEPS_ANNEALED),
    processor=N_STEPS_WARMUP,
    memory=memory,
    gamma=.99,
    target_model_update=N_TARGET_MODEL_UPDATE_PERIOD,
    train_interval=4,
    delta_clip=1.)

dqn.compile(keras.optimizers.Adam(learning_rate=0.00025), metrics=keras.metrics.mean_squared_error)


# %% Training callbacks


# %% train mode
from rl.callbacks import ModelIntervalCheckpoint
from env2048 import Game2048Env

# Create environment
env = Game2048Env(SCORING_METHOD)

if PERFORM_TRAIN:
    WEIGHTS_PATH = PATH_PREFIX + '_weights.h5f'
    CHECKPOINT_WEIGHTS_PATH = PATH_PREFIX + 'weights_{step}.h5f'

    callbacks = []
    callbacks += [ModelIntervalCheckpoint(CHECKPOINT_WEIGHTS_PATH, interval=250000)]

    dqn.fit(env, callbacks=callbacks, nb_steps=N_STEPS_TRAINING, visualize=False,
            verbose=0, log_interval=N_LOG_INTERVAL, nb_max_episode_steps=N_MAX_EPISODE_STEPS)
