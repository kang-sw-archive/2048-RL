# %% Define game constants
'''
    pip install -r requirements.txt로 필요 구성 요소를 모두 설치한 후, 가상 환경의 Lib 디렉토리에 proj/2048-python-master/game2048을 복사해야 합니다.

    Python 버전 3.8.5에서 테스트됨
'''
import itertools
from game2048 import logic
import numpy as np
from tensorflow.python.keras.layers.core import Activation
import matplotlib.pyplot as plt
import train

import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


GRID_SIZE = 4
N_ACTIONS = g2048.NUM_ACTIONS()
g2048.GRID_SIZE = GRID_SIZE
MODEL_TYPE = 'DNN'


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


# %% Create DQN
import tensorflow.keras as keras
import train

INPUT_SHAPE = [GRID_SIZE, GRID_SIZE, 1]
MODEL_PATH = 'model-DQN.h5'


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


import os
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print('Model loaded')
else:
    print('Creating model ', MODEL_TYPE, '...')
    input_state = keras.layers.Input(shape=INPUT_SHAPE, name="state_input")
    from keras import layers

    if MODEL_TYPE == 'DNN':
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
    elif MODEL_TYPE == 'CNN0':
        CNN0_CONV_ACTIVATION = 'relu'
        CNN0_DENSE_ACTIVATION = 'relu'
        CNN0_OUTPUT_ACTIVATION = 'linear'

        output = input_state
    else:
        # input layers
        # input_valid_actions = keras.layers.Input(shape=[N_ACTIONS, 1], name="state_valid_actions")

        KSIZE_MAX = (GRID_SIZE & ~1) | 1  # 작거나 같은 가장 가까운 홀수로 만듭니다.

        hidden = concat_parallel_layers(
            input_state, [[
                keras.layers.Conv2D(64, 1, activation='relu', padding='same'),
                keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
                keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            ], [
                keras.layers.Conv2D(128, 1, activation='relu', padding='same'),
            ]]
        )

        hidden = keras.layers.Flatten()(hidden)
        hidden = concat_layers(hidden, [
            keras.layers.Dense(256, activation='relu'),
        ])

        output = keras.layers.Dense(N_ACTIONS, name="output")(hidden)

    model = keras.Model(inputs=[input_state], outputs=[output])
    print('New model was created.')


from keras.utils.vis_utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% Load previous session variables
import pickle
REWARDS_PATH = 'model-record.bin'

if os.path.exists(REWARDS_PATH):
    with open(REWARDS_PATH, 'rb') as fp:
        total = pickle.load(fp)
    print('[Rewards] loaded')
else:
    total = {
        'iterations': [0],
        'rewards': [0],
        'losses': [0],
        'steps': [0],
        'evaluations': [(0, 0., 0.)],
    }

total_iters = total['iterations']
total_rewards = total['rewards']
total_losses = total['losses']
total_steps = total['steps']
total_evals = total['evaluations']

ITER_OFFSET = total_iters[len(total_rewards) - 1]

# %% Create trainer
LEARNING_RATE_STEPS = 1000
INITIAL_LEARNING_RATE = 0.00025
LEARNING_RATE_DECAY = 0.96

initial_learning = INITIAL_LEARNING_RATE * pow(LEARNING_RATE_DECAY, ITER_OFFSET / LEARNING_RATE_STEPS)

trainer = g2048.Trainer(
    model,
    keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning, LEARNING_RATE_STEPS, LEARNING_RATE_DECAY, True)),
    keras.losses.mean_squared_error,
    discount_factor=0.95,
    max_replay=4096
)

import math
import numpy as np
trainer.redundant_action_penalty = math.pi
trainer.reset_as = logic.new_game(GRID_SIZE)

MODEL_TYPE = -1
if MODEL_TYPE == 0:
    trainer.reward_fn = lambda v: g2048.calc_score(v)
elif MODEL_TYPE == 1:
    trainer.reward_fn = lambda v: math.log2(np.max(np.max(v)))
elif MODEL_TYPE == 2:
    trainer.reward_fn = lambda v: math.log2(np.sum(np.sum(v)))
else:
    trainer.reward_fn = lambda v: np.sum(np.sum(v))


# %% define utility

import itertools
from train import log2_on_game


def render_2048_grid(grid):
    grid = np.array(grid)
    plt.imshow(log2_on_game(grid))
    for (i, j) in itertools.product(range(GRID_SIZE), range(GRID_SIZE)):
        plt.text(j, i, grid[i][j])


# %% Iterate DQN training

N_ITERATION = 10000
N_BATCH = 256

NEWLINE_PERIOD = 1000000000000
CHECKPOINT_PERIOD = 100
EVALUATION_PERIOD = 5
RECORD_PERIOD = 5
tr = trainer

EVALUATION_PERIOD = min(EVALUATION_PERIOD, CHECKPOINT_PERIOD)
EVALUATION_PERIOD -= CHECKPOINT_PERIOD % EVALUATION_PERIOD

for iter in range(ITER_OFFSET + 1, N_ITERATION):
    epsilon = (1 - iter / N_ITERATION) * 0.5
    reward, steps, loss = tr.train_single_episode(epsilon, N_BATCH)

    if iter % RECORD_PERIOD == 0:
        total_iters.append(iter)
        total_rewards.append(reward)
        total_losses.append(loss)
        total_steps.append(steps)

    if iter % NEWLINE_PERIOD == 0:
        print()

    if iter > 0 and iter % CHECKPOINT_PERIOD == 0 or iter == N_ITERATION - 1:
        tr.update_target_weights()
        tr.model.save(MODEL_PATH)

        try:
            from IPython.display import clear_output
            clear_output()
        except:
            pass

        with open(REWARDS_PATH, 'wb') as fp:
            pickle.dump(total, fp)

        print(f'\n\n\n>>>>>>> Checkpoint reached ... ')
        plt.ylabel('reward')
        plt.scatter(total_iters, total_rewards, s=1.5, c='black')
        plt.tick_params(axis='y')
        plt.legend(['reward'])
        show_plot()

        plt.ylabel('steps')
        plt.scatter(total_iters, total_steps, s=1.5, c='black')
        plt.tick_params(axis='y')
        plt.legend(['steps'])
        show_plot()

        plt.ylabel('loss')
        plt.plot(total_iters, total_losses, color='red')
        plt.tick_params(axis='y')
        plt.legend(['loss'])
        plt.yscale('log')
        show_plot()

        print([zip(*total_evals)])
        x, y_reward, y_invalid_rate = zip(*total_evals)
        plt.scatter(x, y_reward, s=1.5, c='black')
        plt.tick_params(axis='y')
        plt.legend(['reward evaluations'])
        show_plot()

        plt.scatter(x, y_invalid_rate, s=1.5, c='black')
        plt.tick_params(axis='y')
        plt.legend(['invalid prediction rate'])
        show_plot()

        print('\n')

    if iter % EVALUATION_PERIOD == 0:
        # Try single episode
        # Evaluate
        trainer.env_reset()
        reward = 0
        hist_tot = [0] * 4
        hist_invalid = [0] * 4
        n_invalid = 0
        probs = []
        for step in itertools.count(1):
            # prob = trainer.predict(trainer.state).numpy()
            reward, action, done, valid, prob = trainer.step()
            probs.append(prob)
            str = [f'{prob:.5f}' for prob in prob]

            if iter % CHECKPOINT_PERIOD == 0:
                print(f'\r    Step {step:4}: {action=:2}, {reward=:6}, {str} -> {np.argmax(prob)}', end='')

            hist_tot[action] += 1

            if not valid:
                n_invalid += 1
                hist_invalid[action] += 1

            if done or trainer.redundant_action_count > 10:
                break

            if step % 50 == 0 and iter % CHECKPOINT_PERIOD == 0:
                print()

        if iter % CHECKPOINT_PERIOD == 0:
            print('\n    occurences         = ', hist_tot, ', max_probabilites = ', np.max(np.array(probs), axis=0))
            print(f'    random_occurences  =  {hist_invalid}, invalid_hits = {n_invalid} over {len(probs)}')
            render_2048_grid(trainer.state)
            show_plot()
            print()

        total_evals.append((iter, g2048.calc_score(tr.state), n_invalid / len(probs)))

    print(f'\rEPISODE {iter + 1:-8} :: For [{steps:-5}] steps -> Got Reward: {reward:8} (e: {epsilon:.3f})', end='')


# %% Single iteration
if False:
    import pickle
    episode_rewards = []

    def ep_cb(epindex, actions, rewards):
        reward = np.sum(rewards)
        print(f'\r    Episode {epindex + 1:4}: steps = {len(actions):4}, reward = {reward}', end='')
        episode_rewards.append(reward)

        if (epindex + 1) % 10 == 0:
            print()

    N_ITER = 1000
    N_EPISODES = 8

    REWARDS_PATH = 'reward_list.bin'

    if os.path.exists(REWARDS_PATH):
        with open(REWARDS_PATH, 'rb') as fp:
            rewards = pickle.load(fp)
        print('[Rewards] loaded')
    else:
        rewards = []

    for iter in range(N_ITER):
        print(f'\n\n\n\nITERATION {iter:05} >>>>>>>> \n')
        # Evaluate
        trainer.env_reset()
        reward = 0
        hist_tot = [0] * 4
        hist_invalid = [0] * 4
        n_invalid = 0
        probs = []
        for step in itertools.count(1):
            # prob = trainer.predict(trainer.state).numpy()
            reward, action, done, _, valid, prob = trainer.step()
            probs.append(prob)
            str = [f'{prob:.5f}' for prob in prob]
            print(f'\r    Step {step:4}: {action=:2}, {reward=:6}, {str} -> {np.argmax(prob)}', end='')

            hist_tot[action] += 1

            if not valid:
                n_invalid += 1
                hist_invalid[action] += 1

            if done or trainer.redundant_action_count > 10:
                break

            if step % 50 == 0:
                print()

        print('\n    occurences         = ', hist_tot, ', max_probabilites = ', np.max(np.array(probs), axis=0))
        print(f'    random_occurences  =  {hist_invalid}, invalid_hits = {n_invalid} over {len(probs)}')

        print('\n')
        episode_rewards = []
        trainer.training_iteration(N_EPISODES, ep_cb)

        trainer.model.save(MODEL_PATH)
        with open(REWARDS_PATH, 'wb') as fp:
            pickle.dump(rewards, fp)

        rewards.append(reward)
        render_2048_grid(trainer.state)
        show_plot()

        if (iter) % 3 == 0:
            plt.plot(rewards)
            show_plot()

    plt.plot(rewards)
    plt.show(block=False)
    print("\nDone")

# %%
