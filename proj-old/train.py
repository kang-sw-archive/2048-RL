
# %% Define learning utility
import itertools
from numba import jit
import tensorflow.keras as keras
import tensorflow as tf
import math as m
import numpy as np
from game2048 import logic
from typing import Callable
from collections import deque

from tensorflow.python.framework.ops import _run_using_default_session

GRID_SIZE = 5
_ACTION_SPACE = [logic.up, logic.right, logic.down, logic.left]
_N_ACTIONS = len(_ACTION_SPACE)


def NUM_ACTIONS():
    return _N_ACTIONS


def log2_on_game(state) -> np.ndarray:
    nparr = np.array(state)
    mask = (nparr == 0).astype(np.int32)
    return np.log2(mask + nparr)


def epsilon_greedy_policy(model: keras.models.Sequential, state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(_N_ACTIONS)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


def calc_score(state):
    v = np.array(state)
    return np.sum(np.sum(v * (log2_on_game(v) - 1)))


class ReplayQueue:
    def __init__(self, n_max) -> None:
        super().__init__()
        self.__array = deque(maxlen=n_max)
        self.__max = n_max

    @property
    def capacity(self):
        return self.__max

    @property
    def size(self):
        return len(self.__array)

    def push(self, state, action, reward, next_state, done):
        self.__array.append((state, action, reward, next_state, done))

    def sample(self, n_batch):
        indices = np.random.randint(len(self.__array), size=n_batch)
        batch = [self.__array[i] for i in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones


class Trainer:
    def __init__(
            s,
            model: keras.Model,
            optimizer: keras.optimizers.Adam,
            loss_fn: Callable,
            discount_factor=1,
            max_replay=40,) -> None:
        super().__init__()
        s.__model = model
        s.__replays = ReplayQueue(max_replay)
        s.__state = logic.new_game(GRID_SIZE)
        s.__loss_fn = loss_fn
        s.__optimizer = optimizer
        logic.c.GRID_LEN = GRID_SIZE

        s.discount_factor = discount_factor
        s.train_start_replay_buffer_size_rate = 0.1

        # initialize target model
        s.__target = keras.models.clone_model(s.__model)
        s.update_target_weights()

        # 무의미한 실행 횟수입니다.
        # 발생 즉시 보상을 큰 폭으로 깎습니다.
        s.redundant_action_penalty = 100
        s.__redundant_action_cnt = 0

        s.reward_fn = lambda reward: reward
        s.reset_as: list = None

    @property
    def state(self):
        return self.__state

    @property
    def redundant_action_count(self):
        return self.__redundant_action_cnt

    @property
    def model(self):
        return self.__model

    def refresh_weights(s, weights):
        s.__model.set_weights(weights)
        s.update_target_weights()

    def update_target_weights(s):
        s.__target.set_weights(s.__model.get_weights())

    def env_reset(s):
        s.__redundant_action_cnt = 0

        if s.reset_as == None:
            s.__state = logic.new_game(GRID_SIZE)
        else:
            s.__state = s.reset_as.copy()

        s.__prev_reward = s.calc_reward()

    def calc_reward(s):
        state = np.array(s.__state)
        return s.reward_fn(state) - s.redundant_action_penalty * s.redundant_action_count

    def training_step(s, n_batch):
        states, actions, rewards, next_states, dones = s.__replays.sample(n_batch)
        next_Q_values = s.__model.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, _N_ACTIONS).numpy()
        next_best_Q_values = (s.__target.predict(next_states) * next_mask).sum(axis=1)
        target_Q_values = (rewards +
                           (1 - dones) * s.discount_factor * next_best_Q_values)

        mask = tf.one_hot(actions, _N_ACTIONS)

        with tf.GradientTape() as tape:
            all_Q_values = s.__model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(s.__loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, s.__model.trainable_variables)
        s.__optimizer.apply_gradients(zip(grads, s.__model.trainable_variables))

        return loss

    def step(s, epsilon=0):
        valid_input_mask = np.array([_ACTION_SPACE[idx](s.state)[1] for idx in range(_N_ACTIONS)])
        valids = [idx for idx in range(_N_ACTIONS) if valid_input_mask[idx]]
        random_action = valids[np.random.randint(len(valids))]

        probability = []
        log_state = log2_on_game(s.state)
        # log_state = np.array(s.state)

        if epsilon > np.random.rand():
            action = random_action
            valid = False
        else:
            probabilities = s.model(log_state[np.newaxis])
            probability = probabilities[0].numpy()
            action = np.argmax(probabilities[0])
            valid = True

            if action not in valids:
                action = random_action
                valid = False

        s.__state, successful = _ACTION_SPACE[action](s.state)
        next_log_state = log2_on_game(s.__state)
        # next_log_state = np.array(s.state)

        ##
        if successful:
            logic.add_two(s.__state)
        else:
            raise 'logic should not enter here!'
            s.__redundant_action_cnt += 1
        reward = s.calc_reward()

        game_state = logic.game_state(s.__state)
        done = True if game_state == 'win' or game_state == 'lose' else False

        s.__replays.push(log_state, action, reward - s.__prev_reward, next_log_state, done)
        s.__prev_reward = reward
        return reward, action, done, valid, probability

    def run_single_episode(s, epsilon):
        s.env_reset()

        reward, n_steps = 0, 0
        for n_steps in itertools.count(1):
            reward, _, done, *_ = s.step(epsilon)
            if done:
                break

        return reward, n_steps

    def train_single_episode(s, epsilon, n_batch=32):
        reward, n_steps = s.run_single_episode(epsilon)

        loss = None
        if s.__replays.size > s.__replays.capacity * s.train_start_replay_buffer_size_rate:
            loss = s.training_step(n_batch)
        return reward, n_steps, loss

    def retrieve_weight(self):
        return self.__model.get_weights()


# %%
