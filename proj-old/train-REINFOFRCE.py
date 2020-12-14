
# %% Define learning utility
from numba import jit
import tensorflow.keras as keras
import tensorflow as tf
import math as m
import numpy as np
from game2048 import logic
from typing import Callable
from collections import deque

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


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted


def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

# %% Replay Queue


class ReplayQueue:
    def __init__(self, n_max=40) -> None:
        super().__init__()
        self.__array = deque(maxlen=n_max)

    def push(self, state, action, reward, next_state, done):
        self.__array.append((state, action, reward, next_state, done))

    def sample(self, n_batch):
        indices = np.random.randint(len(self.__array), size=n_batch)
        batch = [self.__array[i] for i in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones


# %% Replay Queue
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

    def calc_reward(s):
        state = np.array(s.__state)
        return s.reward_fn(state) - s.redundant_action_penalty * s.redundant_action_count

    def step(s):
        log_state = log2_on_game(s.state)

        with tf.GradientTape() as tape:
            # 이동 가능한 위치 마스크를 선별합니다.
            valid_input_mask = np.array([_ACTION_SPACE[idx](s.state)[1] for idx in range(_N_ACTIONS)])
            valids = [idx for idx in range(_N_ACTIONS) if valid_input_mask[idx]]

            # probabilities = s.model([log_state, valid_input_mask.astype(np.float32)[np.newaxis]])
            probabilities = s.model(log_state[np.newaxis])
            action_prob = probabilities[0].numpy().max()

            original_action = np.argmax(probabilities[0])
            random_action = valids[np.random.randint(len(valids))]

            # 이동이 가능한 방향만 선별해 Target을 설정합니다.
            # 만약 이동이 불가능한 방향이 선정된 경우, action을 재지정해줍니다.
            if original_action in valids and action_prob > np.random.rand():
                valid = True
                action = original_action
            else:
                valid = False
                action = random_action

            y_target = tf.one_hot(action, _N_ACTIONS)
            loss = tf.reduce_mean(s.__loss_fn(y_target, probabilities[0]))
            # print(probabilities.numpy(), y_target.numpy(), loss.numpy())

        grads = tape.gradient(loss, s.model.trainable_variables)
        s.__state, successful = _ACTION_SPACE[action](s.state)

        ##
        if successful:
            logic.add_two(s.__state)
        else:
            s.__redundant_action_cnt += 1
        reward = s.calc_reward()

        game_state = logic.game_state(s.__state)
        done = True if game_state == 'win' or game_state == 'lose' else False

        return reward, action, done, grads, valid, probabilities[0].numpy()

    def play_n_episodes(s, n_episodes, callback=None):
        all_rewards = []
        all_grads = []
        for episode_index in range(n_episodes):
            current_rewards = []
            current_grads = []
            current_actions = []
            s.env_reset()

            prev_reward = s.calc_reward()

            while True:
                reward, action, done, grads, *_ = s.step()

                # Reward는 항상 delta-reward를 반영합니다.
                current_rewards.append(reward - prev_reward)
                prev_reward = reward

                current_grads.append(grads)
                current_actions.append(action)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_grads.append(current_grads)

            if callback != None:
                callback(episode_index, current_actions, current_rewards)

        return all_rewards, all_grads

    def training_iteration(s, n_episodes, episode_callback: Callable[[int, list, list], None] = None):
        all_rewards, all_grads = s.play_n_episodes(n_episodes, episode_callback)
        all_final_rewards = discount_and_normalize_rewards(all_rewards, s.discount_factor)
        all_mean_grads = []

        model = s.model
        optimizer = s.__optimizer

        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)],
                axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    def retrieve_weight(self):
        return self.__model.get_weights()


# %%
