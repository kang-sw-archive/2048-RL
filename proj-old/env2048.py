import gym
import numpy as np
import cv2

from game2048 import logic
from gym import spaces

from typing import Callable

GRID_SIZE = 5
_ACTION_SPACE = [logic.up, logic.right, logic.down, logic.left]
_N_ACTIONS = len(_ACTION_SPACE)


def NUM_ACTIONS():
    return _N_ACTIONS


def default_eval_method(state):
    return np.sum(np.sum(state))


def log2_on_game(state) -> np.ndarray:
    nparr = np.array(state)
    mask = (nparr == 0).astype(np.int32)
    return np.log2(mask + nparr)


def calc_score(state):
    return np.sum(np.sum(v * (log2_on_game(v) - 1)))


class Game2048Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, eval_method: Callable[[np.ndarray], float] = default_eval_method):
        super(Game2048Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(_N_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=float('inf'),
                                            shape=(GRID_SIZE, GRID_SIZE), dtype=np.float)

        # 게임 스테이트
        self.__state = logic.new_game(GRID_SIZE)
        self.__evaluate = eval_method

        self.render_size = 480

    def step(self, action):
        self.__state, successful = _ACTION_SPACE[action](self.__state)
        state = np.array(self.__state)

        eval = self.__evaluate(state)
        reward = eval - self.__prev_eval
        self.__prev_eval = eval

        game_state = logic.game_state(self.__state)

        return np.log2(state), reward, game_state != 'not over', None

    def reset(self):
        self.__prev_eval = 0.
        return self.__state_log2  # reward, done, info can't be included

    def render(self, mode='human'):
        if mode == 'human':
            cell_size = self.render_size / GRID_SIZE
            half = cell_size / 2
            output = np.ndarray((cell_size, cell_size))
            state = self.__state

            FONTFACE = cv2.FONT_HERSHEY_PLAIN
            FONT_SCALE = 1
            FONT_THICK = 1

            for y, x in np.ndindex((GRID_SIZE, GRID_SIZE)):
                rp0 = (cell_size * (x + 0), cell_size * (y + 0))
                rp1 = (cell_size * (x + 1), cell_size * (y + 1))

                cv2.rectangle(output, rp0, rp1, (0, 0, 0), 1)
                extent = cv2.getTextSize(str(state[y][x]), FONTFACE, FONT_SCALE, FONT_THICK)
                cv2.putText(output, str(state[y][x]), rp0 + (half, half) - extent[::-1] * 0.5, FONTFACE, FONT_SCALE, (0, 0, 0), FONT_THICK)

            return output

        raise Exception('unsupported mode')

    def close(self):
        pass

    @property
    def __state_log2(s):
        return np.log2(np.array(s.__state))
