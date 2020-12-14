from typing import Callable
from gin.config import _order_by_signature
from tensorflow import keras
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from game2048 import logic as game
from game2048.constants import GRID_LEN as GRID_SIZE
from tf_agents.trajectories import time_step as ts

import numpy as np
import cv2

_ACTION_SPACE = [game.up, game.right, game.down, game.left]
_N_ACTIONS = len(_ACTION_SPACE)


def NUM_ACTION():
    return _N_ACTIONS


def default_eval_method(state):
    return np.sum(np.sum(state))


def log2_on_game(state) -> np.ndarray:
    nparr = np.array(state)
    mask = (nparr == 0).astype(np.int32)
    return (np.log2(mask + nparr)).astype(np.float32).reshape((4, 4, 1))


def calc_score(v):
    return np.sum(np.sum(v * (log2_on_game(v) - 1)))


class Game2048Env(PyEnvironment):
    def __init__(
        self, evaluate: Callable[[np.ndarray], np.float] = default_eval_method
    ):
        PyEnvironment.__init__(self)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=_N_ACTIONS - 1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(GRID_SIZE, GRID_SIZE, 1),
            dtype=np.float32,
            minimum=0,
            name="observation",
        )
        self._evaluate = default_eval_method
        self.render_size = 320

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    @property
    def observation(self):
        return log2_on_game(np.array(self._state))

    def _reset(self):
        self._state = game.new_game(GRID_SIZE)
        self._prev_eval = self._evaluate(np.array(self._state))
        self._episode_ended = False
        self._false_input_cnt = 0
        self._latest_action = -1

        return ts.restart(self.observation)

    def _step(self, action):
        # 이전 단계에서 에피소드가 끝났다면, 이번에 평가된 액션을 무시하고 게임을 리셋합니다.
        if self._episode_ended:
            return self.reset()

        self._latest_action = action
        self._state, successful = _ACTION_SPACE[action](self._state)
        state = np.array(self._state)

        # 이동에 성공한 경우에만 새로운 블록을 추가합니다.
        if successful:
            self._state = game.add_two(self._state)

            # 누적 보상을 먼저 평가하고, 이전 누적 보상으로부터 델타-보상을 계산합니다.
            eval = self._evaluate(state)
            reward = eval - self._prev_eval
            self._prev_eval = eval

            # 실패 횟수 초기화
            self._false_input_cnt = 0
        else:
            # 이동에 반복적으로 실패하면 (즉, 갇힌 경우)에피소드를 끝냅니다.
            reward = 0
            self._false_input_cnt += 1

            if self._false_input_cnt > 50:
                self._episode_ended = True
                return ts.termination(self.observation, reward)

        # 더는 움직일 수 있는 블록이 없다면 게임 오버입니다.
        done = game.game_state(self._state) != "not over"
        if done:
            self._episode_ended = True
            return ts.termination(self.observation, reward)
        else:
            return ts.transition(self.observation, reward)

    def render(self, mode="human"):
        if mode == "human":
            render_size = self.render_size
            cell_size = int(self.render_size / GRID_SIZE)
            half = cell_size / 2
            output = np.ndarray((render_size, render_size, 3), np.uint8)
            output.fill(255)

            state = self._state

            FONTFACE = cv2.FONT_HERSHEY_PLAIN
            FONT_SCALE = 1
            FONT_THICK = 1

            for y, x in np.ndindex((GRID_SIZE, GRID_SIZE)):
                rp0 = (cell_size * (x + 0), cell_size * (y + 0))
                rp1 = (cell_size * (x + 1), cell_size * (y + 1))

                cv2.rectangle(output, rp0, tuple(np.subtract(rp1, (1, 1))), (0, 0, 0), 1)
                extent, _ = cv2.getTextSize(
                    str(state[y][x]), FONTFACE, FONT_SCALE, FONT_THICK
                )
                extent = (extent[0], -extent[1])
                org = (np.array(rp0) + np.array(rp1) - np.array(extent)) * 0.5

                cv2.putText(
                    output,
                    str(state[y][x]),
                    org=tuple(int(i) for i in org),
                    fontFace=FONTFACE,
                    fontScale=FONT_SCALE,
                    color=(0, 0, 0),
                    thickness=FONT_THICK,
                    bottomLeftOrigin=False,
                )

            if self._latest_action >= 0:
                p: np.ndarray = np.array([[0, -half], [half / 4, 0], [-half / 4, 0]])
                center = np.array([render_size / 2, render_size / 2])
                angle = self._latest_action * np.pi * 0.5
                sin, cos = np.sin(angle), np.cos(angle)
                rotmat = np.array([[cos, -sin], [sin, cos]])
                poly = [center + rotmat @ np.transpose(rvec) for rvec in p]
                color = (0, 255, 0) if self._false_input_cnt == 0 else (255, 0, 0)
                cv2.drawContours(output, np.array([poly]).astype(int), -1, color, -1)

            return output

        raise Exception("unsupported mode")


# %% Validate
from tf_agents.environments.utils import validate_py_environment
from IPython.display import Image

env = Game2048Env()
validate_py_environment(Game2048Env(), 5)

# %%
