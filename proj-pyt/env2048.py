from typing import Callable
from game2048 import logic as game
from game2048.constants import GRID_LEN as GRID_SIZE
from gym import Env

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


class Game2048Env(Env):
    def __init__(
        self, evaluate: Callable[[np.ndarray], np.float] = default_eval_method
    ):
        super().__init__()
        self._evaluate = default_eval_method
        self.render_size = 320

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    @property
    def observation(self):
        return log2_on_game(np.array(self._state))

    def reset(self):
        self._state = game.new_game(GRID_SIZE)
        self._prev_eval = self._evaluate(np.array(self._state))
        self._false_input_cnt = 0
        self._latest_action = -1

        return self.observation

    def step(self, action):
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
            reward = -5
            self._false_input_cnt += 1

            if self._false_input_cnt > 50:
                return self.observation, reward, True, None

        # 더는 움직일 수 있는 블록이 없다면 게임 오버입니다.
        done = game.game_state(self._state) != "not over"
        if done:
            return self.observation, reward, True, None
        else:
            return self.observation, reward, False, None

    def render(self):
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
