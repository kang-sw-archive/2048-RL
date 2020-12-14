# %% Imports
from multiprocessing import sharedctypes
from typing import Any, Callable, Mapping, Optional, Tuple

# %%
import train as tr
GRID_SIZE = tr.GRID_SIZE
N_ACTIONS = tr.NUM_ACTIONS()

# %% Define Consumer
import threading as thr
import numpy as np
keras = tr.keras


class _SharedTrainData:
    def __init__(self) -> None:
        super().__init__()
        self.should_dispose = thr.Event()

        self.fence = -1
        self.weights = []

        self.train_time = 10.

        self.n_batch = 32
        self.epsilon = 1

        self.__n_working = 0
        self.__lock = thr.Lock()
        self.__cond = thr.Condition()

    @property
    def num_working_workers(s):
        with s.__lock:
            return s.__n_working

    def lock_worker(s):
        with s.__lock:
            s.__n_working += 1

    def release_worker(s):
        with s.__lock:
            s.__n_working -= 1

    def wait(self, pred, timeout: float = 1.e10):
        with self.__cond:
            self.__cond.wait_for(pred, timeout)

    def notify_all(self):
        with self.__cond:
            self.__cond.notify_all()


class _TrainTaskConsumer(thr.Thread):
    """
    하나의 프로세스를 점유하고, 비동기적으로 훈련을 실행하는 워커 클래스입니다.
    다음의 과정을 통해 이루어집니다.

        0. 워커 초기화 및 모델 클론
        1. 워커 루프
            1. [Main] 이전 iteration의 best weight를 shared state에 upload
            2. [Worker] Shared state에서 best weight download 및 process local model에 적용
            3. [Worker] 지정된 시간만큼 로컬에서 학습 iteration, 최종 모델 및 리워드 result queue에 push
            4. [Main] 각 worker가 반환한 최종 model에서 best candidate를 선택
    """

    def __init__(
            self,
            model_prototype: keras.Model,
            shared_data: _SharedTrainData,
            optimizer,
            loss_fn,
            discount_factor=1,
            max_replay=40) -> None:
        super().__init__()
        self.__local = thr.local()
        self.__shared = shared_data
        self.dispose_event = thr.Event()
        self.__fence = -1
        self.__model = keras.models.clone_model(model_prototype)

        self.__trainer_factory = lambda: tr.Trainer(self.__model, optimizer, loss_fn, discount_factor, max_replay)

    def run(self) -> None:
        local = self.__local
        local.trainer = self.__trainer_factory()

        while not self.__shared.should_dispose.is_set():
            self.__wait_any()

            # 새로운 best weight 입력을 대기합니다.
            # 값의 준비 여부는 fence로 판별합니다.
            # 만약 notify에서 깨어났을 때 입력이 준비되지 않았다면,
            # notify timeout이거나 dispose 시 아래 condition은 false
            if self.__fence >= self.__shared.fence:
                continue

            # 훈련 시작시 shared lock을 increase합니다.
            self.__shared.lock_worker()
            duration = self.__shared.train_time

            # Model weight update 다운로드
            trainer: tr.Trainer = local.trainer
            trainer.refresh_weights(self.__shared.weights)
            trainer.update_target_weights()

            # 지정된 시간동안 최대한 많은 iteration 수행
            import time
            until = time.process_time() + duration
            n_batch = self.__shared.n_batch
            epsilon = self.__shared.epsilon
            rewards = []

            while time.process_time() < until:
                rewards.append(trainer.train_single_episode(n_batch, epsilon=epsilon))

            # target model에 weight update를 반영하고, reward 평균을 기록
            # Manager는 reward 평균이 최대인 훈련을 채택하게 됩니다.
            trainer.update_target_weights()
            n_episodes = len(rewards)
            latest = rewards[n_episodes - 1]
            rewards: np.ndarray = np.array(rewards)
            mean = np.sum(rewards) / rewards.shape[0]
            max = rewards.max()
            min = rewards.min()
            std = rewards.std()
            self.__reward = latest, mean, max, min, std, n_episodes

            # shared lock은 release
            self.__fence = self.__shared.fence
            self.__weight = trainer.retrieve_weight()
            self.__shared.release_worker()

    def __wait_any(s):
        s.__shared.wait(lambda: True, 10)

    @property
    def reward(self):
        return self.__reward
        # TODO: Returns latest, mean, max, min, std

    @property
    def weight(self):
        return self.__weight


# %% Define worker system
from multiprocessing import cpu_count


class TrainManager(object):
    def __init__(self, model_prototype: keras.Model,
                 optimizer,
                 loss_fn,
                 discount_factor=1,
                 max_replay=40,
                 n_thrd=0) -> None:
        super().__init__()
        if n_thrd <= 0:
            n_thrd = cpu_count()

        self.__best_model = model_prototype
        self.__shared = _SharedTrainData()
        self.__workers = [_TrainTaskConsumer(model_prototype, self.__shared, optimizer,
                                             loss_fn, discount_factor, max_replay)
                          for _ in range(n_thrd)]
        self.__disposed = False
        self.__launched = False

    def launch(self):
        if self.__launched == False:
            self.__launched = True
            for worker in self.__workers:
                worker.start()
        else:
            raise Exception("TrainManager already launched")

    def __enter__(self):
        self.launch()

    def access_trainer(self, callback):
        for worker in self.__workers:
            callback(worker)

    def dispose(self):
        if self.__launched and not self.__disposed:
            self.__disposed = True
            self.__shared.should_dispose.set()
            self.__shared.fence = -1
            self.__shared.notify_all()

            for worker in self.__workers:
                worker.join()

            self.__shared.should_dispose.clear()
            self.__disposed = False
            self.__launched = False

    def __exit__(self, a, b, c):
        self.dispose()

    def __del__(self):
        self.dispose()

    def step_train_sequence(self, n_batch=32, duration=10., epsilon=1.):
        """
        한 번의 훈련 과정을 수행합니다.
        """
        import time

        # 진입 시점에서 항상 모든 worker는 idle 상태
        self.__shared.weights = self.__best_model.get_weights()
        self.__shared.epsilon = epsilon
        self.__shared.n_batch = n_batch
        self.__shared.train_time = duration

        # 작업을 트리거합니다
        self.__shared.fence += 1
        self.__shared.notify_all()
        time.sleep(duration)

        # 모든 스레드의 처리 완료를 대기합니다.
        # polling
        while self.__shared.num_working_workers > 0:
            time.sleep(0.1)

        # best model을 다운로드하고, 모델에 반영합니다.
        reward_tuples = [worker.reward for worker in self.__workers]
        print(reward_tuples)
        latests, means, maxs, mins, stds, n_episodes = [np.array(tup) for tup in zip(*reward_tuples)]
        latests: np.ndarray

        best_idx = np.argmax(latests)
        self.__best_model.set_weights(self.__workers[best_idx].weight)
        return latests, means, maxs, mins, stds, n_episodes

    @property
    def model(self):
        return self.__best_model


# %%
trainer = TrainManager(
    keras.models.Sequential([
        keras.layers.Dense(256, activation='elu', input_shape=[GRID_SIZE * GRID_SIZE]),
        keras.layers.Dense(256, activation='elu'),
        keras.layers.Dense(256, activation='elu'),
        keras.layers.Dense(256, activation='elu'),
        keras.layers.Dense(256, activation='elu'),
        keras.layers.Dense(N_ACTIONS)]),
    keras.optimizers.Adam(learning_rate=0.01),
    keras.losses.mean_squared_error,
    max_replay=200,
    n_thrd=1)

# %% Iterate train

with trainer:
    for i in range(10):
        latests, means, maxs, mins, stds, n_episodes = trainer.step_train_sequence(duration=20, epsilon=0.5)
        model = trainer.model
        print(f'{n_episodes=}')


# %%
