# %% Preset
import os

pathstr = ";C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin;"
os.environ["PATH"] += pathstr

# %% Imports
import tensorflow
import env2048
import tensorflow as tf
import numpy as np

from tensorflow.python.training.tracking.util import Checkpoint
from tf_agents import policies, replay_buffers, trajectories
from tf_agents.agents import tf_agent
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tensorflow import keras
from multiprocessing import cpu_count

# %% constants
NUM_TOTAL_STEPS = int(1e6)

UPDATE_PERIOD = 8
TARGET_UPDATE_PERIOD = 5000

REPLAY_BUFFER_MAXLEN = 1000000

ITER_NEWLINE_PERIOD = 250
ITER_CHECKPOINT_PERIOD = 1000
ITER_LOG_PERIOD = 250

SAMPLE_BATCH_SIZE = 32
PARALLEL_STEPS = cpu_count()

CHECKPOINT_DIR = "./checkpoint"
POLICY_SAVE_DIR = "./models"

# %% Resolve CuDNN error
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %% Create Q network
# WARNING: 이 프로젝트는 EncodingNetwork의 생성자의 기본 padding을 valid에서 same으로 바꾸는 하드코딩을 포함하고 있습니다.
#          따라서 다시 빌드 시 적용이 안 될 가능성이 높습니다.  유의!
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.sequential import Sequential
from tf_agents.environments.tf_py_environment import TFPyEnvironment


def max_hype_evaluator(state: np.ndarray):
    """
    현재 게임 스테이트를 평가합니다.
    가급적 높은 숫자의 셀을 우선적으로 병합하도록, 최대 크기의 셀에 가중치를 부여합니다.
    또한, 가급적 공간을 확보하는 방향으로 진행하도록, 합친 셀에 더 큰 가중치를 부여합니다.
    """

    s = state.ravel()

    # 더 큰 블록에 보너스를 부여합니다. 0의 보너스가 음수가 되지 않게 조정합니다.
    # slog = np.log2(s + (s == 0) * 2)
    # bonus = 2 ** (slog - 1) - 1
    # return 2 * np.log2(np.max(s)) + np.sum(s + bonus)
    return 2 * np.log2(np.max(s)) + np.sum(s)


env = env2048.Game2048Env(max_hype_evaluator)
tf_env = TFPyEnvironment(env)

preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32))
conv_layer_params = [(1024, (3, 3), 1), (512, (2, 2), 1)]
fc_layer_params = [512, 256]

from tensorflow.keras import layers


CONV_ACTIVATION = "relu"
FC_ACTIVATION = "relu"
q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
)


# %% Initialize DQN agent
from tf_agents.agents.dqn.dqn_agent import DqnAgent

train_step = tf.Variable(0)
update_period = UPDATE_PERIOD
optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0, epsilon=1e-5)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=int(250e3) // update_period,
    end_learning_rate=0.01,
)

agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=TARGET_UPDATE_PERIOD,
    td_errors_loss_fn=keras.losses.Huber(reduction="none"),
    gamma=0.99,
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step),
)

agent.initialize()


# %% Create rezplay buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=int(REPLAY_BUFFER_MAXLEN),
)


# %% Create observer
from tf_agents.trajectories.trajectory import Trajectory


class ProgressObserver:
    def __init__(
        self,
        total,
        store_path,
        log_period_steps=5000,
        store_period=100000,
    ):
        self.step_counter = 0
        self.episode_couinter = 0
        self.total = total

        self.store_path = store_path
        self.store_period = store_period
        self.log_period = log_period_steps

    def __call__(self, trajectory: Trajectory):
        if not trajectory.is_boundary():
            self.step_counter += 1
        else:
            self.episode_couinter += 1

        if self.step_counter % self.log_period == 0:
            print(
                f"...Step {self.step_counter:12} of Episode {self.episode_couinter+1:8}",
                end="\r",
            )


replay_buffer_observer = [replay_buffer.add_batch]

# %% Metrics
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging

training_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric()]

training_metrics_2 = [
    tf_metrics.MaxReturnMetric(),
    tf_metrics.MinReturnMetric()]

logging.getLogger().setLevel(logging.INFO)

# %% Driver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=replay_buffer_observer + training_metrics + training_metrics_2,
    num_steps=update_period,
)

from tf_agents.policies.random_tf_policy import RandomTFPolicy

# initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
# init_driver = DynamicStepDriver(
#     tf_env,
#     initial_collect_policy,
#     observers=replay_buffer_observer + training_metrics + training_metrics_2,
#     num_steps=update_period,
# )
# final_time_step, final_policty_state = init_driver.run()

# %% Dataset
dataset = replay_buffer.as_dataset(sample_batch_size=SAMPLE_BATCH_SIZE, num_steps=2, num_parallel_calls=PARALLEL_STEPS)

# %% Checkpoint
from tf_agents.utils.common import Checkpointer

train_checkpointer = Checkpointer(
    ckpt_dir=CHECKPOINT_DIR,
    max_to_keep=4,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step,
)

train_checkpointer.initialize_or_restore()

# %% Policy saver
tf_policy_saver = policies.policy_saver.PolicySaver(agent.policy)


def save_policy():
    tf_policy_saver.save(POLICY_SAVE_DIR)


# %% Iterate
from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

fp = open("log/log-{}.txt".format(agent.train_step_counter.value().numpy()), "w")
fp.write(", ".join(["Step"] + [type(f).__name__ for f in training_metrics]) + "\n")


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policty_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)

        print("\r{} loss:{:.5f}".format(
            agent.train_step_counter.value().numpy(), train_loss.loss.numpy()), end="")

        if iteration % ITER_LOG_PERIOD == 0:
            fp.write(", ".join(["{}".format(agent.train_step_counter.value().numpy())] + ["{}".format(m.result()) for m in training_metrics]) + "\n")

        if iteration % ITER_NEWLINE_PERIOD == 0:
            print()
            log_metrics(training_metrics + training_metrics_2)
            fp.flush()

        if iteration and iteration % ITER_CHECKPOINT_PERIOD == 0:
            train_checkpointer.save(train_step)
            # tf_policy_saver.save(POLICY_SAVE_DIR)
            print()


try:
    train_agent(10000000)
finally:
    fp.close()
    save_policy()
