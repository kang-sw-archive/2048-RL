# %% Test - Import
import os
import sys
from tempfile import mkdtemp, mkstemp
from numpy.testing._private.utils import tempdir, temppath

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tf_agents.policies as policies
import imageio
import IPython
import base64
import io
import env2048

from pygifsicle import optimize
from tf_agents.environments.tf_py_environment import TFPyEnvironment


env = env2048.Game2048Env()
tf_env = TFPyEnvironment(env)
policy = tf.compat.v2.saved_model.load("./models")


def embed_gif(gif_buffer):
    """Embeds a gif file in the notebook."""
    tag = '<img src="data:image/gif;base64,{0}"/>'.format(
        base64.b64encode(gif_buffer).decode()
    )
    return IPython.display.HTML(tag)


def run_episodes_and_create_video(policy, eval_tf_env, eval_py_env, path):
    num_episodes = 1
    frames = []
    fps = 16
    for episode in range(num_episodes):
        time_step = eval_tf_env.reset()
        frames.append(eval_py_env.render())

        step = 1
        while not time_step.is_last() and step < 1000:
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            frames.append(eval_py_env.render())
            print(f"\rperforming {step} step of Ep.{episode}", end="")
            step += 1

        # 몇 초 간 유예합니다
        for _ in range(fps * 3):
            frames.append(frames[-1])
    with open(path, "wb") as gif_file:
        imageio.mimsave(gif_file, frames, format="gif", fps=fps)

    optimize(path)

    with open(path, "rb") as gif_file:
        IPython.display.display(embed_gif(gif_file.read()))


# %% Save to file
run_episodes_and_create_video(policy, tf_env, env, mkdtemp() + '/temp.gif')

# %%
os.startfile("run-sample\\env_result.gif")

# %%
