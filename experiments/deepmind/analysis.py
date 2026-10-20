import os
import pickle
import argparse
import matplotlib.pyplot as plt 

import numpy as np
from PIL import Image

from experiments.deepmind.utils import experiment_name

# ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--kernel", choices=("ramp", "rpssm"), default="ramp")
parser.add_argument("--domain", type=str, default="cheetah")
parser.add_argument("--task", type=str, default="run")
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument("--actor-history", type=int, default=8)
parser.add_argument("--N", dest="N", type=int, default=1)
parser.add_argument("--D", dest="D", type=int, default=32)
parser.add_argument("--T", dest="T", type=int, default=8)
parser.add_argument("--pixels", type=int, default=64)
parser.add_argument("--rgb", action="store_true")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.99)
parser.add_argument("--pretrain-iter", dest="pretrain_iter", type=int, default=50000)
parser.add_argument("--model-batch-size", type=int, default=32)
parser.add_argument("--sac-batch-size", type=int, default=256)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--debug", dest="debug", action="store_true")
parser.add_argument("--no-debug", dest="debug", action="store_false")
parser.set_defaults(debug=False)

args = parser.parse_args()


# LOAD SAVED DATA, PARAMS and LOSS
EXPERIMENT_NAME = experiment_name(args)
DIRPATH = f"results/{EXPERIMENT_NAME}"

if not os.path.exists(DIRPATH):
    raise FileNotFoundError(f"No saved parameter results for experiment: {EXPERIMENT_NAME}")

with open(f"{DIRPATH}/params.pkl", "rb") as f:
    PARAMS = pickle.load(f)

with open(f"{DIRPATH}/loss.pkl", "rb") as f:
    LOSS = pickle.load(f)

NUM_ITER = len(LOSS["model"])

with open(f"{DIRPATH}/buffer.pkl", "rb") as f:
    BUFFER = pickle.load(f)

with open(f"{DIRPATH}/rewards.pkl", "rb") as f:
    REWARDS = pickle.load(f)

with open(f"{DIRPATH}/log_alphas.pkl", "rb") as f:
    LOG_ALPHA_HIST = pickle.load(f)

with open(f"{DIRPATH}/actor_stats.pkl", "rb") as f:
    ACTOR_STATS = pickle.load(f)

PLOTDIR = f"{DIRPATH}/plots"
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)


def plot_loss():
    model_losses = LOSS["model"]
    critic_losses = LOSS["critic"][5:]      # exclude potential extreme losses for random initialisations
    actor_losses = LOSS["actor"][5:]
    alpha_losses = LOSS["alpha"][5:]

    def _plot(_loss, _name, start=0):
        plt.figure(figsize=(15, 5))
        plt.plot(range(start, start + len(_loss)), _loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{_name} loss over training iterations")
        plt.tight_layout()
        plt.savefig(f"{PLOTDIR}/{_name}_loss.png")
        plt.close()

    _plot(model_losses, "model")
    _plot(critic_losses, "critic", 5)
    _plot(actor_losses, "actor", 5)
    _plot(alpha_losses, "alpha", 5)


def plot_buffer():
    _, actions, rewards, flags, log_probs = BUFFER["data"]

    def _plot(_series, _name):
        plt.figure(figsize=(35, 5))
        plt.plot(_series)
        plt.xlabel("Iteration")
        plt.ylabel(f"{_name}")
        plt.title(f"{_name} over training iterations")
        plt.tight_layout()
        plt.savefig(f"{PLOTDIR}/{_name}.png", dpi=200)
        plt.close()

    _plot(actions[0, :NUM_ITER], "actions")
    _plot(rewards[0, :NUM_ITER], "rewards")
    print("rewards shape: ", rewards.shape)
    # print(rewards[0, :NUM_ITER])

    
def plot_replay_images(steps=100, buffer_index=0, fps=25):
    """Save start/middle/end GIFs from one replay buffer in time order.

    Each clip contains up to `steps` real transitions (and their endpoint
    frames). Sparse terminal frames are included when the buffer stores them.
    Start/end refer to the oldest/newest retained data, not all training data.
    """
    observations = BUFFER["data"][0]
    size = int(BUFFER["size"])
    if size == 0:
        print("Replay buffer is empty; skipping image previews.")
        return
    if observations.ndim not in (4, 5) or (observations.ndim == 5 and observations.shape[-1] not in (1, 3, 4)):
        print("Replay observations are not image frames; skipping image previews.")
        return
    if steps < 1 or fps <= 0:
        raise ValueError("steps and fps must be positive.")
    if not 0 <= buffer_index < observations.shape[0]:
        raise ValueError("buffer_index is outside the replay buffer batch.")

    length = min(steps, size)
    clips = (("start", 0), ("middle", (size - length) // 2), ("end", size - length))
    if size <= steps:
        clips = (("end", 0),)
    output_dir = os.path.join(PLOTDIR, "replay_images")
    os.makedirs(output_dir, exist_ok=True)
    capacity = BUFFER["data"][1].shape[1]
    transition_start = (int(BUFFER["write"]) - size) % capacity
    obs_start = int(BUFFER["obs_start"])
    terminal_stores = BUFFER.get("terminal_observations")
    terminal_frames = {} if terminal_stores is None else terminal_stores[buffer_index]

    def image(frame):
        frame = np.asarray(frame)
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.rint(np.clip(frame, 0.0, 1.0) * 255.0)
        frame = frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        return Image.fromarray(frame)

    def observation_at(index):
        slot = (obs_start + index) % observations.shape[1]
        return observations[buffer_index, slot]

    for name, start in clips:
        frames = [image(observation_at(start))]
        for index in range(start, start + length):
            slot = (transition_start + index) % capacity
            if slot in terminal_frames:
                frames.append(image(terminal_frames[slot]))
            frames.append(image(observation_at(index + 1)))
        path = os.path.join(output_dir, f"buffer_{buffer_index}_{name}.gif")
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=max(10, round(1000 / fps)), loop=0, disposal=2)
        print(f"Saved {path}: retained steps {start}:{start + length}, {len(frames)} frames at {fps:g} fps.")


def plot_rpm_params():
    """
    Plot:
    1. Loss over time
    2. Learned A matrix
    """

    def _plot(mat, _name):
        plt.figure()
        plt.imshow(mat)
        plt.title(f"Learned {_name} on representation")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{PLOTDIR}/{_name}.png")
        plt.close()

    A = PARAMS["prior"]["A"]  # (D, D)
    B = PARAMS["prior"]["B"]  # (D, K)

    _plot(A, "A")
    _plot(B, "B")


def plot_actor_stats():

    means = ACTOR_STATS["mean"]
    stds = ACTOR_STATS["std"]

    def _plot(_series, _name):
        plt.figure(figsize=(15, 5))
        plt.plot(_series)
        plt.xlabel("Iteration")
        plt.ylabel(f"{_name}")
        plt.title(f"{_name} over training iterations")
        plt.tight_layout()
        plt.savefig(f"{PLOTDIR}/{_name}.png")
        plt.close()

    _plot(means.mean(axis=1), "actor_mean")   # (T, B, 1) -> (T, 1)
    _plot(stds.mean(axis=1), "actor_std")


def plot_training_stats():

    average_rewards = REWARDS
    log_alpha_hist = LOG_ALPHA_HIST 

    def _plot(_series, _name):
        plt.figure(figsize=(15, 5))
        plt.plot(_series)
        plt.xlabel("Iteration")
        plt.ylabel(f"{_name}")
        plt.title(f"{_name} over training iterations")
        plt.tight_layout()
        plt.savefig(f"{PLOTDIR}/{_name}.png")
        plt.close()

    _plot(average_rewards, "average_reward")
    _plot(log_alpha_hist, "log_alphas")

if __name__ == "__main__":

    print(f"Loaded {NUM_ITER} training iterations from {DIRPATH}.")
    print(f"Log alpha: {PARAMS['log_alpha']}")

    plot_training_stats()
    plot_buffer()
    plot_replay_images()
    plot_loss()
    # plot_rpm_params()
    plot_actor_stats()
