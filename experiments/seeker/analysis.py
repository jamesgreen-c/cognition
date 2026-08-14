import os
import pickle
import argparse
import matplotlib.pyplot as plt 

import numpy as np


# ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--N", dest="N", type=int, default=1)
parser.add_argument("--D", dest="D", type=int, default=10)
parser.add_argument("--T", dest="T", type=int, default=50)

parser.add_argument("--stabilise", dest="stabilise", default="clip")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.99)

parser.add_argument("--pretrain-iter", dest="pretrain_iter", type=int, default=3000)
parser.add_argument("--num-iter", dest="num_iter", type=int, default=2300)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)

parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--debug", dest="debug", action="store_true")
parser.add_argument("--no-debug", dest="debug", action="store_false")
parser.set_defaults(debug=False)

args = parser.parse_args()


# LOAD SAVED DATA, PARAMS and LOSS
EXPERIMENT_NAME = f"D={args.D},N={args.N},T={args.T},iter={args.num_iter},stabilise={args.stabilise},seed={args.seed}"
DIRPATH = f"results/{EXPERIMENT_NAME}"

if not os.path.exists(DIRPATH):
    raise FileNotFoundError(f"No saved parameter results for experiment: {EXPERIMENT_NAME}")

with open(f"{DIRPATH}/params.pkl", "rb") as f:
    PARAMS = pickle.load(f)

with open(f"{DIRPATH}/loss.pkl", "rb") as f:
    LOSS = pickle.load(f)

with open(f"{DIRPATH}/buffer.pkl", "rb") as f:
    BUFFER = pickle.load(f)

with open(f"{DIRPATH}/rewards.pkl", "rb") as f:
    REWARDS = pickle.load(f)

with open(f"{DIRPATH}/log_alphas.pkl", "rb") as f:
    LOG_ALPHA_HIST = pickle.load(f)

with open(f"{DIRPATH}/actor_stats.pkl", "rb") as f:
    ACTOR_STATS = pickle.load(f)

with open(f"{DIRPATH}/action_grid.pkl", "rb") as f:
    ACTION_GRID = pickle.load(f)

PLOTDIR = f"{DIRPATH}/plots"
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)


def plot_loss():
    model_losses = LOSS["model"]
    critic_losses = LOSS["critic"]
    actor_losses = LOSS["actor"]
    alpha_losses = LOSS["alpha"]

    def _plot(_loss, _name):
        plt.figure(figsize=(15, 5))
        plt.plot(_loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{_name} loss over training iterations")
        plt.tight_layout()
        plt.savefig(f"{PLOTDIR}/{_name}_loss.png")
        plt.close()

    _plot(model_losses, "model")
    _plot(critic_losses, "critic")
    _plot(actor_losses, "actor")
    _plot(alpha_losses, "alpha")


def plot_buffer():
    obs, actions, rewards, flags, log_probs = BUFFER

    def _plot(_series, _name):
        plt.figure(figsize=(15, 5))
        plt.plot(_series)
        plt.xlabel("Iteration")
        plt.ylabel(f"{_name}")
        plt.title(f"{_name} over training iterations")
        plt.tight_layout()
        plt.savefig(f"{PLOTDIR}/{_name}.png")
        plt.close()

    _plot(actions[0], "actions")
    _plot(rewards[0], "rewards")

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    # pos
    ax[0].plot(obs[0, :, 0], label="position")
    ax[0].set_title("Position over training iterations")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Position")

    # vels
    ax[1].plot(obs[0, :, 1], label="velocity")
    ax[1].set_title("Velocity over training iterations")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Velocity")

    plt.tight_layout()
    plt.savefig(f"{PLOTDIR}/observations.png")
    plt.close()
    
    
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


def plot_action_grid():
    positions = ACTION_GRID["positions"]
    velocities = ACTION_GRID["velocities"]
    actions = ACTION_GRID["actions"]

    plt.figure(figsize=(10, 7))

    contour = plt.contourf(
        positions,
        velocities,
        actions,
        levels=50,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    plt.colorbar(contour, label="Action")
    plt.contour(
        positions,
        velocities,
        actions,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )

    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Actor action surface")
    plt.tight_layout()
    plt.savefig(f"{PLOTDIR}/action_surface.png")
    plt.close()


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

    print(f"Log alpha: {PARAMS['log_alpha']}")

    plot_training_stats()
    plot_buffer()
    plot_loss()
    plot_rpm_params()
    plot_actor_stats()
    plot_action_grid()
