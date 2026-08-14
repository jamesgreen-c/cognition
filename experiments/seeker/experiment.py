import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import argparse

import numpy as np
import jax.random as jr

from rp_slac.training import RPSLAC

from experiments.seeker.data import CentreSeekingEnvironment
from experiments.seeker.setup import setup


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


# SETUP
ENV = CentreSeekingEnvironment(T=args.T)
CONFIG, MODEL_FE, CONTROL_FE = setup(
    sequence_length=args.T,
    latent_dim=args.D,
    env_dim=1,
    batch_size=args.batch_size, 
    num_buffers=args.N,
    pretrain_iter=args.pretrain_iter,
    num_iter=args.num_iter,
    gamma=args.gamma,
    seed=args.seed, 
    stabilise_A=args.stabilise
)

# create results directory
EXPERIMENT_NAME = f"D={args.D},N={args.N},T={args.T},iter={args.num_iter},stabilise={args.stabilise},seed={args.seed}"
DIRPATH = f"results/{EXPERIMENT_NAME}"
if not os.path.exists(DIRPATH): os.makedirs(DIRPATH, exist_ok=True)

def main():

    trainer = RPSLAC(
        model=MODEL_FE, 
        control=CONTROL_FE, 
        environment=ENV, 
        config=CONFIG
    )
    _, replay_buffer = trainer.fit(use_pbar=True)

    # create results directory

    # save params
    with open(f"{DIRPATH}/params.pkl", "wb") as f: 
        pickle.dump(trainer.params, f)

    # save losses
    with open(f"{DIRPATH}/loss.pkl", "wb") as f: 
        losses = {
            "model": trainer.model_losses, 
            "critic": trainer.critic_losses, 
            "actor": trainer.actor_losses,
            "alpha": trainer.alpha_losses
        }
        pickle.dump(losses, f)

    # save replay buffer
    with open(f"{DIRPATH}/buffer.pkl", "wb") as f: 
        pickle.dump(replay_buffer, f)

    # save average rewards
    with open(f"{DIRPATH}/rewards.pkl", "wb") as f:
        pickle.dump(trainer.average_rewards, f)

    # save log alphas
    with open(f"{DIRPATH}/log_alphas.pkl", "wb") as f:
        pickle.dump(trainer.alpha_hist, f)

    # save actor stats history
    actor_stats = trainer.actor_stats
    means = np.array([_s["mean"] for _s in trainer.actor_stats])
    stds = np.array([_s["std"] for _s in trainer.actor_stats])
    actor_stats = {"mean": means, "std": stds}
    with open(f"{DIRPATH}/actor_stats.pkl", "wb") as f:
        pickle.dump(actor_stats, f)

    return trainer


def get_action_grid(trainer: RPSLAC):
    key = jr.PRNGKey(args.seed + 10)

    # make observation grid
    grid = ENV.grid(num_positions=101, num_velocities=101)
    grid_shape = grid.shape[:-1]
    observations = grid.reshape((-1, grid.shape[-1]))

    # get action grid
    actions = trainer.apply(key, trainer.params, observations, deterministic=True)
    actions = actions.squeeze(-1).reshape(grid_shape)

    # save action_grid
    action_grid = {
        "positions": np.asarray(grid[..., 0]),
        "velocities": np.asarray(grid[..., 1]),
        "actions": np.asarray(actions),
    }

    with open(f"{DIRPATH}/action_grid.pkl", "wb") as f:
        pickle.dump(action_grid, f)

    return action_grid

if __name__ == "__main__":
    trainer = main()
    get_action_grid(trainer)
