import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import argparse

import numpy as np
import jax.random as jr

from rp_slac.training import RPSLAC

from experiments.deepmind.data import DmControlEnvironment
from experiments.deepmind.setup import setup


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--domain", type=str, default="cheetah")
parser.add_argument("--task", type=str, default="run")
parser.add_argument("--action-repeat", type=int, default=1)

parser.add_argument("--N", type=int, default=4)
parser.add_argument("--D", type=int, default=32)
parser.add_argument("--T", type=int, default=8)

parser.add_argument("--stabilise", type=str, default="clip")
parser.add_argument("--gamma", type=float, default=0.99)

parser.add_argument("--pretrain-iter", type=int, default=3000)
parser.add_argument("--num-iter", type=int, default=250000)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--capacity", type=int, default=5000)
parser.add_argument("--collection-steps", type=int, default=1)

parser.add_argument("--eval-episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

# SETUP
ENV = DmControlEnvironment(
    domain_name="cheetah",
    task_name="run",
    num_buffers=args.num_buffers,
    action_repeat=1,
    width=64,
    height=64,
    camera_id=0,
)

CONFIG, MODEL_FE, CONTROL_FE = setup(
    sequence_length=args.T,
    latent_dim=args.D,
    action_dim=ENV.action_lower.shape[0],
    action_low=ENV.action_lower,
    action_high=ENV.action_upper,
    batch_size=args.batch_size,
    num_buffers=args.N,
    pretrain_iter=args.pretrain_iter,
    num_iter=args.num_iter,
    collection_steps=args.collection_steps,
    capacity=args.capacity,
    gamma=args.gamma,
    seed=args.seed,
    stabilise_A=args.stabilise,
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


def evaluate_policy(trainer: RPSLAC):
    key = jr.PRNGKey(args.seed + 10)

    def policy(policy_key, observation):
        return trainer.control.policy(
            policy_key,
            trainer.params,
            observation,
        )

    results = ENV.evaluate(
        key=key,
        policy=policy,
        num_episodes=args.eval_episodes,
    )

    with open(f"{DIRPATH}/evaluation.pkl", "wb") as f:
        pickle.dump(results, f)

    print(
        "Average return over {} episodes: {:.2f} ± {:.2f}".format(
            args.eval_episodes,
            results["mean_return"],
            results["std_return"],
        )
    )

    return results

if __name__ == "__main__":
    trainer = main()
    evaluate_policy(trainer)
