import os
import argparse

# ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default="cheetah")
parser.add_argument("--task", type=str, default="run")
parser.add_argument("--action-repeat", type=int, default=1)
parser.add_argument("--actor-history", type=int, default=3)
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--D", type=int, default=24)
parser.add_argument("--T", type=int, default=8)
parser.add_argument("--pixels", type=int, default=64)
parser.add_argument("--rgb", action="store_true")
parser.add_argument("--stabilise", type=str, default="clip")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--pretrain-iter", type=int, default=3000)
parser.add_argument("--num-iter", type=int, default=2500)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--capacity", type=int, default=2500)
parser.add_argument("--eval-episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()

os.environ["JAX_PLATFORMS"] = args.platform
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle

import jax
import numpy as np
import jax.random as jr

from rp_slac.training import RPSLAC

from experiments.deepmind.data import DmControlEnvironment
from experiments.deepmind.setup import setup

# SETUP
ENV = DmControlEnvironment(
    domain_name=args.domain,
    task_name=args.task,
    num_buffers=args.N,
    action_repeat=args.action_repeat,
    actor_history=args.actor_history,
    width=args.pixels,
    height=args.pixels,
    rgb=args.rgb
)

CONFIG, MODEL_FE, CONTROL_FE = setup(
    sequence_length=args.T,
    latent_dim=args.D,
    actor_history=args.actor_history,
    action_dim=ENV.action_lower.shape[0],
    action_low=ENV.action_lower,
    action_high=ENV.action_upper,
    batch_size=args.batch_size,
    pixels=args.pixels,
    num_buffers=args.N,
    pretrain_iter=args.pretrain_iter,
    num_iter=args.num_iter,
    capacity=args.capacity,
    gamma=args.gamma,
    seed=args.seed,
    stabilise_A=args.stabilise,
)


# create results directory
EXPERIMENT_NAME = (
    f"domain={args.domain},task={args.task},D={args.D},N={args.N},T={args.T},"
    f"iter={args.num_iter},history={args.actor_history},repeat={args.action_repeat},"
    f"rgb={args.rgb},pixels={args.pixels},stabilise={args.stabilise},seed={args.seed}"
)
DIRPATH = f"results/{EXPERIMENT_NAME}"

def main():

    trainer = RPSLAC(
        model=MODEL_FE, 
        control=CONTROL_FE, 
        environment=ENV, 
        config=CONFIG
    )
    _, replay_buffer = trainer.fit(use_pbar=True)

    if not os.path.exists(DIRPATH):
        os.makedirs(DIRPATH, exist_ok=True)

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
    if args.eval_episodes < 1:
        raise ValueError("eval_episodes must be positive.")

    key = jr.PRNGKey(args.seed + 10)
    key, initial_key = jr.split(key)

    # use the deterministic mean policy
    @jax.jit
    def mean_policy_step(actor_params, actor_observation):
        actor_observation = ENV.preprocess_observation(actor_observation)
        _mean_policy = lambda observation: trainer.control.mean_policy(actor_params, observation)
        return jax.vmap(_mean_policy)(actor_observation)

    # compile step sampling steps    
    initial_carry = jax.jit(lambda init_key: ENV.initial_carry(init_key, args.N))
    collect_step = jax.jit(ENV.collect_step)

    # run evaluation
    carry = initial_carry(initial_key)
    running_returns = np.zeros(args.N, dtype=np.float64)
    completed_returns = []

    while len(completed_returns) < args.eval_episodes:
        key, env_key = jr.split(key)
        actor_observation = ENV.actor_observation(carry)
        actions = mean_policy_step(trainer.params["actor"], actor_observation)
        carry, _, rewards, flags = collect_step(env_key, carry, actions)

        running_returns += np.asarray(rewards)
        episode_ends = np.asarray(flags) == 0.0
        completed_returns.extend(running_returns[episode_ends].tolist())
        running_returns[episode_ends] = 0.0

    # store returns and save to disc
    episode_returns = np.asarray(completed_returns[:args.eval_episodes], dtype=np.float64)
    results = {
        "episode_returns": episode_returns,
        "mean_return": float(episode_returns.mean()),
        "std_return": float(episode_returns.std()),
    }

    with open(f"{DIRPATH}/evaluation.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Average return over {} episodes: {:.2f} ± {:.2f}".format(args.eval_episodes,
                                                                    results["mean_return"],
                                                                    results["std_return"]))

    return results

if __name__ == "__main__":
    trainer = main()
    # evaluate_policy(trainer)

