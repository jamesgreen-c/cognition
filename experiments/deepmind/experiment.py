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
parser.add_argument("--continue", dest="continue_run", action="store_true")
parser.add_argument("--verbose-warp", dest="verbose_warp", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()

os.environ["JAX_PLATFORMS"] = args.platform
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

if not args.verbose_warp:
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING

import pickle
import tempfile
from pathlib import Path

import cloudpickle
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
    f"history={args.actor_history},repeat={args.action_repeat},pretrain={args.pretrain_iter},"
    f"rgb={args.rgb},pixels={args.pixels},stabilise={args.stabilise},seed={args.seed}"
)
DIRPATH = f"results/{EXPERIMENT_NAME}"


def find_checkpoint():
    """Find the checkpoint for this experiment's stable directory name."""
    path = Path(DIRPATH)
    required = ("params.pkl", "opt_states.pkl", "opts.pkl",
                "env_states.pkl", "key.pkl", "loss.pkl", "buffer.pkl",
                "rewards.pkl", "log_alphas.pkl", "actor_stats.pkl")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files in {path}: {', '.join(missing)}")

    requested_config = vars(args).copy()
    for ignored in ("num_iter", "continue_run", "eval_episodes", "debug", "platform", "verbose_warp"):
        requested_config.pop(ignored)

    
    metadata_path = path / "run_config.pkl"
    if metadata_path.is_file():
        with metadata_path.open("rb") as f:
            if pickle.load(f) != requested_config:
                raise ValueError(f"Checkpoint settings do not match this experiment: {path}")
    return path


def load_checkpoint(trainer, path):

    def read(name):
        with (path / name).open("rb") as f:
            return pickle.load(f)

    params = read("params.pkl")
    replay_buffer = read("buffer.pkl")
    opt_states = read("opt_states.pkl")
    opts = read("opts.pkl")
    env_states = read("env_states.pkl")
    key = read("key.pkl")

    losses = read("loss.pkl")
    completed = len(losses["model"])
    trainer.model_losses = losses["model"]
    trainer.critic_losses = losses["critic"]
    trainer.actor_losses = losses["actor"]
    trainer.alpha_losses = losses["alpha"]

    trainer.average_rewards = read("rewards.pkl")
    trainer.alpha_hist = read("log_alphas.pkl")
    stats = read("actor_stats.pkl")
    trainer.actor_stats = [dict(mean=mean, std=std) for mean, std in zip(stats["mean"], stats["std"])]

    histories = (trainer.critic_losses, trainer.actor_losses, trainer.alpha_losses,
                 trainer.average_rewards, trainer.alpha_hist, trainer.actor_stats)
    
    if any(len(history) != completed for history in histories):
        raise ValueError(f"Inconsistent training history lengths in {path}.")
    
    if (len(replay_buffer["data"][0]) != args.N
            or len(replay_buffer["data"][1][0]) != args.capacity
            or replay_buffer["size"] < args.T):
        raise ValueError(f"Replay buffer dimensions do not match this experiment: {path}.")

    print(f"Loaded checkpoint from {path} ({completed} training steps).")
    return params, replay_buffer, env_states, opts, opt_states, key, completed

def main():

    trainer = RPSLAC(
        model=MODEL_FE, 
        control=CONTROL_FE, 
        environment=ENV, 
        config=CONFIG
    )
    if args.continue_run:
        checkpoint_path = find_checkpoint()
        params, replay_buffer, env_states, opts, opt_states, key, completed = load_checkpoint(trainer, checkpoint_path)

        print(f"Continuing {checkpoint_path} from step {completed} to {completed + args.num_iter}.")

        _, replay_buffer, env_states, key = trainer.train_continue(replay_buffer, 
                                                                   env_states, 
                                                                   params, 
                                                                   opts=opts, 
                                                                   opt_states=opt_states,
                                                                   start_itr=completed, 
                                                                   key=key, 
                                                                   use_pbar=True)
    else:
        if (Path(DIRPATH) / "params.pkl").exists():
            raise FileExistsError(f"An experiment already exists in {DIRPATH}; use --continue to extend it.")
        
        _, replay_buffer, env_states, key = trainer.fit(use_pbar=True)

    if not os.path.exists(DIRPATH):
        os.makedirs(DIRPATH, exist_ok=True)

    run_config = vars(args).copy()
    for ignored in ("num_iter", "continue_run", "eval_episodes", "debug", "platform", "verbose_warp"):
        run_config.pop(ignored)

    checkpoint = {
        "params.pkl": trainer.params,
        "opt_states.pkl": trainer.opt_states,
        "opts.pkl": trainer.opts,
        "env_states.pkl": env_states,
        "key.pkl": key,
        "run_config.pkl": run_config,
        "loss.pkl": {
            "model": trainer.model_losses,
            "critic": trainer.critic_losses,
            "actor": trainer.actor_losses,
            "alpha": trainer.alpha_losses,
        },
        "buffer.pkl": replay_buffer,
        "rewards.pkl": trainer.average_rewards,
        "log_alphas.pkl": trainer.alpha_hist,
        "actor_stats.pkl": {
            "mean": np.array([stat["mean"] for stat in trainer.actor_stats]),
            "std": np.array([stat["std"] for stat in trainer.actor_stats]),
        },
    }

    # finish every serialization before replacing files from an existing checkpoint.
    with tempfile.TemporaryDirectory(dir=DIRPATH) as staging:
        for name, value in checkpoint.items():
            with open(Path(staging) / name, "wb") as f:
                if name == "opts.pkl":
                    cloudpickle.dump(value, f)
                else:
                    pickle.dump(value, f)
        for name in checkpoint:
            os.replace(Path(staging) / name, Path(DIRPATH) / name)

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
