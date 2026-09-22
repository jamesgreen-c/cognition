import argparse
import os


# ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default="cheetah")
parser.add_argument("--task", type=str, default="run")
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument("--actor-history", type=int, default=3)
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--D", type=int, default=24)
parser.add_argument("--T", type=int, default=8)
parser.add_argument("--pixels", type=int, default=64)
parser.add_argument("--rgb", action="store_true")
parser.add_argument("--stabilise", type=str, default="clip")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--pretrain-iter", type=int, default=3000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--capacity", type=int, default=2500)
parser.add_argument("--eval-episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--verbose-warp", dest="verbose_warp", action="store_true")
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

import jax
import jax.random as jr
import numpy as np

from rp_slac.training import RPSLAC

from experiments.deepmind.data import DmControlEnvironment
from experiments.deepmind.setup import setup


ENV = DmControlEnvironment(
    domain_name=args.domain,
    task_name=args.task,
    num_buffers=args.N,
    action_repeat=args.action_repeat,
    actor_history=args.actor_history,
    width=args.pixels,
    height=args.pixels,
    rgb=args.rgb,
)

# This only constructs the same trainer objects as experiment.py;
#  evaluation does not execute the training schedule.
CONFIG, MODEL_FE, CONTROL_FE = setup(
    sequence_length=args.T,
    initial_steps=10000,
    latent_dim=args.D,
    actor_history=args.actor_history,
    action_dim=ENV.action_lower.shape[0],
    action_low=ENV.action_lower,
    action_high=ENV.action_upper,
    batch_size=args.batch_size,
    pixels=args.pixels,
    num_buffers=args.N,
    pretrain_iter=args.pretrain_iter,
    num_iter=2500,
    capacity=args.capacity,
    gamma=args.gamma,
    seed=args.seed,
    stabilise_A=args.stabilise,
)

EXPERIMENT_NAME = (
    f"domain={args.domain},task={args.task},D={args.D},N={args.N},T={args.T},"
    f"history={args.actor_history},repeat={args.action_repeat},pretrain={args.pretrain_iter},"
    f"rgb={args.rgb},pixels={args.pixels},stabilise={args.stabilise},seed={args.seed}"
)
DIRPATH = Path("results") / EXPERIMENT_NAME


def find_checkpoint() -> Path:
    """Find and validate the checkpoint for the requested experiment."""
    required = ("params.pkl", "loss.pkl", "run_config.pkl")
    missing = [name for name in required if not (DIRPATH / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint files in {DIRPATH}: {', '.join(missing)}"
        )

    requested_config = vars(args).copy()
    for ignored in ("eval_episodes", "platform", "verbose_warp"):
        requested_config.pop(ignored)

    with (DIRPATH / "run_config.pkl").open("rb") as f:
        saved_config = pickle.load(f)
    if saved_config != requested_config:
        mismatches = {
            name: (saved_config.get(name), requested_config.get(name))
            for name in sorted(set(saved_config) | set(requested_config))
            if saved_config.get(name) != requested_config.get(name)
        }
        raise ValueError(
            f"Checkpoint settings do not match this experiment: {DIRPATH}. "
            f"Mismatches: {mismatches}"
        )
    return DIRPATH

def load_trainer(checkpoint_path: Path) -> tuple[RPSLAC, int]:
    """Reconstruct the trainer and load the parameters at the checkpoint."""
    trainer = RPSLAC(
        model=MODEL_FE,
        control=CONTROL_FE,
        environment=ENV,
        config=CONFIG,
    )

    with (checkpoint_path / "params.pkl").open("rb") as f:
        trainer.params = pickle.load(f)
    with (checkpoint_path / "loss.pkl").open("rb") as f:
        losses = pickle.load(f)

    completed = len(losses["model"])
    if any(len(losses[name]) != completed for name in ("critic", "actor", "alpha")):
        raise ValueError(f"Inconsistent training history lengths in {checkpoint_path}.")

    print(f"Loaded checkpoint from {checkpoint_path} ({completed} training steps).")
    return trainer, completed


def evaluate_policy(trainer: RPSLAC, completed: int) -> dict:
    if args.eval_episodes < 1:
        raise ValueError("eval_episodes must be positive.")

    key = jr.PRNGKey(args.seed + 10)
    key, initial_key = jr.split(key)

    @jax.jit
    def mean_policy_step(params, actor_observation):
        actor_observation = ENV.preprocess_observation(actor_observation)
        policy = lambda observation: trainer.control.mean_policy(params, 
                                                                 observation)
        return jax.vmap(policy)(actor_observation)

    initial_carry = jax.jit(lambda init_key: ENV.initial_carry(init_key, args.N))
    collect_step = jax.jit(ENV.collect_step)

    carry = initial_carry(initial_key)
    running_returns = np.zeros(args.N, dtype=np.float64)
    completed_returns = []

    while len(completed_returns) < args.eval_episodes:
        key, env_key = jr.split(key)
        actor_observation = ENV.actor_observation(carry)
        actions = mean_policy_step(trainer.params, actor_observation)
        carry, _, rewards, flags = collect_step(env_key, carry, actions)

        # undiscounted episode return.
        running_returns += np.asarray(rewards)
        episode_ends = np.asarray(flags) == 0.0
        completed_returns.extend(running_returns[episode_ends].tolist())
        running_returns[episode_ends] = 0.0

    episode_returns = np.asarray(completed_returns[: args.eval_episodes], dtype=np.float64)
    
    results = {
        "num_iter": completed,
        "eval_episodes": args.eval_episodes,
        "episode_returns": episode_returns,
        "mean_return": float(episode_returns.mean()),
        "std_return": float(episode_returns.std()),
        "policy": "deterministic_mean",
        "seed": args.seed,
        "action_repeat": args.action_repeat,
    }

    evaluation_dir = DIRPATH / f"evaluation,num-iter={completed}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Complete serialization before atomically replacing an earlier evaluation.
    with tempfile.NamedTemporaryFile(dir=evaluation_dir, delete=False) as f:
        temporary_path = Path(f.name)
        pickle.dump(results, f)
    os.replace(temporary_path, evaluation_dir / "evaluation.pkl")

    print(
        "Average return over {} episodes at iteration {}: {:.2f} +/- {:.2f}".format(
            args.eval_episodes,
            completed,
            results["mean_return"],
            results["std_return"],
        )
    )
    return results


def main():
    checkpoint_path = find_checkpoint()
    trainer, completed = load_trainer(checkpoint_path)
    return evaluate_policy(trainer, completed)


if __name__ == "__main__":
    main()
