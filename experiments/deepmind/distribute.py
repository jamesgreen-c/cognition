import argparse
import pickle
import shlex
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default="cheetah")
parser.add_argument("--task", type=str, default="run")
parser.add_argument("--action-repeat", type=int, default=4)
parser.add_argument("--actor-history", type=int, default=8)
parser.add_argument("--N", type=int, default=4)
parser.add_argument("--D", type=int, default=32)
parser.add_argument("--T", type=int, default=32)
parser.add_argument("--pixels", type=int, default=64)
parser.add_argument("--rgb", action="store_true")
parser.add_argument("--stabilise", type=str, default="clip")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--pretrain-iter", type=int, default=100000)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--capacity", type=int, default=100000)
parser.add_argument("--eval-episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--max-iter", type=int, default=500000)
parser.add_argument("--verbose-warp", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()


EXPERIMENT_NAME = (
    f"domain={args.domain},task={args.task},D={args.D},N={args.N},T={args.T},"
    f"history={args.actor_history},repeat={args.action_repeat},pretrain={args.pretrain_iter},"
    f"rgb={args.rgb},pixels={args.pixels},stabilise={args.stabilise},seed={args.seed}"
)
DIRPATH = Path("results") / EXPERIMENT_NAME


def evaluation_targets(max_iter: int) -> list[int]:
    """Evaluate once at 2,500, then at each 10,000-step checkpoint."""
    if max_iter < 2500:
        raise ValueError("max_iter must be at least 2500.")
    targets = [2500]
    targets.extend(range(10000, max_iter + 1, 10000))
    if targets[-1] != max_iter:
        targets.append(max_iter)
    return targets


def checkpoint_iteration() -> int:
    """Return the number of completed updates, or zero if no run exists."""
    params_path = DIRPATH / "params.pkl"
    loss_path = DIRPATH / "loss.pkl"
    if not params_path.exists() and not loss_path.exists():
        return 0
    if not params_path.is_file() or not loss_path.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint in {DIRPATH}.")

    with loss_path.open("rb") as f:
        losses = pickle.load(f)
    completed = len(losses["model"])
    if any(len(losses[name]) != completed for name in ("critic", "actor", "alpha")):
        raise ValueError(f"Inconsistent training history lengths in {DIRPATH}.")
    return completed


def evaluation_exists(num_iter: int) -> bool:
    return (DIRPATH / f"evaluation,num-iter={num_iter}" / "evaluation.pkl").is_file()


def common_arguments() -> list[str]:
    command = [
        "--domain", args.domain,
        "--task", args.task,
        "--action-repeat", str(args.action_repeat),
        "--actor-history", str(args.actor_history),
        "--N", str(args.N),
        "--D", str(args.D),
        "--T", str(args.T),
        "--pixels", str(args.pixels),
        "--stabilise", args.stabilise,
        "--gamma", str(args.gamma),
        "--pretrain-iter", str(args.pretrain_iter),
        "--batch-size", str(args.batch_size),
        "--capacity", str(args.capacity),
        "--seed", str(args.seed),
        "--platform", args.platform,
    ]
    if args.rgb:
        command.append("--rgb")
    if args.verbose_warp:
        command.append("--verbose-warp")
    return command


def run(command: list[str]) -> None:
    print("\nExecuting:", " ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def train(num_iter: int, *, continue_run: bool) -> None:
    command = [
        sys.executable,
        "experiment.py",
        *common_arguments(),
        "--num-iter", str(num_iter),
    ]
    if continue_run:
        command.append("--continue")
    if args.debug:
        command.append("--debug")
    run(command)


def evaluate() -> None:
    command = [
        sys.executable,
        "evaluate.py",
        *common_arguments(),
        "--eval-episodes", str(args.eval_episodes),
    ]
    run(command)


def main() -> None:
    targets = evaluation_targets(args.max_iter)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Evaluation checkpoints: {targets}")

    completed = checkpoint_iteration()
    if completed > args.max_iter:
        raise ValueError(
            f"Checkpoint already contains {completed} iterations, beyond "
            f"max_iter={args.max_iter}."
        )

    for target in targets:
        if completed < target:
            increment = target - completed
            train(increment, continue_run=completed > 0)
            if args.dry_run:
                completed = target
            else:
                completed = checkpoint_iteration()
                if completed != target:
                    raise RuntimeError(
                        f"Expected checkpoint at iteration {target}, got {completed}."
                    )

        if completed == target:
            if evaluation_exists(target):
                print(f"Skipping existing evaluation at iteration {target}.")
            else:
                evaluate()
        elif not evaluation_exists(target):
            raise FileNotFoundError(
                f"Checkpoint is already at iteration {completed}, but evaluation "
                f"data for iteration {target} is missing. The historical parameters "
                "are no longer available to reconstruct that evaluation."
            )


if __name__ == "__main__":
    main()
