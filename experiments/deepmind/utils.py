from pathlib import Path
import pickle

#
def find_checkpoint(dirpath: str, args):
    """Find the checkpoint for this experiment's stable directory name."""
    path = Path(dirpath)
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


def load_checkpoint(trainer, path, args):

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