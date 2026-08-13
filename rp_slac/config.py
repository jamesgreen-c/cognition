import optax

from dataclasses import dataclass, field
from typing import Callable, Union, Optional


LearningRate = Union[float, Callable[[int], float]]


@dataclass
class OptimConfig:
    """Configuration for a single parameter-block optimiser."""

    optimizer: Callable[[LearningRate], optax.GradientTransformation] = optax.adam
    lr: LearningRate = 1e-3
    max_grad_norm: Optional[float] = None

    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    staircase: bool = False

    def schedule(self) -> LearningRate:
        if callable(self.lr):
            return self.lr

        if self.decay_steps is None or self.decay_rate is None:
            return self.lr

        return optax.exponential_decay(
            init_value=self.lr,
            transition_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase,
        )

    def build(self) -> optax.GradientTransformation:
        transforms = []

        if self.max_grad_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.max_grad_norm))

        transforms.append(self.optimizer(self.schedule()))
        return optax.chain(*transforms)


@dataclass
class Config:
    # replay sampling
    batch_size: int = 32       # B windows from each replay buffer
    sequence_length: int = 50  # tau transitions; observations have tau + 1
    num_buffers: int = 1       # N independent chronological buffers

    # training
    num_pretrain: int = 1000
    num_iter: int = 1000

    collection_steps: int = 1
    capacity: int = 500

    # control
    gamma: float = 0.99
    temperature: float = 0.95
    target_update_rate: float = 0.005
    actor_state: str = "observation"
    stop_actor_gradient: bool = True

    # model
    beta_schedule: LearningRate = lambda i: 1.0
    stabilise_A: Optional[str] = "scale"
    em: bool = False

    # execution
    seed: int = 0
    jit: bool = True

    prior: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=1e-3,
        max_grad_norm=10.0,
    ))

    recognition: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=1e-3,
        max_grad_norm=10.0,
    ))

    actor: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=5e-4,
        max_grad_norm=10.0,
    ))

    critic: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=1e-3,
        max_grad_norm=10.0,
    ))