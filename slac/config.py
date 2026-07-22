
import optax

from typing import Callable, Union, Optional
from dataclasses import dataclass, field


LearningRate = Union[float, Callable[[int], float]]


@dataclass
class OptimConfig:
    """ Configuration for a single parameter block optimiser. """
    optimizer: Callable[[float], optax.GradientTransformation] = optax.adam
    lr: LearningRate = 1e-3

    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    staircase: bool = False

    def schedule(self) -> LearningRate:
        """
        Returns either a constant learning rate, a user-provided schedule, or an exponential-decay schedule.
        """
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
        """ Builds the Optax optimiser. """
        return self.optimizer(self.schedule())


@dataclass
class Config:
    """
    Training configuration for recognition and prior optimisation.
    """
    num_iter: int = 100
    seed: int = 0

    debug: bool = False

    actor: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            optimizer=optax.adam,
            lr=1e-3,
        )
    )

    critic: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            optimizer=optax.adam,
            lr=1e-3,
        )
    )

