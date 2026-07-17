import optax

from typing import Callable, Union, Optional
from flax.struct import dataclass


LearningRate = Union[float, Callable[[int], float]]


@dataclass
class Config:
    batch_size: int = 32
    num_iter: int = 1000
    seed: int = 0
    jit: bool = True
    beta_schedule: LearningRate = lambda i: 1.
    prior_opt: Callable = optax.adam
    prior_lr: LearningRate = 1e-3
    rec_lr: tuple[LearningRate, ...] = (1e-3,)
    stabilize_A: Optional[str] = 'scale'
    em: bool = False  # if True, perform EM (don't backprop through posterior)

    # adaptive learning rate decay
    lr_decay_steps: int = None
    lr_decay_rate: float = None
    lr_staircase: bool = False

