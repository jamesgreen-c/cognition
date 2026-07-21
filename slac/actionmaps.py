import jax.numpy as jnp
from jax import Array


class ActionMap:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    @property
    def input_dim(self):
        pass

    def __call__(self, x: Array):
        pass


class Binary(ActionMap):
    """map NN output to range [-1, +1]"""

    @property
    def input_dim(self):
        return 1
    
    def __call__(self, x: Array):
        return jnp.tanh(x)