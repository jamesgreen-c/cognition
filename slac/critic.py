"""

Actor job:

1. Take latents from RPSSM output and produce a value: 
    - apply NN to posterior mean
    - posterior means will be shaped (N, D) so this needs 
        to be distributed over observations N using a single network
    - actions need to be mapped to scalar values
2. Initialise the network according to its input (RPSSM posterior means)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from jax import vmap, Array
from typing import Callable


INITIALIZER = jax.nn.initializers.variance_scaling(
    scale=0.1, mode='fan_in', distribution='truncated_normal'
)


class CriticNetwork(nn.Module):
    network: nn.Module
    kernel_init: Callable = INITIALIZER
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: Array):
        x = self.network(x)
        x = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x


class Critic:

    def __init__(self,  network: CriticNetwork):
        self.network = network

    def init(self, key: Array, data: Array):
        """
        Parameters
        ----------
        key:   PRNGKey
        data:  (N, D) Example input vector (any posterior mean from RPSSM)
        """
        params = self.network.init(key, data[0, 0])
        return params
    
    def get_values(self, params, data: Array):
        """
        Parameters
        ----------
        params:  Network parameters
        data:    (N, D) The current posterior means from the RPSSM to estimate values for
        """
        outs = vmap(lambda x: self.network.apply(params, x))(data)
        return outs
    


