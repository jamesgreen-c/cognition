"""

Actor job:

1. Take latents from RPSSM output and produce an action: 
    - apply NN to posterior mean
    - posterior means will be shaped (N, D) so this needs 
        to be distributed over observations N using a single network
    - actions need to be mapped to the correct space for the 
        problem at hand (ActionMap)
2. Initialise the network according to its input (RPSSM posterior means)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from jax import vmap, Array
from typing import Callable

from slac.actionmaps import ActionMap


INITIALIZER = jax.nn.initializers.variance_scaling(
    scale=0.1, mode='fan_in', distribution='truncated_normal'
)


class ActorNetwork(nn.Module):
    network: nn.Module
    kernel_init: Callable = INITIALIZER
    bias_init: Callable = jax.nn.initializers.zeros
    action_map: ActionMap

    @nn.compact
    def __call__(self, x: Array):
        x = self.network(x)
        x = nn.Dense(
            self.action_map.input_dim, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init
        )(x)
        return self.action_map(x)


class Actor:

    def __init__(self,  network: ActorNetwork):
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
    
    def get_actions(self, params, data: Array):
        """
        Parameters
        ----------
        params:  Network parameters
        data:    (N, D) The current posterior means from the RPSSM to take actions on
        """
        outs = vmap(lambda x: self.network.apply(params, x))(data)
        return outs
    


