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
        return jnp.squeeze(x, axis=-1)


class Critic:

    def __init__(self,  network: CriticNetwork):
        self.network = network

    def init(self, key: Array, latent: Array, action: Array):
        """
        Parameters
        ----------
        key:     PRNGKey
        latent:  (D) Example latent state
        action:  (K) Example action vector 
        """
        key_1, key_2 = jr.split(key)
        data = jnp.concatenate((latent, action), axis=-1)

        # two critics for soft actor-critic
        params_1 = self.network.init(key_1, data)
        params_2 = self.network.init(key_2, data)

        # set target params equal to starting params
        params = {
            "latest": {"one": params_1, "two": params_2},
            "target": {"one": params_1, "two": params_2},
        }
        return params
    
    def apply(self, params, latent: Array, action: Array, mode: str):
        """
        Parameters
        ----------
        params:  Network parameters
        latent:  (N, D) sample latent from the LGSSM posterior
        action:  (N, K) the action taken at this state
        mode:    The mode to use. If
                     - target:  
                     - latest: 
        """

        data = jnp.concatenate((latent, action), axis=-1)
        if mode == "target":
            values_1 = self.network.apply(params["target"]["one"], data)
            values_2 = self.network.apply(params["target"]["two"], data)

        elif mode == "latest":
            values_1 = self.network.apply(params["latest"]["one"], data)
            values_2 = self.network.apply(params["latest"]["two"], data)
        else:
            raise NotImplementedError(f"Critic mode {mode} is not a valid mode. Choose 'target' or 'latest'")
        return values_1, values_2
    
