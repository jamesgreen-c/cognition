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

from typing import Callable

from jax import vmap, Array
from jax.random import PRNGKey

from rp_ssm.distributions import DistMap, NatParam


INITIALIZER = jax.nn.initializers.variance_scaling(
    scale=0.1, mode='fan_in', distribution='truncated_normal'
)


class ActorNetwork(nn.Module):
    network: nn.Module
    dist_map: DistMap
    kernel_init: Callable = INITIALIZER
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: Array) -> NatParam:
        x = self.network(x)
        x = nn.Dense(
            self.dist_map.input_dim, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init
        )(x)
        return self.dist_map(x)


class Actor:

    def __init__(self,  network: ActorNetwork, action_shape: tuple):
        self.network = network
        self.action_shape = action_shape

    def init(self, key: Array, data: Array):
        """
        Parameters
        ----------
        key:   PRNGKey
        data:  (D) Example input vector (any posterior mean from RPSSM)
        """
        params = self.network.init(key, data)
        return params
    
    def apply(self, key: PRNGKey, params, data: Array):
        """
        Network parametrises a differentiable distribution over actions.
        Sample the action.
        Calculate the log probability of this action.
        Gradients wrt to actor.apply are therefore: d log[p(a | s)]

        Parameters
        ----------
        params:  Network parameters
        data:    (D) The current posterior means from the RPSSM to take actions on
        """
        nat_params = self.network.apply(params, data)
        dist = nat_params.dist_param

        action = dist.sample(key=key, action_shape=self.action_shape)
        log_prob = dist.log_prob(action)

        return log_prob, action



    


