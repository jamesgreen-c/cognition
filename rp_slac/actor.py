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
from jax.lax import stop_gradient as stop_grad
from jax.random import PRNGKey

from rp_slac.distributions import NatParam
from rp_slac.recognition.distmaps import DistMap


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

    def __init__(
            self,
            network: ActorNetwork,
            action_low: Array,
            action_high: Array,
        ):
        self.network = network

        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

    def init(self, key: Array, state: Array):
        """
        Parameters
        ----------
        key:   PRNGKey
        data:  Tuple[(B, T, ...)] batched replay buffer
        """
        return self.network.init(key, state)

    def apply(self, key: PRNGKey, params, state: Array):
        """
        Network parametrises a differentiable distribution over actions.
        Sample the action.
        Calculate the log probability of this action.
        Gradients wrt to actor.apply are therefore: d log[p(a | s)]

        Parameters
        ----------
        params:  Network parameters
        data:    State to apply actor to
        """

        # apply network
        nat_params = self.network.apply(params, state)
        dist = nat_params.dist_param

        # squash action to [-1, 1]
        raw_action = dist.sample(key=key, shape=())
        squashed_action = jnp.tanh(raw_action)

        # affine transormation to actual action space
        action = self.action_bias + self.action_scale * squashed_action

        # calculate log pi(a_t | s_t) and jacobian for tanh squash
        log_det = 2.0 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action))   # jnp.log(1.0 - squashed_action ** 2 + 1e-6)
        log_prob = dist.log_prob(raw_action)
        log_prob -= jnp.sum(jnp.log(self.action_scale) + log_det)

        return action, log_prob

    def stats(self, params, state):
        nat_params = self.network.apply(params, state)
        dist = nat_params.dist_param

        mean = dist.params["mean"]
        std = jnp.sqrt(jnp.diag(dist.params["cov"]))

        action = self.action_bias + self.action_scale * jnp.tanh(mean)
        return action, std



# class Actor:

#     def __init__(self,  network: ActorNetwork):
#         self.network = network

#     def init(self, key: Array, state: Array):
#         """
#         Parameters
#         ----------
#         key:   PRNGKey
#         data:  Tuple[(B, T, ...)] batched replay buffer
#         """
#         params = self.network.init(key, state)
#         return params

    
#     def apply(self, key: PRNGKey, params, state: Array):
#         """
#         Network parametrises a differentiable distribution over actions.
#         Sample the action.
#         Calculate the log probability of this action.
#         Gradients wrt to actor.apply are therefore: d log[p(a | s)]

#         Parameters
#         ----------
#         params:  Network parameters
#         data:    State to apply actor to
#         """
#         nat_params = self.network.apply(params, state)
#         dist = nat_params.dist_param

#         raw_action = dist.sample(key=key, shape=())

#         action = jnp.tanh(raw_action)
#         log_prob = dist.log_prob(raw_action)
#         log_prob -= jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6))

#         return action, log_prob

#     def stats(self, params, state):
#         nat_params = self.network.apply(params, state)
#         dist = nat_params.dist_param

#         mean = dist.params["mean"]
#         std = jnp.sqrt(jnp.diag(dist.params["cov"]))

#         return jnp.tanh(mean), std
