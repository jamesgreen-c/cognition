import optax
import jax.numpy as jnp
import jax.random as jr

from typing import Any

from jax import vmap, Array
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from jax.lax import stop_gradient as stopgrad

from rp_slac.utils.math import inv_quad_form
from rp_slac.distributions import AllParam, AllParams

from rp_slac.actor import Actor
from rp_slac.config import Config
from rp_slac.critic import Critic


class ControlFreeEnergy:
    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.actor.apply = vmap(self.actor.apply, in_axes=(0, None, 0))

        self.critic = critic
        self.critic.apply = vmap(self.critic.apply, in_axes=(None, 0, 0, None))

    def init(
            self,
            key: Array,
            data: list[Array],
            config: Config,
            latent_dim: int,
        ) -> tuple[AllParams, list[optax.OptState], list[optax.GradientTransformation]]:
        """
        
        Parameters
        ----------
        data: tuple[(B, T, ...)]
        """

        self.batch_size = config.batch_size
        self.num_buffers = config.num_buffers
        self.temperature = config.temperature
        self.gamma = config.gamma
        self.actor_state = config.actor_state

        params = {}
        params["actor"] = self.actor.init(key, data[0][0, 0])  # applied to first observation
        params["critic"] = self.critic.init(key, jnp.zeros(latent_dim))

        opts = {"actor": config.actor.build(), "critic": config.critic.build()}
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def actor_loss(self, key: PRNGKey, params: dict, latents: Array, data: tuple[Array]) -> tuple[float, Any]:
        """
        
        Parameters
        ----------
        latents:  (B, T, D) last latent inferred by RPM
        data:     Tuple[(B, T, ...)] batched replay buffer
        """
        latent = stopgrad(latents[:, -1])                                               # (B, D)
        actor_state = data[0][:, -1] if self.actor_state == "observation" else latent

        B = latent.shape[0]
        keys = jr.split(key, B)
        actions, log_probs = self.actor.apply(keys, params["actor"], actor_state)  # (B,), (B,)

        # get loss terms
        values_1, values_2 = self.critic.apply(params["critic"], 
                                               latent, 
                                               actions,
                                               mode="latest")                      # (B,), (B,)
        values = jnp.minimum(values_1, values_2)                                   # (B,)
        terms = (self.temperature * log_probs) - values                            # (B,)
        loss =  terms.sum() / (self.batch_size * self.num_buffers)                 # TODO check if should be negative

        return loss

    def critic_loss(self, key: PRNGKey, params: dict, latents: Array, data: tuple[Array]) -> tuple[float, Any]:
        """
        
        Parameters
        ----------
        latents:  (B, T, D) last latent inferred by RPM
        data:     Tuple[(B, T, ...)] batched replay buffer
        """

        latent = stopgrad(latents[:, -2])                                                 # (B, D)
        next_latent = stopgrad(latents[:, -1])                                            # (B, D)
        actor_state = data[0][:, -1] if self.actor_state == "observation" else next_latent

        action = data[1][:, -1]                                                           # (B, K)
        reward = data[2][:, -1]                                                           # (B,)
        discount = data[3][:, -1]                                                         # (B,)

        next_action, log_probs = self.actor.apply(key, params["actor"], actor_state)      # (B,), (B,)

        target_1, target_2 = self.critic.apply(params["critic"], 
                                               next_latent, 
                                               next_action, 
                                               mode="target")                       # (B,), (B,)

        values_1, values_2 = self.critic.apply(params["critic"], 
                                               latent, 
                                               action, 
                                               mode="latest")                       # (B,), (B,)
        
        # get loss term
        diff = jnp.minimum(target_1, target_2) - self.temperature * log_probs       # (B,)
        target = stopgrad(reward + self.gamma * discount * diff)                    # (B,)
        terms = jnp.square(values_1 - target) + jnp.square(values_2 - target)       # (B,)
        loss = terms.sum() / (2 * self.batch_size * self.num_buffers) 

        return loss


    def policy(self, key: PRNGKey, params: dict, state: Array):
        action, _ = self.actor.apply(key, params["actor"], state)
        return action


