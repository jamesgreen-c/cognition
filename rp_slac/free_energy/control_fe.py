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
        self.vmapped_actor = vmap(self.actor.apply, in_axes=(0, None, 0))

        self.critic = critic
        self.vmapped_target_critic = vmap(
            lambda _ps, _ls, _as: self.critic.apply(_ps, _ls, _as, mode="target"), 
            in_axes=(None, 0, 0)
        )
        self.vmapped_latest_critic = vmap(
            lambda _ps, _ls, _as: self.critic.apply(_ps, _ls, _as, mode="latest"), 
            in_axes=(None, 0, 0)
        )

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
        action_dim = data[1].shape[-1]

        self.batch_size = config.batch_size
        self.num_buffers = config.num_buffers
        self.gamma = config.gamma
        self.actor_state = config.actor_state
        self.target_entropy = -float(action_dim) if config.target_entropy is None else config.target_entropy

        params = {}
        params["actor"] = self.actor.init(key, data[0][0, 0])                           # applied to first observation
        params["critic"] = self.critic.init(key, jnp.zeros(latent_dim), data[1][0, 0])  # applied to first action
        params["log_alpha"] = jnp.asarray(config.initial_log_alpha, dtype=jnp.float32)
        
        opts = {
            "actor": config.actor.build(), 
            "critic": config.critic.build(), 
            "alpha": config.alpha.build()
        }
        opt_states = {
            "actor": opts["actor"].init(params["actor"]),
            "critic": opts["critic"].init(params["critic"]["latest"]),                  # only create opt state for latest
            "alpha": opts["alpha"].init(params["log_alpha"]),
        }
        # opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def actor_loss(self, key: PRNGKey, params: dict, latents: Array, data: tuple[Array]) -> tuple[float, Any]:
        """
        
        Parameters
        ----------
        latents:  (B, T, D) last latent inferred by RPM
        data:     Tuple[(B, T, ...)] batched replay buffer
        """
        alpha = jnp.exp(stopgrad(params["log_alpha"]))
        latent = stopgrad(latents[:, -1])                                                    # (B, D)
        actor_state = data[0][:, -1] if self.actor_state == "observation" else latent

        B = latent.shape[0]
        keys = jr.split(key, B)
        actions, log_probs = self.vmapped_actor(keys, params["actor"], actor_state)          # (B,), (B,)

        # track actor stats
        means, stds = vmap(lambda _state: self.actor.stats(params["actor"], _state))(actor_state)

        # get loss terms
        values_1, values_2 = self.vmapped_latest_critic(params["critic"], latent, actions)   # (B,), (B,)
        values = jnp.minimum(values_1, values_2)                                             # (B,)
        terms = (alpha * log_probs) - values                                      # (B,)
        loss = terms.mean()
        # loss =  terms.sum() / (self.batch_size * self.num_buffers)

        aux = {"mean": means, "std": stds}
        return loss, aux

    def critic_loss(self, key: PRNGKey, params: dict, latents: Array, data: tuple[Array]) -> tuple[float, Any]:
        """
        
        Parameters
        ----------
        latents:  (B, T, D) last latent inferred by RPM
        data:     Tuple[(B, T, ...)] batched replay buffer
        """

        alpha = jnp.exp(stopgrad(params["log_alpha"]))
        latent = stopgrad(latents[:, -2])                                                 # (B, D)
        next_latent = stopgrad(latents[:, -1])                                            # (B, D)
        actor_state = data[0][:, -1] if self.actor_state == "observation" else next_latent

        action = data[1][:, -1]                                                           # (B, K)
        reward = data[2][:, -1]                                                           # (B,)
        discount = data[3][:, -1]                                                         # (B,)

        B = latent.shape[0]
        keys = jr.split(key, B)

        next_action, log_probs = self.vmapped_actor(keys, params["actor"], actor_state)               # (B,), (B,)
        target_1, target_2 = self.vmapped_target_critic(params["critic"], next_latent, next_action)   # (B,), (B,)
        values_1, values_2 = self.vmapped_latest_critic(params["critic"], latent, action)             # (B,), (B,)
        
        # get loss term
        diff = jnp.minimum(target_1, target_2) - alpha * log_probs                  # (B,)
        target = stopgrad(reward + self.gamma * discount * diff)                    # (B,)
        terms = jnp.square(values_1 - target) + jnp.square(values_2 - target)       # (B,)
        loss = 0.5 * terms.mean()
        # loss = terms.sum() / (2 * self.batch_size * self.num_buffers) 

        return loss

    def alpha_loss(self, params: dict, data: tuple[Array]) -> Array:
        """
        Parameters 
        ----------
        params:     dict
        log_probs:  (B,) batch of log probabilities for the actions taken in this training iteration.
        """
        log_alpha = params["log_alpha"]
        log_probs = data[4][:, -1]

        terms = log_alpha * stopgrad(-log_probs - self.target_entropy)
        loss = terms.mean()
        return loss

    def policy(self, key: PRNGKey, params: dict, state: Array):
        action, log_prob = self.actor.apply(key, params["actor"], state)
        return action, log_prob

    def mean_policy(self, params: dict, state: Array):
        mean, _ = self.actor.stats(params["actor"], state)
        return mean


