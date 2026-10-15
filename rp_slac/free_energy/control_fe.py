"""SAC losses with a shared, model-trained image encoder and action history."""

from typing import Any

import optax
import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap
from jax.lax import stop_gradient as stopgrad

from rp_slac.actor import Actor
from rp_slac.config import Config
from rp_slac.critic import Critic
from rp_slac.recognition.ramp import RAMPSSM


class ControlFreeEnergy:

    def __init__(self, actor: Actor, critic: Critic, encoder: RAMPSSM):

        self.encoder = encoder

        self.actor = actor
        self.vmapped_actor = vmap(self.actor.apply, in_axes=(0, None, 0))

        self.critic = critic
        self.vmapped_target_critic = vmap(
            lambda ps, z, a: self.critic.apply(ps, z, a, mode='target'),
            in_axes=(None, 0, 0),
        )
        self.vmapped_latest_critic = vmap(
            lambda ps, z, a: self.critic.apply(ps, z, a, mode='latest'),
            in_axes=(None, 0, 0),
        )


    def configure(self, config: Config, data=None):
        action_dim = self.actor.action_scale.shape[-1] if data is None else data[1].shape[-1]

        self.actor_history = config.actor_history
        self.batch_size = config.sac_batch_size
        self.num_buffers = config.num_buffers
        self.gamma = config.gamma
        self.actor_state = config.actor_state
        self.target_entropy = -float(action_dim) if config.target_entropy is None else config.target_entropy


    def actor_input(self, params, observations, past_actions, features=None):
        """Build (B, F*feature_dim+(F-1)*action_dim) policy inputs.

        observations are oldest-to-newest (B,F,...); past_actions contains
        the F-1 actions between those frames. No rewards enter the policy.
        Shared features are stopped for SAC, and trained by the model loss.
        """

        if observations.shape[1] != self.actor_history:
            raise ValueError('The policy needs exactly actor_history observations.')
        if past_actions.shape[:2] != (observations.shape[0], self.actor_history - 1):
            raise ValueError('The policy needs actor_history-1 aligned past actions.')
        # if self.encoder is None:
        #     return self._stack_frames(observations)
        
        if features is None:
            features = self.encoder.get_features(params['rpm'], observations)

        features = stopgrad(features)
        B = features.shape[0]
        return jnp.concatenate((features.reshape((B, -1)), past_actions.reshape((B, -1))), axis=-1)


    def init(self, key: Array, data, config: Config, latent_dim: int, model_params: dict):
        """
        
        Parameters
        ----------
        data: tuple[(B, T, ...)]
        """

        self.configure(config, data)
        actor_key, critic_key = jr.split(key)

        if config.sequence_length < self.actor_history:
            raise ValueError('sequence_length must be at least actor_history for current-state SAC histories.')
        
        if self.actor_state == 'observation':
            frames = data[0][:1, :self.actor_history]
            actions = data[1][:1, :self.actor_history - 1]
            example = self.actor_input(model_params, frames, actions)[0]
        else:
            example = jnp.zeros(latent_dim)

        params = {}
        params["actor"] = self.actor.init(actor_key, example)                                  # applied to first observation
        params["critic"] = self.critic.init(critic_key, jnp.zeros(latent_dim), data[1][0, 0])  # applied to first action
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
        return params, opt_states, opts


    def _actor_state(
            self, params: dict, latents: Array, data, 
            next_state: bool, features: Array | None = None
        ):

        if self.actor_state == 'latent':
            index = -1 if next_state else -2
            return stopgrad(latents[:, index])

        end = data[0].shape[1] if next_state else data[0].shape[1] - 1
        start = end - self.actor_history
        
        frames = data[0][:, start:end]
        actions = data[1][:, start:end - 1]
        feature_window = None if features is None else features[:, start:end]
        return self.actor_input(params, frames, actions, feature_window)


    def actor_loss(self, key, params, latents, data, features: Array) -> tuple[Array, Any]:
        """
        
        Parameters
        ----------
        latents:  (B, T, D) last latent inferred by RPM
        data:     Tuple[(B, T, ...)] batched replay buffer
        """
        alpha = jnp.exp(stopgrad(params["log_alpha"]))
        latent = stopgrad(latents[:, -2])                                                    # (B, D)
        actor_state = self._actor_state(params, latents, data, False, features)

        keys = jr.split(key, latent.shape[0])                                                # (B,)
        actions, log_probs = self.vmapped_actor(keys, params["actor"], actor_state)          # (B,), (B,)

        # track actor stats
        means, stds = vmap(lambda _x: self.actor.stats(params["actor"], _x))(actor_state)

        # get loss terms
        values_1, values_2 = self.vmapped_latest_critic(params["critic"], latent, actions)   # (B,), (B,)
        values = jnp.minimum(values_1, values_2)                                             # (B,)
        terms = (alpha * log_probs) - values                                                 # (B,)
        loss = terms.mean()

        aux = {"mean": means, "std": stds, "log_probs": log_probs}
        return loss, aux


    def critic_loss(self, key, params, latents, data, features=None):
        """
        
        Parameters
        ----------
        latents:  (B, T, D) last latent inferred by RPM
        data:     Tuple[(B, T, ...)] batched replay buffer
        """

        alpha = jnp.exp(stopgrad(params["log_alpha"]))
        latent = stopgrad(latents[:, -2])                                                 # (B, D)
        next_latent = stopgrad(latents[:, -1])                                            # (B, D)
        actor_state = self._actor_state(params, latents, data, True, features)

        action = data[1][:, -1]                                                           # (B, K)
        reward = data[2][:, -1]                                                           # (B,)
        discount = data[3][:, -1]                                                         # (B,)

        keys = jr.split(key, latent.shape[0])                                                         # (B,)
        next_action, log_probs = self.vmapped_actor(keys, params["actor"], actor_state)               # (B,), (B,)
        target_1, target_2 = self.vmapped_target_critic(params["critic"], next_latent, next_action)   # (B,), (B,)
        values_1, values_2 = self.vmapped_latest_critic(params["critic"], latent, action)             # (B,), (B,)
        
        # get loss term
        diff = jnp.minimum(target_1, target_2) - alpha * log_probs                  # (B,)
        target = stopgrad(reward + self.gamma * discount * diff)                    # (B,)
        terms = jnp.square(values_1 - target) + jnp.square(values_2 - target)       # (B,)
        loss = 0.5 * terms.mean()

        return loss


    def alpha_loss(self, params, log_probs):
        """
        Parameters 
        ----------
        params:     dict
        log_probs:  (B,) batch of log probabilities for the actions taken in this training iteration.
        """
        gap = stopgrad(-log_probs - self.target_entropy)
        loss = (params['log_alpha'] * gap).mean()
        return loss, {'entropy_gap': gap.mean()}


    def policy(self, key, params, state):
        """Apply the actor to an already-built policy feature vector."""
        return self.actor.apply(key, params['actor'], state)


    def mean_policy(self, params, state):
        return self.actor.stats(params['actor'], state)[0]


    # def _stack_frames(self, frames: Array) -> Array:
    #     """
    #     frames: (B, F, H, W, C)
    #     returns: (B, H, W, F * C)
    #     """
    #     return jnp.concatenate([frames[:, i] for i in range(self.actor_history)], axis=-1)
