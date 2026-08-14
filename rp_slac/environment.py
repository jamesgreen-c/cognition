from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr

from typing import Callable

from jax import Array, lax, vmap
from jax.random import PRNGKey



class Environment(ABC):

    def __init__(self):
        pass

    def sample(
            self, 
            key: PRNGKey, 
            policy: Callable, 
            num_samples: int, 
            num_steps: int, 
            state_0: Array | None = None, 
        ):
        """
        Generate independent trajectories using a policy.

        The policy must have signature
            action = policy(key, state)

        Parameters
        ----------
        key:          JAX random key.
        policy:       Callable mapping (key, state) to an action.
        num_samples:  Number of trajectories M.
        state_0:      If provided, should be shape (D,) for a single state, or (M, D) for a batch of starting states

        Returns
        -------
        final_states:  Final state of each trajectory, shape (M, *D).
        states:        State trajectories, shape (M, T, *D).
        observations:  Observation trajectories, shape (M, T, *O).
        actions:       Actions, shape (M, T - 1, *K).
        rewards:       Rewards, shape (M, T - 1).
        flags:         Continuation flags, shape (M, T - 1).  A flag is 0 when the
                         corresponding transition terminates an episode and 1 otherwise.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")

        if num_steps < 1:
            raise ValueError("num_steps must be positive.")

        init_key, sample_key = jr.split(key)
        state_0 = self.initial_state(init_key) if state_0 is None else state_0

        if state_0.ndim == 1:
            initial_states = jnp.broadcast_to(state_0, (num_samples,) + state_0.shape)
        else:
            if state_0.shape[0] != num_samples:
                raise ValueError(
                    "Batched state_0 must have leading dimension {}, "
                    "received shape {}.".format(num_samples, state_0.shape)
                )
            initial_states = state_0

        trajectory_keys = jr.split(sample_key, num_samples)
        return vmap(lambda k, s: self._sample_single(k, s, policy, num_steps))(trajectory_keys, initial_states)

    def _sample_single(self, key: PRNGKey, state_0: Array, policy: Callable, num_steps: int):
        """
        Generate one trajectory containing num_steps observations.

        When an episode terminates, the environment is reset before the
        following action is selected. The terminating transition receives
        flag 0.
        """
        obs_key, step_key = jr.split(key)
        step_keys = jr.split(step_key, num_steps)

        def scan_step(carry, key):
            policy_key, model_key, reset_key, observation_key = jr.split(key, 4)
            state, obs = carry

            # action-conditioned transition
            action, log_prob = policy(policy_key, obs)
            next_state, reward = self.model(model_key, state, action)

            # if terminal reset environment
            terminal = self.is_terminal_state(next_state)
            flag = 1.0 - terminal.astype(jnp.float32)
            reset_state = self.initial_state(reset_key)

            # choose next state and observe
            next_state = jnp.where(terminal, reset_state, next_state)
            next_obs = self.observe(observation_key, next_state)
            return (next_state, next_obs), (next_state, next_obs, action, reward, flag, log_prob)

        obs_0 = self.observe(obs_key, state_0)
        carry_0 = (state_0, obs_0)
        (final_state, _), outputs = lax.scan(scan_step, carry_0, step_keys)
        states, observations, actions, rewards, flags, log_probs = outputs

        # insert state 0
        states = jnp.concatenate([state_0[None], states], axis=0)
        observations = jnp.concatenate([obs_0[None], observations], axis=0)
        return final_state, states, observations, actions, rewards, flags, log_probs

    def is_terminal_state(self, state: Array) -> Array:
        """
        Return whether the state terminates the current episode.
        Continuing environments use the default implementation.
        """
        return jnp.asarray(False)

    @abstractmethod
    def transition(self, key: PRNGKey, state: Array, action: Array):
        """
        Implement environment specific transition step 
        based on the current state and taken action.
        
        Parameters
        ---------- 
        key:     PRNGKey
        state:   (*D) current state of the environment
        action:  (*K) action to apply to state and environment

        Returns
        -------
        next_state:  The next state of the environment
        """
        pass

    @abstractmethod
    def model(self, key: PRNGKey, state: Array, action: Array):
        """
        Implement p(S', r | S, a)
        
        Parameters
        ---------- 
        key:     PRNGKey
        state:   (*D) current state of the environment
        action:  (*K) action to apply to state and environment

        Returns
        ------- 
        reward:      The sampled reward associated with the reached state and action
        next_state:  The next state of the environment
        """
        pass

    @abstractmethod
    def observe(self, key: PRNGKey, state: Array):
        """
        Implement observation of env state. 
        Ie could be exactly the environment state with no noise, could be an image generator etc
        
        Parameters
        ---------- 
        key:     PRNGKey
        state:   (*D) current state of the environment

        Returns
        ------- 
        observation:  The observation to be passed to RPM
        """
        pass

    @abstractmethod
    def initial_state(self, key: PRNGKey):
        """
        Implement a sampling function for the initial state.
        It could be a predetermined start state.

        Parameters
        ----------
        key:  RNG

        Returns
        -------
        initial_state:  a sample of the initial state
        """
        pass

    @abstractmethod
    def random_action(self, key: PRNGKey, state: Array):
        """Sample a valid action for replay-buffer initialisation."""
        pass