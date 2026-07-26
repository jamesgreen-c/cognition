from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable

from jax import Array, lax, vmap
from jax.random import PRNGKey

from rp_ssm.utils.dataset import Dataset


class Environment(ABC):

    def __init__(self):
        pass

    def sample(
            self, 
            key: PRNGKey, 
            policy: Callable, 
            num_samples: int
        ):
        """
        Generate independent trajectories using a policy.

        The policy must have signature
            action = policy(key, state)
        where state has shape (4,) and action is a scalar in [-1, 1].

        Parameters
        ----------
        key:          JAX random key.
        state_0:      Common initial state with shape (4,), or separate initial states with shape (M, 4).
        policy:       Callable mapping (key, state) to an action.
        num_samples:  Number of trajectories M.

        Returns
        -------
        final_states:  Final state of each trajectory, shape (M, 4).
        states:        State trajectories including state_0, shape (M, T, 4).
        actions:       Actions generating states[:, 1:], shape (M, T - 1).
        """
        init_key, subkey = jr.split(key)
        state_0 = self.initial_state(init_key) 

        if state_0.ndim == 1:
            initial_states = jnp.broadcast_to(state_0, (num_samples, state_0.shape[0]))
        elif state_0.shape == (num_samples, 4):
            initial_states = state_0
        else:
            raise ValueError(f"state_0 must have shape (4,) or ({num_samples}, 4), received {state_0.shape}.")

        trajectory_keys = jr.split(subkey, num_samples)
        return vmap(lambda k, s: self._sample_single(k, s, policy))(trajectory_keys, initial_states)
    

    def _sample_single(self, key: PRNGKey, state_0: Array, policy: Callable):
        """Generate one trajectory containing T states."""
        obs_key, step_key = jr.split(key)
        step_keys = jr.split(step_key, self.T - 1)

        obs_0 = self.observe(obs_key, state_0)
        def scan_step(state, key):
            policy_key, transition_key, observation_key = jr.split(key, 3)
            action = policy(policy_key, state)
            next_state = self.transition(transition_key, state, action)
            next_obs = self.observe(observation_key, next_state)
            return next_state, (next_state, next_obs, action)

        final_state, (states, observations, actions) = lax.scan(scan_step, state_0, step_keys)
        states = jnp.concatenate([state_0[None], states], axis=0)
        observations = jnp.concatenate([obs_0[None], observations], axis=0)
        return final_state, states, observations, actions

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
