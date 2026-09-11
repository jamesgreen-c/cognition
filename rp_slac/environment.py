from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from typing import Callable

from jax import Array, lax, vmap
from jax.random import PRNGKey
from jax.tree_util import tree_map

from abc import ABC, abstractmethod
from typing import Callable


class JAXEnvironment(ABC):

    jax_compatible = True

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

    def preprocess_observation(self, observation: Array) -> Array:
        """Transform raw observations before passing them to a network."""
        return observation

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
    def random_action(self, key: PRNGKey, observation: Array):
        """Sample a valid action for replay-buffer initialisation."""
        pass


from typing import Any, NamedTuple

class MuJoCoCarry(NamedTuple):
    state: Any
    observation: Array
    
class MuJoCoSimulationEnvironment(ABC):
    """
    Base class for JAX-compatible MuJoCo Playground environments.

    MuJoCo physics is vectorised over replay buffers while trajectory
    generation is scanned over time. This ordering is required by the
    fixed-batch MJX-Warp renderer.
    """

    jax_compatible = True

    def __init__(
            self,
            num_buffers: int,
            action_repeat: int = 1,
        ):
        if num_buffers < 1:
            raise ValueError("num_buffers must be positive.")

        if action_repeat < 1:
            raise ValueError("action_repeat must be positive.")

        self.num_buffers = num_buffers
        self.action_repeat = action_repeat
        self.env = self._make_env()
        self._setup_observation()

    @abstractmethod
    def _make_env(self):
        """Construct the MuJoCo Playground environment."""
        pass

    @abstractmethod
    def _setup_observation(self):
        """Construct any observation resources required outside JIT."""
        pass

    @abstractmethod
    def _state(self, state) -> Array:
        """Extract the physical state stored in the replay buffer."""
        pass

    @abstractmethod
    def _observe(self, state):
        """
        Construct batched observations.

        Returns
        -------
        state:        Environment state containing any updated execution token.
        observation:  Batched observations with leading dimension num_buffers.
        """
        pass

    @abstractmethod
    def _action_bounds(self) -> tuple[Array, Array]:
        """ Return the lower and upper action bounds. """
        pass

    def initial_state(self, key: PRNGKey, num_samples: int):
        return vmap(self.env.reset)(jr.split(key, num_samples))

    def _step(self, state, action: Array):
        """ Apply one agent action, including action repetition. """

        def repeat_step(carry, _):
            state, terminated = carry
            stepped_state = self.env.step(state, action)

            reward = jnp.where(terminated, jnp.asarray(0.0, dtype=stepped_state.reward.dtype), stepped_state.reward)
            state = self._select_state(terminated, state, stepped_state)
            terminated = jnp.logical_or(terminated, stepped_state.done.astype(bool))
            return (state, terminated), reward

        terminal_0 = jnp.asarray(False)
        (next_state, terminal), rewards = lax.scan(repeat_step,
                                                   (state, terminal_0),
                                                   None,
                                                   length=self.action_repeat)
        
        return next_state, rewards.sum(), terminal

    def sample(
            self,
            key: PRNGKey,
            policy: Callable,
            num_samples: int,
            num_steps: int,
            state_0: Array | None = None,
        ):
        """
        Advance a batch of environments by num_steps transitions.
        state_0 is the batched environment state returned by the preceding call. When omitted, a fresh batch is generated.
        """
        if num_samples != self.num_buffers:
            raise ValueError(f"Expected num_samples={self.num_buffers}, received {num_samples}.")
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")

        init_key, sample_key = jr.split(key)
        state_0 = self.initial_state(init_key, num_samples) if state_0 is None else state_0 

        return self._sample(
            sample_key,
            state_0,
            policy,
            num_samples,
            num_steps,
        )

    def _sample(
            self,
            key: PRNGKey,
            state_0,
            policy: Callable,
            num_samples: int,
            num_steps: int,
        ):
        step_keys = jr.split(key, num_steps)

        # observe the initial state once and carry the observation.
        state_0, observation_0 = self._observe(state_0)
        initial_physical_state = self._state(state_0)

        def scan_step(carry, key):
            state, observation = carry
            policy_key, reset_key = jr.split(key)

            policy_keys = jr.split(policy_key, num_samples)
            actions, log_probs = vmap(policy)(policy_keys, observation)

            # vectorise the pure Playground physics transitions.
            next_state, rewards, terminals = vmap(self._step)(state, actions)

            # reset terminated environments before constructing the next observation
            reset_state = self.initial_state(reset_key, num_samples)
            next_state = vmap(self._select_state)(terminals, reset_state, next_state)

            # rendering operates on the complete environment batch.
            next_state, next_observation = self._observe(next_state)

            flags = 1.0 - terminals.astype(jnp.float32)
            outputs = (self._state(next_state), next_observation, actions, rewards, flags, log_probs)
            return (next_state, next_observation), outputs

        (final_state, _), outputs = lax.scan(scan_step, (state_0, observation_0), step_keys)
        states, observations, actions, rewards, flags, log_probs = outputs

        states = jnp.concatenate([initial_physical_state[None], states], axis=0)
        observations = jnp.concatenate([observation_0[None], observations], axis=0)

        # convert (T, M, ...) into the replay-buffer layout (M, T, ...).
        states = jnp.swapaxes(states, 0, 1)
        observations = jnp.swapaxes(observations, 0, 1)
        actions = jnp.swapaxes(actions, 0, 1)
        rewards = jnp.swapaxes(rewards, 0, 1)
        flags = jnp.swapaxes(flags, 0, 1)
        log_probs = jnp.swapaxes(log_probs, 0, 1)

        return (final_state, states, observations, actions, rewards, flags, log_probs)

    @classmethod
    def _select_state(cls, condition: Array, true_state, false_state):
        """
        Select complete MJX states.
        Data.where is required for compatibility with MJX-Warp's private implementation fields.
        """
        data = false_state.data.where(condition, true_state.data)
        # condition = jnp.asarray(condition, dtype=bool)
        # data_condition = condition if condition.ndim == 0 else condition[..., None]
        # data = false_state.data.where(data_condition, true_state.data)

        obs = tree_map(lambda true, false: cls._select(condition, true, false), true_state.obs, false_state.obs)
        reward = cls._select(condition, true_state.reward, false_state.reward)
        done = cls._select(condition, true_state.done, false_state.done)
        metrics = tree_map(lambda true, false: cls._select(condition, true, false), true_state.metrics, false_state.metrics)
        info = tree_map(lambda true, false: cls._select(condition, true, false), true_state.info, false_state.info)
        return false_state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    @staticmethod
    def _select(condition: Array, true_value: Array, false_value: Array):
        condition = jnp.asarray(condition, dtype=bool)
        shape = condition.shape + (1,) * (false_value.ndim - condition.ndim)
        return jnp.where(condition.reshape(shape), true_value, false_value)

    def random_action(self, key: PRNGKey, observation: Array):
        # del observation

        lower, upper = self._action_bounds()
        action = jr.uniform(key, shape=lower.shape, minval=lower, maxval=upper)
        log_prob = -jnp.log(upper - lower).sum()
        return action, log_prob

    def preprocess_observation(self, observation: Array) -> Array:
        """ Pixel observations returned by subclasses are already float32 values in [0, 1]. """
        return observation