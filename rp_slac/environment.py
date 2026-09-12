from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as jr

from jax import Array, lax, vmap
from jax.random import PRNGKey
from jax.tree_util import tree_map


class JAXCarry(NamedTuple):
    """Batched state and its current observation."""

    state: Any
    observation: Array


class JAXEnvironment(ABC):
    """Base class for lightweight, fully JAX-native environments."""

    jax_compatible = True
    actor_history = 1

    def initial_carry(self, key: PRNGKey, num_samples: int) -> JAXCarry:
        """Initialise ``num_samples`` independent environments."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")

        state_key, observation_key = jr.split(key)
        states = vmap(self.initial_state)(jr.split(state_key, num_samples))
        observations = vmap(self.observe)(jr.split(observation_key, num_samples), states)
        return JAXCarry(state=states, observation=observations)

    def collect_step(
            self,
            key: PRNGKey,
            carry: JAXCarry,
            actions: Array,
        ) -> tuple[JAXCarry, Array, Array, Array]:
        """Advance every environment by exactly one transition."""
        num_samples = actions.shape[0]
        if carry.observation.shape[0] != num_samples:
            raise ValueError("The action and environment batches must have the same size.")

        model_key, reset_key, observation_key = jr.split(key, 3)
        next_states, rewards = vmap(self.model)(
            jr.split(model_key, num_samples),
            carry.state,
            actions,
        )

        episode_ends = vmap(self.is_terminal_state)(next_states)
        reset_states = vmap(self.initial_state)(jr.split(reset_key, num_samples))
        next_states = tree_map(
            lambda reset, stepped: self._select(episode_ends, reset, stepped),
            reset_states,
            next_states,
        )
        next_observations = vmap(self.observe)(jr.split(observation_key, num_samples), next_states)

        flags = 1.0 - episode_ends.astype(jnp.float32)
        next_carry = JAXCarry(state=next_states, observation=next_observations)
        return next_carry, next_observations, rewards, flags

    @staticmethod
    def actor_observation(carry: JAXCarry) -> Array:
        """Return the network input used by the online policy."""
        return carry.observation

    @staticmethod
    def current_observation(carry: JAXCarry) -> Array:
        """Return the raw observation stored in replay."""
        return carry.observation

    @staticmethod
    def _select(condition: Array, true_value: Array, false_value: Array) -> Array:
        condition = jnp.asarray(condition, dtype=bool)
        shape = condition.shape + (1,) * (false_value.ndim - condition.ndim)
        return jnp.where(condition.reshape(shape), true_value, false_value)

    def preprocess_observation(self, observation: Array) -> Array:
        """Transform raw observations before passing them to a network."""
        return observation

    def is_terminal_state(self, state: Array) -> Array:
        """Continuing environments use the default non-terminal result."""
        return jnp.asarray(False)

    @abstractmethod
    def transition(self, key: PRNGKey, state: Array, action: Array) -> Array:
        """Apply one environment-specific state transition."""
        pass

    @abstractmethod
    def model(self, key: PRNGKey, state: Array, action: Array) -> tuple[Array, Array]:
        """Sample ``(next_state, reward)`` from the environment model."""
        pass

    @abstractmethod
    def observe(self, key: PRNGKey, state: Array) -> Array:
        """Construct an observation from an environment state."""
        pass

    @abstractmethod
    def initial_state(self, key: PRNGKey) -> Array:
        """Sample one initial environment state."""
        pass

    @abstractmethod
    def random_action(self, key: PRNGKey, observation: Array) -> tuple[Array, Array]:
        """Sample ``(action, log_probability)`` for replay prefill."""
        pass


class MuJoCoCarry(NamedTuple):
    """State required to advance the fixed-size MJX environment batch."""

    state: Any
    frames: Array
    episode_steps: Array


class MuJoCoSimulationEnvironment(ABC):
    """Base class for one-step collection from MJX Playground environments."""

    jax_compatible = True

    def __init__(
            self,
            num_buffers: int,
            actor_history: int,
            action_repeat: int = 1,
            episode_length: int = 1000,
        ):
        if num_buffers < 1:
            raise ValueError("num_buffers must be positive.")
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive.")
        if actor_history < 1:
            raise ValueError("actor_history must be positive.")
        if episode_length < 1:
            raise ValueError("episode_length must be positive.")

        self.num_buffers = num_buffers
        self.action_repeat = action_repeat
        self.actor_history = actor_history
        self.episode_length = episode_length
        self.env = self._make_env()
        self._setup_observation()

    @abstractmethod
    def _make_env(self):
        """Construct the MuJoCo Playground environment."""
        pass

    @abstractmethod
    def _setup_observation(self):
        """Construct observation resources that must live outside JIT."""
        pass

    @abstractmethod
    def _observe(self, state) -> tuple[Any, Array]:
        """Render the fixed-size environment batch and thread its data token."""
        pass

    @abstractmethod
    def _action_bounds(self) -> tuple[Array, Array]:
        """Return the lower and upper action bounds."""
        pass

    def initial_state(self, key: PRNGKey, num_samples: int):
        return vmap(self.env.reset)(jr.split(key, num_samples))

    def initial_carry(self, key: PRNGKey, num_samples: int) -> MuJoCoCarry:
        """Reset and render the fixed-size MJX batch once."""
        if num_samples != self.num_buffers:
            raise ValueError(f"Expected num_samples={self.num_buffers}, received {num_samples}.")

        state = self.initial_state(key, num_samples)
        state, observation = self._observe(state)
        return MuJoCoCarry(
            state=state,
            frames=self._initial_actor_frames(observation),
            episode_steps=jnp.zeros((num_samples,), dtype=jnp.int32),
        )

    def _step(self, state, action: Array):
        """Apply one agent action, including action repetition."""

        def repeat_step(carry, _):
            state, terminated = carry
            stepped_state = self.env.step(state, action)

            reward = jnp.where(
                terminated,
                jnp.asarray(0.0, dtype=stepped_state.reward.dtype),
                stepped_state.reward,
            )
            state = self._select_state(terminated, state, stepped_state)
            terminated = jnp.logical_or(terminated, stepped_state.done.astype(bool))
            return (state, terminated), reward

        R = self.action_repeat
        carry_0 = (state, jnp.asarray(False))
        (next_state, terminal), rewards = lax.scan(repeat_step, carry_0, None, length=R)
        return next_state, rewards.sum(), terminal

    def collect_step(
            self,
            key: PRNGKey,
            carry: MuJoCoCarry,
            actions: Array,
        ) -> tuple[MuJoCoCarry, Array, Array, Array]:
        """Advance the fixed-size MJX batch by exactly one agent action."""
        next_state, rewards, terminals = vmap(self._step)(carry.state, actions)

        episode_steps = carry.episode_steps + self.action_repeat
        timeouts = episode_steps >= self.episode_length
        episode_ends = jnp.logical_or(terminals, timeouts)

        reset_state = self.initial_state(key, self.num_buffers)
        next_state = vmap(self._select_state)(episode_ends, reset_state, next_state)
        next_state, next_observation = self._observe(next_state)

        rolled_frames = self._append_actor_frame(carry.frames, next_observation)
        reset_frames = self._initial_actor_frames(next_observation)
        next_frames = self._select(episode_ends, reset_frames, rolled_frames)

        episode_steps = jnp.where(episode_ends, 0, episode_steps)
        flags = 1.0 - episode_ends.astype(jnp.float32)

        next_carry = MuJoCoCarry(state=next_state, frames=next_frames, episode_steps=episode_steps)
        return next_carry, next_observation, rewards, flags

    @classmethod
    def _select_state(cls, condition: Array, true_state, false_state):
        """Select complete Playground states, including MJX-Warp data."""
        data = false_state.data.where(condition, true_state.data)
        obs = tree_map(lambda true, false: cls._select(condition, true, false), true_state.obs, false_state.obs)
        reward = cls._select(condition, true_state.reward, false_state.reward)
        done = cls._select(condition, true_state.done, false_state.done)
        metrics = tree_map(lambda true, false: cls._select(condition, true, false), true_state.metrics, false_state.metrics)
        info = tree_map(lambda true, false: cls._select(condition, true, false), true_state.info, false_state.info)
        return false_state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    @staticmethod
    def _select(condition: Array, true_value: Array, false_value: Array) -> Array:
        condition = jnp.asarray(condition, dtype=bool)
        shape = condition.shape + (1,) * (false_value.ndim - condition.ndim)
        return jnp.where(condition.reshape(shape), true_value, false_value)

    def random_action(self, key: PRNGKey, observation: Array) -> tuple[Array, Array]:
        del observation
        lower, upper = self._action_bounds()
        action = jr.uniform(key, shape=lower.shape, minval=lower, maxval=upper)
        log_prob = -jnp.log(upper - lower).sum()
        return action, log_prob

    def preprocess_observation(self, observation: Array) -> Array:
        """Pixel observations are already float32 values in [0, 1]."""
        return observation

    def actor_observation(self, carry: MuJoCoCarry) -> Array:
        """Stack the actor's frame history along the channel axis."""
        return self._stack_actor_frames(carry.frames)

    @staticmethod
    def current_observation(carry: MuJoCoCarry) -> Array:
        """Return the most recent raw frame for replay storage."""
        return carry.frames[:, -1]

    def _initial_actor_frames(self, frame: Array) -> Array:
        return jnp.repeat(frame[:, None], self.actor_history, axis=1)

    @staticmethod
    def _append_actor_frame(frames: Array, frame: Array) -> Array:
        return jnp.concatenate((frames[:, 1:], frame[:, None]), axis=1)

    @staticmethod
    def _stack_actor_frames(frames: Array) -> Array:
        """Convert ``(N, F, H, W, C)`` into ``(N, H, W, F * C)``."""
        num_buffers, history, height, width, channels = frames.shape
        frames = jnp.transpose(frames, (0, 2, 3, 1, 4))
        return frames.reshape(num_buffers, height, width, history * channels)

