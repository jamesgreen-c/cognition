from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from typing import Callable

from jax import Array, lax, vmap
from jax.random import PRNGKey

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


class MuJoCoSimulationEnvironment(ABC):
    """
    Base class for stateful MuJoCo environments.

    Each chronological replay buffer owns one persistent simulator.
    Calling sample() advances every simulator by exactly num_steps
    agent transitions.
    """

    jax_compatible = False

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
        self.envs = [None] * num_buffers

    @abstractmethod
    def _make_env(self, seed: int):
        """Construct one independently seeded simulator."""
        pass

    @abstractmethod
    def _state(self, env) -> Array:
        """Extract the ground-truth state from a simulator."""
        pass

    @abstractmethod
    def _observation(self, env) -> Array:
        """Extract the observation given to the agent."""
        pass

    @abstractmethod
    def _step_simulator(self, env, action: Array):
        """
        Apply an action for one native simulator step.

        Returns
        -------
        reward:    Reward for the native step.
        terminal:  Whether the episode has ended.
        """
        pass

    @abstractmethod
    def _action_bounds(self) -> tuple[Array, Array]:
        """Return the lower and upper action bounds."""
        pass

    def _reset(self, env):
        env.reset()
        return self._state(env), self._observation(env)

    def _step(self, env, action: Array):
        """
        Apply one agent action, including action repetition.
        """
        reward = 0.0
        terminal = False

        for _ in range(self.action_repeat):
            step_reward, terminal = self._step_simulator(env, action)
            reward += step_reward

            if terminal:
                break

        return (
            self._state(env),
            self._observation(env),
            jnp.asarray(reward),
            terminal,
        )

    def sample(
            self,
            key: PRNGKey,
            policy: Callable,
            num_samples: int,
            num_steps: int,
            state_0: Array | None = None,
        ):
        """
        Advance each persistent simulator by num_steps transitions.

        Parameters
        ----------
        key:          JAX random key.
        policy:       Callable with signature
                          action, log_prob = policy(key, observation)
        num_samples:  Number of chronological replay buffers.
        num_steps:    Number of new transitions per buffer.
        state_0:      Accepted for compatibility with the pure JAX
                      Environment interface. Persistent simulators retain
                      their own complete internal states.

        Returns
        -------
        final_states:  (M, *D)
        states:        (M, num_steps + 1, *D)
        observations:  (M, num_steps + 1, *O)
        actions:       (M, num_steps, *K)
        rewards:       (M, num_steps)
        flags:         (M, num_steps)
        log_probs:     (M, num_steps)
        """
        if num_samples != self.num_buffers:
            raise ValueError(f"Expected num_samples={self.num_buffers}, received {num_samples}.")

        if num_steps < 1:
            raise ValueError("num_steps must be positive.")

        buffer_keys = jr.split(key, num_samples)
        outputs = [
            self._sample_single(
                key=buffer_keys[m],
                policy=policy,
                num_steps=num_steps,
                buffer_idx=m,
            )
            for m in range(num_samples)
        ]

        return tuple(jnp.stack([output[i] for output in outputs]) for i in range(7))

    def _sample_single(
            self,
            key: PRNGKey,
            policy: Callable,
            num_steps: int,
            buffer_idx: int,
        ):
        """
        Advance one persistent simulator by num_steps transitions.
        """
        init_key, sample_key = jr.split(key)
        env = self.envs[buffer_idx]

        if env is None:
            seed = int(jr.randint(init_key, (), 0, 2 ** 31 - 1))
            env = self._make_env(seed)
            self.envs[buffer_idx] = env
            state, observation = self._reset(env)
        else:
            state = self._state(env)
            observation = self._observation(env)

        states = [state]
        observations = [observation]
        actions = []
        rewards = []
        flags = []
        log_probs = []

        step_keys = jr.split(sample_key, num_steps)

        for step_key in step_keys:
            action, log_prob = policy(step_key, observation)
            next_state, next_observation, reward, terminal = self._step(env, action)

            actions.append(jnp.asarray(action))
            rewards.append(reward)
            flags.append(jnp.asarray(1.0 - float(terminal)))
            log_probs.append(jnp.asarray(log_prob))

            if terminal:
                next_state, next_observation = self._reset(env)

            states.append(next_state)
            observations.append(next_observation)

            state = next_state
            observation = next_observation

        return (
            state,
            jnp.stack(states),
            jnp.stack(observations),
            jnp.stack(actions),
            jnp.stack(rewards),
            jnp.stack(flags),
            jnp.stack(log_probs),
        )

    def random_action(self, key: PRNGKey, observation: Array):
        """
        Sample uniformly over the valid action space.
        """
        lower, upper = self._action_bounds()
        action = jr.uniform(key, shape=lower.shape, minval=lower, maxval=upper)
        log_prob = -jnp.log(upper - lower).sum()
        return action, log_prob

    def preprocess_observation(self, observation: Array) -> Array:
        """Transform raw observations before passing them to a network."""
        return observation

    def evaluate(
        self,
        key: PRNGKey,
        policy: Callable,
        num_episodes: int = 10,
    ):
        """
        Evaluate a policy using fresh, independent complete episodes.

        The returned episode returns are undiscounted sums of all native
        rewards, including rewards accumulated by action repetition.
        """
        episode_keys = jr.split(key, num_episodes)
        episode_returns = []
        episode_lengths = []

        for episode_key in episode_keys:
            init_key, policy_key = jr.split(episode_key)
            seed = int(jr.randint(init_key, (), 0, 2 ** 31 - 1))

            env = self._make_env(seed)
            _, observation = self._reset(env)

            episode_return = 0.0
            episode_length = 0
            terminal = False

            while not terminal:
                policy_key, action_key = jr.split(policy_key)
                action, _ = policy(action_key, observation)
                _, observation, reward, terminal = self._step(env, action)

            episode_return += float(reward)
            episode_length += self.action_repeat

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        episode_returns = np.asarray(episode_returns)
        episode_lengths = np.asarray(episode_lengths)

        return {
            "mean_return": episode_returns.mean(),
            "std_return": episode_returns.std(ddof=1),
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
        }
