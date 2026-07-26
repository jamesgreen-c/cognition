"""
Trainer job:

1. Needs access to data generation process given an action and current state
2. Needs access to reward sampling process given resultant state


3. define a jitted training step that:
    - applies the RPM to observations to get latents
    - applies actor to latents to generate next state
    - applies critic to next states to get values estimates
    - samples true reward of next state
    - applies semi-gradient updates to both the actor and critic networks
    - returns (action, next_state, reward) for tracking and next training step

4. needs a config for:
    - alphas (step sizes for semi-gradient steps) not the same as learning rates
    - batch size
    - debug
    - optimiser: adam
    - num-iter: int = 1?
"""

import optax

import jax
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm

from jax import Array
from jax.lax import stop_gradient as stop_grad
from jax.tree_util import tree_map

from rp_control.loss import ActorCriticLoss
from rp_control.config import Config
from rp_control.environment import Environment


class ActorCritic:

    opt_states: dict[str, optax.OptState]
    itr: int

    def __init__(
            self,
            loss: ActorCriticLoss,
            config: Config,
            environment: Environment
        ):

        self.config = config
        self.loss = loss

        self.actor_opt = config.actor.build()
        self.critic_opt = config.critic.build()

        self.env = environment

        self.itr = 0

    def train_step(
            self,
            keys: Array,
            params: dict,
            opt_states: dict,
            traces: dict,
            discount: Array,
            posteriors,
            env_states: Array
        ):
        """
        Applies one actor-critic update across B parallel sequences

        Each sequence produces its own TD error and eligibility traces. Their
        delta-weighted traces are averaged before updating the shared parameters.

        Parameters
        ----------
        keys:        (B, 2) PRNG keys for the parallel environment transitions
        params:      shared actor, critic and RPSSM parameters
        opt_states:  shared actor and critic optimiser states
        traces:      (B, *parameter_shape) batched actor and critic eligibility traces
        discount:    scalar cumulative actor discount
        posteriors:  batched RPSSM filtering distributions
        env_states:  (B, *state_shape) current environment states

        Other
        -----
        rewards:     (B,) rewards from the transitions
        deltas:      (B,) one-step TD errors
        """

        # apply actor and critic independently to each sequence using shared parameters
        grads, aux = jax.vmap(self.loss.grads, in_axes=(0, None, 0, 0))(keys, params, posteriors, env_states)
        next_posteriors, next_env_states, rewards, next_values, values = aux

        # continuing discounted eligibility trace updates
        deltas = stop_grad(rewards + self.config.gamma * next_values - values)
        traces = self._update_traces(traces, grads, discount)

        # average delta-weighted traces across sequences
        actor_loss = self._mean_trace_update(traces["actor"], deltas)
        critic_loss = self._mean_trace_update(traces["critic"], deltas)

        # apply one shared semi-gradient update
        new_params, new_opt_states = self._update_params(actor_loss=actor_loss,
                                                         critic_loss=critic_loss,
                                                         params=params,
                                                         opt_states=opt_states)

        return new_params, new_opt_states, next_posteriors, next_env_states, traces, rewards, deltas

    def fit(self, rpm_params):
        """
        Trains the actor and critic using B parallel rollout sequences

        Each iteration initialises B independent sequences and applies online
        actor-critic updates over T - 1 timesteps. Actor, critic and RPSSM
        parameters are shared across sequences, while environment states,
        posteriors and eligibility traces are maintained separately.

        Parameters
        ----------
        rpm_params:  trained RPSSM parameters held fixed during actor-critic training

        Returns
        -------
        params:  trained actor and critic parameters with unchanged RPSSM parameters
        """

        train_step = jax.jit(self.train_step) if not self.config.debug else self.train_step
        B = self.config.batch_size

        # init shared params
        key, init_key, obs_key, param_key = jr.split(jr.PRNGKey(self.config.seed), 4)
        dummy_state = self.env.initial_state(init_key)
        dummy_obs = self.env.observe(obs_key, dummy_state)
        self.params = self.loss.init(param_key, dummy_obs, rpm_params)
        self.opt_states = {
            "actor": self.actor_opt.init(self.params["actor"]),
            "critic": self.critic_opt.init(self.params["critic"])
        }

        pbar = tqdm(range(self.config.num_iter))
        for self.itr in pbar:

            # sample B starting states and observations
            key, init_key, obs_key, episode_key = jr.split(key, 4)
            init_keys = jr.split(init_key, B)
            obs_keys = jr.split(obs_key, B)

            initial_states = jax.vmap(self.env.initial_state)(init_keys)
            initial_obs = jax.vmap(self.env.observe)(obs_keys, initial_states)

            # apply RPM to initial data
            posteriors = jax.vmap(
                self.loss.get_initial_distribution, in_axes=(None, 0)
            )(self.params, initial_obs)

            # separate eligibility traces for every sequence
            discount = jnp.array(1.0)
            traces = {
                "actor": tree_map(lambda p: jnp.zeros((B,) + p.shape), self.params["actor"]),
                "critic": tree_map(lambda p: jnp.zeros((B,) + p.shape), self.params["critic"])
            }

            def _body(carry, _keys):
                _params, _opt_states, _posts, _envs, _traces, _discount = carry

                # apply batched RL step
                outs = train_step(_keys, _params, _opt_states, _traces, _discount, _posts, _envs)
                _params_p1, _opt_states_p1, _posts_p1, _envs_p1, _traces_p1, rewards, deltas = outs

                # update actor discount
                _discount_p1 = self.config.gamma * _discount

                # pack next carry
                carry_p1 = (_params_p1, _opt_states_p1, _posts_p1, _envs_p1, _traces_p1, _discount_p1)
                return carry_p1, (rewards, deltas)

            # initial carry and inputs
            carry_0 = (self.params, self.opt_states, posteriors, initial_states, traces, discount)
            keys = jr.split(episode_key, (self.env.T - 1, B))

            # rewards, deltas: (T - 1, B, ...)
            carry, (rewards, deltas) = jax.lax.scan(_body, carry_0, keys)
            self.params, self.opt_states = carry[:2]

            mean_return = jnp.mean(jnp.sum(rewards, axis=0))
            mean_delta = jnp.mean(jnp.abs(deltas))
            pbar.set_postfix(reward=float(mean_return), delta=float(mean_delta))

        return self.params

    def generate(self, key, N: int):
        """
        Generates N trajectories using the trained actor

        Parameters
        ----------
        key:   PRNGkey used to generate the trajectories
        N:     Number of independent trajectories

        Returns
        -------
        data:  N generated trajectories containing states, observations and actions
        """
        keys = jr.split(key, N)
        return jax.vmap(lambda _key: self.loss.episode(_key, self.params))(keys)

    def _update_trace(self, decay_rate, trace, gradient):
        """
        Updates one collection of eligibility traces

        trace_t = gamma * lambda * trace_{t-1} + gradient_t

        Parameters
        ----------
        decay_rate:  (,) Eligibility-trace decay rate lambda
        trace:       (B, *parameter_shape) Previous eligibility traces
        gradient:    (B, *parameter_shape) Current actor or critic gradients

        Returns
        -------
        trace:        (B, *parameter_shape) Updated eligibility traces
        """
        return tree_map(
            lambda z, g: self.config.gamma * decay_rate * z + g,
            trace, gradient
        )

    def _update_traces(self, traces, grads, discount):
        """
        Updates the actor and critic eligibility traces

        The actor gradients are weighted by the cumulative discount before entering
        the trace. The actor and critic traces use their respective decay rates.

        Parameters
        ----------
        traces:    (B, *parameter_shape) Current batched actor & critic eligibility traces
        grads:     (B, *parameter_shape) Batched actor & critic gradients
        discount:  (, ) Scalar cumulative actor discount

        Returns
        -------
        traces: updated batched actor and critic eligibility traces
        """

        # discount actor gradient in trace update
        actor_grad = tree_map(lambda g: discount * g, grads["actor"])
        trace_actor = self._update_trace(
            self.config.actor_trace_decay, traces["actor"], actor_grad
        )

        # update critic trace with regular grad
        trace_critic = self._update_trace(
            self.config.critic_trace_decay, traces["critic"], grads["critic"]
        )

        return {"actor": trace_actor, "critic": trace_critic}

    def _mean_trace_update(self, traces, deltas):
        """
        Returns the mean update across B parallel sequences

        update = - sum_b(delta_b * trace_b) / B

        Parameters
        ----------
        traces:  (B, *parameter_shape) eligibility traces
        deltas:  (B,) one-step TD errors for actor-critic learning

        Returns
        -------
        update:  (*parameter_shape) mean parameter update
        """

        def _mean(trace):
            shape = (deltas.shape[0],) + (1,) * (trace.ndim - 1)
            return -jnp.mean(deltas.reshape(shape) * trace, axis=0)

        return tree_map(_mean, traces)

    def _update_params(self, actor_loss, critic_loss, params, opt_states):
        """
        Applies one shared optimiser update to the actor and critic

        The RPSSM parameters are carried forward unchanged.

        Parameters
        ----------
        actor_loss:   Actor update PyTree with the same structure as actor parameters
        critic_loss:  Critic update PyTree with the same structure as critic parameters
        params:       Current actor, critic and RPSSM parameters
        opt_states:   Current actor and critic optimiser states

        Returns
        -------
        new_params:      Updated actor and critic parameters with unchanged RPSSM parameters
        new_opt_states:  Updated actor and critic optimiser states
        """

        actor_update, actor_opt_state = self.actor_opt.update(
            actor_loss, opt_states["actor"], params["actor"]
        )
        critic_update, critic_opt_state = self.critic_opt.update(
            critic_loss, opt_states["critic"], params["critic"]
        )

        actor_params = optax.apply_updates(params["actor"], actor_update)
        critic_params = optax.apply_updates(params["critic"], critic_update)

        new_opt_states = {
            "actor": actor_opt_state,
            "critic": critic_opt_state
        }
        new_params = {
            "actor": actor_params,
            "critic": critic_params,
            "rpm": params["rpm"]
        }

        return new_params, new_opt_states