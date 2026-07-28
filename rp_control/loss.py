import jax 
import jax.numpy as jnp
import jax.random as jr

from jax import vmap, Array
from jax.random import PRNGKey

from rp_ssm.recognition.rpm import RPSSM

from rp_control.actor import Actor
from rp_control.critic import Critic
from rp_control.environment import Environment


class ActorCriticLoss:

    def __init__(
            self, 
            actor: Actor, 
            critic: Critic, 
            rpm: RPSSM,
            environment: Environment,
        ):

        self.actor = actor
        self.critic = critic
        self.rpm = rpm
        self.env = environment

    def init(self, key, data: tuple[Array], rpm_params):

        # for standardisation
        self.observation_mean = jnp.mean(data[0], axis=(0, 1))
        self.observation_std = jnp.std(data[0], axis=(0, 1))

        # for params
        actor_init_key, critic_init_key = jr.split(key)
        posterior = self.rpm.initial_distribution(rpm_params, (data[0][0], ))
        mean = posterior.params["mean"]

        # TODO test on observation
        _actor_params = self.actor.init(actor_init_key, mean)
        _critic_params = self.critic.init(critic_init_key, mean)

        return {"actor": _actor_params, "critic": _critic_params, "rpm": rpm_params}

    def grads(self, keys, params, posterior, env_state):
        """
        Calculate the gradients used for semi-gradient actor critic updates

        Parameters
        ----------
        keys:        (B, 2) RNG keys
        params:      NN params for rpm, actor and critic
        posterior:   DistParam: params{"mean": (B, D) batched distribution parameters}
        env_state:   (B, K) Batch of current environment states

        Returns
        -------
        grads:       Actor and critic gradients with a leading batch dimension
        aux:         Next posterior, environment states, rewards and values
        """
        # keys
        split_keys = vmap(lambda key: jr.split(key, 3))(keys)
        actor_keys = split_keys[:, 0]
        model_keys = split_keys[:, 1]
        obs_keys = split_keys[:, 2]

        mean = posterior.params["mean"]   # (B, D)

        # calculate one actor gradient and sample one action per sequence
        _actor = jax.value_and_grad(self.actor.apply, argnums=1, has_aux=True)
        (log_probs, actions), actor_grads = vmap(_actor, in_axes=(0, None, 0))(actor_keys, params["actor"], mean)

        # advance each environment independently
        next_env_state, reward = vmap(self.env.model)(model_keys, env_state, actions)
        observation = vmap(self.env.observe)(obs_keys, next_env_state)

        # apply RPM to observation to get next state representation
        next_posterior = self.get_posterior(params["rpm"], posterior, observation)
        next_mean = next_posterior.params["mean"]   # (B, D)

        # calculate estimated values for S and S'
        _critic = jax.value_and_grad(self.critic.apply, argnums=0)
        values, critic_grads = vmap(_critic, in_axes=(None, 0))(params["critic"], mean)
        next_values = self.critic.apply(params["critic"], next_mean)

        # pack and return 
        grads = {"actor": actor_grads, "critic": critic_grads}
        aux = next_posterior, next_env_state, reward, next_values, values
        return grads, aux

    def get_initial_distribution(self, params, observation):
        observation = self._standardise(observation)
        return self.rpm.initial_distribution(params["rpm"], (observation, ))
    
    def get_posterior(self, params, posterior, observation):
        observation = self._standardise(observation)
        return self.rpm.filter(params, posterior, (observation, ))

    def episode(self, key, params, N: int):
        """
        Generate N trajectories from the environment using the trained actor

        Parameters
        ----------
        key:     PRNGKey
        params:  RPM, actor and critic parameters
        N:       Number of independent trajectories

        Returns
        -------
        states:        (N, T, *state_shape) Environment states
        observations:  (N, T, *observation_shape) Observations
        actions:       (N, T - 1, *action_shape) Actions
        """

        def _episode(carry, inp):
            actor_keys, model_keys, obs_keys = inp
            _post, _env = carry

            # actor takes one action for each posterior mean
            mean = _post.params["mean"]   # (N, D)
            _, action = vmap(self.actor.apply, in_axes=(0, None, 0))(actor_keys, params["actor"], mean)

            # evolve N environment instances and observe
            _env_p1, _ = vmap(self.env.model)(model_keys, _env, action)
            _obs_p1 = vmap(self.env.observe)(obs_keys, _env_p1)

            # apply RPM to get states for next action
            _post_p1 = self.get_posterior(params["rpm"], _post, _obs_p1)

            return (_post_p1, _env_p1), (_env_p1, _obs_p1, action)

        actor_key, model_key, obs_key, init_key, init_obs_key = jr.split(key, 5)

        # keys have shape (T - 1, N, 2), so scan runs over time
        actor_keys = jr.split(actor_key, (self.env.T - 1) * N).reshape(self.env.T - 1, N, 2)
        model_keys = jr.split(model_key, (self.env.T - 1) * N).reshape(self.env.T - 1, N, 2)
        obs_keys = jr.split(obs_key, (self.env.T - 1) * N).reshape(self.env.T - 1, N, 2)

        # initialise N independent environments
        init_keys = jr.split(init_key, N)
        init_obs_keys = jr.split(init_obs_key, N)

        initial_states = vmap(self.env.initial_state)(init_keys)
        initial_obs = vmap(self.env.observe)(init_obs_keys, initial_states)
        initial_posterior = self.get_initial_distribution(params, initial_obs)

        # scan
        carry_0 = (initial_posterior, initial_states)
        inps = (actor_keys, model_keys, obs_keys)
        _, (env_states, observations, actions) = jax.lax.scan(_episode, carry_0, inps)

        # insert t=0 
        env_states = jnp.concatenate([initial_states[None], env_states], axis=0)
        observations = jnp.concatenate([initial_obs[None], observations], axis=0)

        # Dataset expects batch-major arrays: (N, T, ...)
        env_states = jnp.swapaxes(env_states, 0, 1)
        observations = jnp.swapaxes(observations, 0, 1)
        actions = jnp.swapaxes(actions, 0, 1)

        return env_states, observations, actions

    def _standardise(self, observation):
        return (observation - self.observation_mean) / self.observation_std